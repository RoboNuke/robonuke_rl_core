"""BlockAdamW's fused CUDA step (``adamw_tensor_`` through torch.compile) against the eager one and
against N torch.optim.AdamW instances: a stepped agent agrees to floating-point reordering, a frozen
agent is bit-identical, the compile happens at construction and the step makes no host sync."""

from __future__ import annotations

import warnings

import pytest
import torch

from robonuke_rl_core.optim import BlockAdamW, adamw_tensor_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused step is the CUDA path")

NUM_AGENTS = 5
SHAPES = [(NUM_AGENTS, 10, 1024, 38), (NUM_AGENTS, 10, 1024), (NUM_AGENTS, 1024, 1024), (NUM_AGENTS, 1024), (NUM_AGENTS, 1, 1024)]


def stacked(seed: int = 0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(device) * 0.05) for s in SHAPES]


def gradients(params, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g).to(p.device)


def test_fused_matches_eager_and_a_frozen_agent_is_bit_identical():
    fused_params, eager_params = stacked(), stacked()
    fused = BlockAdamW(fused_params, NUM_AGENTS, lr=1e-3, weight_decay=0.01)
    eager = BlockAdamW(eager_params, NUM_AGENTS, lr=1e-3, weight_decay=0.01)
    assert fused.fused and eager.fused
    eager.fused = False  # the same tensors through the eager per-tensor path
    for a, b in zip(fused_params, eager_params):
        assert torch.equal(a, b)  # the compile-time dry run changed nothing
    for step in range(20):
        gradients(fused_params, step)
        gradients(eager_params, step)
        keep = torch.tensor([True, step % 2 == 0, step < 5, False, True], device="cuda")
        fused.step(keep)
        eager.step(keep)
        for a, b in zip(fused_params, eager_params):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        for a, b in zip(fused.exp_avg + fused.exp_avg_sq, eager.exp_avg + eager.exp_avg_sq):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-9)
        assert torch.equal(fused.t, eager.t)
    # agent 3 never stepped: untouched bit for bit, by both paths
    fresh = stacked()
    for a, b, c in zip(fused_params, eager_params, fresh):
        assert torch.equal(a[3], c[3]) and torch.equal(b[3], c[3])
    assert fused.t.tolist() == [20, 10, 5, 0, 20]


def test_fused_matches_torch_adamw_per_agent():
    params = stacked()
    block = BlockAdamW(params, NUM_AGENTS, lr=3e-4, weight_decay=0.02)
    refs = [[torch.nn.Parameter(p[a].detach().clone()) for p in params] for a in range(NUM_AGENTS)]
    opts = [torch.optim.AdamW(r, lr=3e-4, weight_decay=0.02) for r in refs]
    for step in range(10):
        gradients(params, step)
        for a in range(NUM_AGENTS):
            for r, p in zip(refs[a], params):
                r.grad = p.grad[a].clone()
            opts[a].step()
        block.step()
    for a in range(NUM_AGENTS):
        for r, p in zip(refs[a], params):
            torch.testing.assert_close(p[a], r, rtol=1e-6, atol=1e-7)


def test_the_step_makes_no_host_sync_and_captures_in_a_cuda_graph():
    params = stacked()
    block = BlockAdamW(params, NUM_AGENTS, lr=1e-3, weight_decay=0.01)
    gradients(params, 0)
    keep = torch.tensor([True, False, True, True, False], device="cuda")
    block.step(keep)  # the first masked call after the dry run
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(1)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            block.step(keep)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert [str(w.message) for w in seen if "synchroniz" in str(w.message).lower()] == []
    # a captured step replays: the same result as an eager step from the same state
    before = [p.detach().clone() for p in params]
    moments = [m.clone() for m in block.exp_avg + block.exp_avg_sq]
    t = block.t.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        block.step(keep)  # warm-up on a side stream, as StepGraphs does
    torch.cuda.current_stream().wait_stream(side)
    for p, b in zip(params, before):
        p.data.copy_(b)
    for m, saved in zip(block.exp_avg + block.exp_avg_sq, moments):
        m.copy_(saved)
    block.t.copy_(t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block.step(keep)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [p.detach().clone() for p in params]
    for p, b in zip(params, before):
        p.data.copy_(b)
    for m, saved in zip(block.exp_avg + block.exp_avg_sq, moments):
        m.copy_(saved)
    block.t.copy_(t)
    block.step(keep)
    for p, r in zip(params, replayed):
        assert torch.equal(p, r)


def test_cpu_path_stays_eager_and_the_eager_function_is_the_reference():
    params = [torch.nn.Parameter(torch.randn(NUM_AGENTS, 7))]
    block = BlockAdamW(params, NUM_AGENTS, lr=1e-3)
    assert not block.fused
    p = torch.randn(NUM_AGENTS, 4, 3)
    g, m, v = torch.randn_like(p), torch.randn_like(p), torch.rand_like(p)
    want = p.clone()
    ones = torch.ones(NUM_AGENTS)
    adamw_tensor_(want, g, m.clone(), v.clone(), ones * 0.999, ones * 1e-3, ones, torch.ones(NUM_AGENTS, dtype=torch.bool), 0.9, 0.999, 1e-8)
    assert not torch.equal(want, p)
    same = p.clone()
    adamw_tensor_(same, g, m.clone(), v.clone(), ones * 0.999, ones * 1e-3, ones, torch.zeros(NUM_AGENTS, dtype=torch.bool), 0.9, 0.999, 1e-8)
    assert torch.equal(same, p)


def test_a_disabled_dynamo_is_refused_instead_of_a_silent_eager_step(monkeypatch):
    import robonuke_rl_core.optim as optim

    monkeypatch.setattr(optim, "_fused_adamw_tensor_", None)
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    with pytest.raises(RuntimeError, match="Dynamo is disabled"):
        BlockAdamW([torch.nn.Parameter(torch.zeros(NUM_AGENTS, 3, device="cuda"))], NUM_AGENTS, lr=1e-3)

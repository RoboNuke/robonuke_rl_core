"""Per-agent gradient clipping and the slicing helpers."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.models.block_utils import (
    assign_block_slice,
    clip_grad_norm_per_agent,
    merge_optimizer_states,
    slice_block_state_dict,
    slice_optimizer_state,
    step_with_frozen_agents,
)

NUM_AGENTS = 3


class BlockOnly(nn.Module):
    """Every parameter carries the leading agent dim."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_agents, 2, 2))
        self.bias = nn.Parameter(torch.zeros(num_agents, 2))


class WithParameterList(BlockOnly):
    """A per-agent ParameterList, the other form clipping accepts."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        super().__init__(num_agents)
        self.log_std = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 2)) for _ in range(num_agents)]
        )


class WithShared(BlockOnly):
    """A parameter shared by every agent: clipping must refuse it."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        super().__init__(num_agents)
        self.shared = nn.Parameter(torch.zeros(5))


def set_grads(module: nn.Module, per_agent_scale) -> None:
    for name, param in module.named_parameters():
        grad = torch.ones_like(param)
        if param.dim() >= 1 and param.shape[0] == NUM_AGENTS:
            for agent, scale in enumerate(per_agent_scale):
                grad[agent] *= scale
        else:  # a ParameterList entry: its index is its agent
            agent = int(name.split(".")[-1]) if name.split(".")[-1].isdigit() else 0
            grad *= per_agent_scale[agent]
        param.grad = grad


def test_only_the_agent_over_the_limit_is_clipped():
    module = BlockOnly()
    set_grads(module, [10.0, 0.01, 10.0])
    before = {n: p.grad.clone() for n, p in module.named_parameters()}

    norms = clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=1.0)
    assert norms.shape == (NUM_AGENTS,)
    # agent 1's norm is below the limit, so its gradient is untouched
    for name, param in module.named_parameters():
        assert torch.equal(param.grad[1], before[name][1])
    # the others were scaled down to exactly the limit
    for agent in (0, 2):
        clipped = torch.cat(
            [p.grad[agent].reshape(-1) for _, p in module.named_parameters()]
        )
        assert float(clipped.norm()) == pytest.approx(1.0, abs=1e-5)
        assert float(norms[agent]) > 1.0


def test_norms_are_reported_pre_clip_and_per_agent():
    module = BlockOnly()
    set_grads(module, [3.0, 1.0, 2.0])
    norms = clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=0.0)  # 0 = report only
    expected = torch.tensor([3.0, 1.0, 2.0]) * torch.tensor(6.0).sqrt()  # 6 elements per agent
    assert torch.allclose(norms, expected, atol=1e-5)


def test_a_per_agent_parameter_list_is_accepted():
    module = WithParameterList()
    set_grads(module, [10.0, 0.0, 0.0])
    norms = clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=1.0)
    assert float(norms[0]) > 1.0 and float(norms[1]) == 0.0
    # only agent 0's own ParameterList entry was scaled
    assert float(module.log_std[0].grad.abs().max()) < 10.0
    assert float(module.log_std[1].grad.abs().max()) == 0.0


def test_a_shared_parameter_raises():
    module = WithShared()
    set_grads(module, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError) as err:
        clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=1.0)
    message = str(err.value)
    assert "shared" in message and "shared parameter" in message


def test_clipping_survives_a_deepcopy():
    """Ownership comes from the module tree, not from attributes a deepcopy would drop."""
    import copy

    module = copy.deepcopy(WithParameterList())
    set_grads(module, [10.0, 1.0, 1.0])
    norms = clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=1.0)
    assert float(norms[0]) > float(norms[1])


def test_state_dict_slice_and_assign_round_trip():
    source = BlockOnly()
    with torch.no_grad():
        source.weight.normal_()
        source.bias.normal_()
    sliced = slice_block_state_dict(source, 2, NUM_AGENTS)
    assert sliced["weight"].shape == (2, 2)

    target = BlockOnly()
    assign_block_slice(target, 0, NUM_AGENTS, sliced)
    assert torch.equal(target.weight[0], source.weight[2])
    assert torch.equal(target.bias[0], source.bias[2])
    # the other slots are untouched
    assert torch.equal(target.weight[1], torch.zeros_like(target.weight[1]))


def test_optimizer_state_slice_and_merge_round_trip():
    module = BlockOnly()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    set_grads(module, [1.0, 2.0, 3.0])
    optimizer.step()

    per_agent = [
        slice_optimizer_state(optimizer.state_dict(), agent, NUM_AGENTS)
        for agent in range(NUM_AGENTS)
    ]
    merged = merge_optimizer_states(per_agent, NUM_AGENTS)
    for param_id, state in optimizer.state_dict()["state"].items():
        for key, value in state.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == NUM_AGENTS:
                assert torch.equal(merged["state"][param_id][key], value)


def test_merge_needs_every_agent():
    module = BlockOnly()
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    sliced = slice_optimizer_state(optimizer.state_dict(), 0, NUM_AGENTS)
    with pytest.raises(ValueError):
        merge_optimizer_states([sliced], NUM_AGENTS)
    with pytest.raises(KeyError):
        merge_optimizer_states([{"state": {}, "param_groups": []}], 1)


# ------------------------------------------------------------------ step_with_frozen_agents
def _random_grads(module: nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for param in module.parameters():
        param.grad = torch.randn(param.shape, generator=gen)


def _init_weights(module: nn.Module) -> None:
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen))


def _agent_view(module: nn.Module, optimizer, agent: int) -> dict:
    """One agent's weights and Adam moments (block slice or its own ParameterList entry)."""
    out = {}
    for name, param in module.named_parameters():
        if param.shape[0] == NUM_AGENTS:
            index = agent
        elif name.startswith("log_std."):
            if int(name.split(".")[1]) != agent:
                continue
            index = slice(None)
        state = optimizer.state.get(param, {})
        out[name] = param.detach()[index].clone()
        for key in ("exp_avg", "exp_avg_sq"):
            out[f"{name}/{key}"] = state[key][index].clone() if key in state else None
    return out


@pytest.mark.parametrize("warm_steps", [0, 3])  # 0: the optimizer state does not exist yet
def test_a_frozen_agent_comes_out_bit_identical(warm_steps):
    module = WithParameterList()
    _init_weights(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2, weight_decay=0.1)
    for step in range(warm_steps):
        _random_grads(module, seed=step)
        optimizer.step()

    reference = WithParameterList()
    reference.load_state_dict(module.state_dict())
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2, weight_decay=0.1)
    # deepcopy: load_state_dict keeps same-device tensors by reference, so the two
    # optimizers would otherwise share their moment tensors
    reference_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    frozen_before = _agent_view(module, optimizer, 1)
    _random_grads(module, seed=99)
    _random_grads(reference, seed=99)
    active = torch.tensor([True, False, True])
    step_with_frozen_agents(optimizer, module, NUM_AGENTS, active)
    reference_opt.step()

    frozen_after = _agent_view(module, optimizer, 1)
    for key, before in frozen_before.items():
        if before is None:  # moments created by this step: the frozen agent's stay at zero
            assert torch.equal(frozen_after[key], torch.zeros_like(frozen_after[key])), key
        else:
            assert torch.equal(before, frozen_after[key]), key
    # the active agents took exactly a plain AdamW step
    for agent in (0, 2):
        got, want = _agent_view(module, optimizer, agent), _agent_view(reference, reference_opt, agent)
        for key in want:
            assert torch.equal(got[key], want[key]), (agent, key)


def test_frozen_step_still_advances_the_step_counter():
    module = BlockOnly()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
    for step in range(3):
        _random_grads(module, seed=step)
        step_with_frozen_agents(optimizer, module, NUM_AGENTS, torch.zeros(NUM_AGENTS, dtype=torch.bool))
    for param in module.parameters():
        assert float(optimizer.state[param]["step"]) == 3.0
        assert torch.equal(param.detach(), torch.zeros_like(param))  # every agent frozen


def test_frozen_step_needs_every_grad():
    module = BlockOnly()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
    with pytest.raises(ValueError):
        step_with_frozen_agents(optimizer, module, NUM_AGENTS, torch.ones(NUM_AGENTS, dtype=torch.bool))
    _random_grads(module, seed=0)
    with pytest.raises(ValueError):
        step_with_frozen_agents(optimizer, module, NUM_AGENTS, torch.ones(2, dtype=torch.bool))

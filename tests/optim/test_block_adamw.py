"""BlockAdamW must be N independent torch.optim.AdamW instances, exactly.

torch.optim.AdamW is the reference: the equivalence tests below are the gate. If one fails
the optimizer is wrong — fix it, do not loosen the tolerance.
"""

from __future__ import annotations

import copy

import pytest
import torch

from robonuke_rl_core.optim import BlockAdamW, lr_at

NUM_AGENTS = 4
D_IN, D_OUT = 5, 3
LR = 3.0e-4
WEIGHT_DECAY = 0.02
STEPS = 50


def plain_model(dtype=torch.float32) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(D_IN, 7, dtype=dtype), torch.nn.Linear(7, D_OUT, dtype=dtype)
    )


def reference_and_block(dtype=torch.float32, lr: float = LR, weight_decay: float = WEIGHT_DECAY):
    """N plain models with their own AdamW, and one stacked copy with BlockAdamW."""
    torch.manual_seed(0)
    models = [plain_model(dtype) for _ in range(NUM_AGENTS)]
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay) for m in models
    ]
    stacked = [
        torch.nn.Parameter(torch.stack([p.detach().clone() for p in group]))
        for group in zip(*[list(m.parameters()) for m in models])
    ]
    block = BlockAdamW(stacked, NUM_AGENTS, lr=lr, weight_decay=weight_decay)
    return models, optimizers, stacked, block


def same_gradients(models, stacked, scale: float = 1.0, seed: int = 0):
    """Give every agent its own gradient, identical on both sides."""
    generator = torch.Generator().manual_seed(seed)
    for agent, model in enumerate(models):
        for index, param in enumerate(model.parameters()):
            grad = torch.randn(param.shape, generator=generator, dtype=param.dtype) * scale
            param.grad = grad.clone()
            if stacked[index].grad is None:
                stacked[index].grad = torch.zeros_like(stacked[index])
            stacked[index].grad[agent] = grad


def compare(models, stacked, block, optimizers, rtol: float, atol: float = 0.0) -> None:
    for agent, model in enumerate(models):
        for index, param in enumerate(model.parameters()):
            torch.testing.assert_close(
                stacked[index][agent], param, rtol=rtol, atol=atol,
                msg=lambda m, a=agent, i=index: f"weights, agent {a}, param {i}: {m}",
            )
            state = optimizers[agent].state[param]
            if state:
                torch.testing.assert_close(
                    block.exp_avg[index][agent], state["exp_avg"], rtol=rtol, atol=atol,
                    msg=lambda m, a=agent, i=index: f"exp_avg, agent {a}, param {i}: {m}",
                )
                torch.testing.assert_close(
                    block.exp_avg_sq[index][agent], state["exp_avg_sq"], rtol=rtol, atol=atol,
                    msg=lambda m, a=agent, i=index: f"exp_avg_sq, agent {a}, param {i}: {m}",
                )


# ------------------------------------------------------------------ 1. equivalence
@pytest.mark.parametrize("dtype, rtol", [(torch.float32, 1.0e-6), (torch.float64, 1.0e-12)])
def test_matches_n_torch_adamw_instances(dtype, rtol):
    models, optimizers, stacked, block = reference_and_block(dtype)
    for step in range(STEPS):
        same_gradients(models, stacked, seed=step)
        for optimizer in optimizers:
            optimizer.step()
        block.step()
        compare(models, stacked, block, optimizers, rtol=rtol)
    assert block.t.tolist() == [STEPS] * NUM_AGENTS


def test_matches_with_zero_weight_decay():
    models, optimizers, stacked, block = reference_and_block(weight_decay=0.0)
    for step in range(10):
        same_gradients(models, stacked, seed=step)
        for optimizer in optimizers:
            optimizer.step()
        block.step()
    compare(models, stacked, block, optimizers, rtol=1.0e-6)


# ------------------------------------------------------------------ 2. equivalence with freezing
def test_matches_when_one_agent_is_frozen_for_a_while():
    """Agent 1 is frozen for steps 10-30; the reference simply does not step it."""
    models, optimizers, stacked, block = reference_and_block()
    frozen_agent, frozen_range = 1, range(10, 30)
    for step in range(STEPS):
        same_gradients(models, stacked, seed=step)
        frozen = step in frozen_range
        for agent, optimizer in enumerate(optimizers):
            if frozen and agent == frozen_agent:
                continue
            optimizer.step()
        keep = torch.ones(NUM_AGENTS, dtype=torch.bool)
        keep[frozen_agent] = not frozen
        block.step(keep)
        compare(models, stacked, block, optimizers, rtol=1.0e-6)
    # the per-agent step count is what makes the bias correction right
    expected = [STEPS] * NUM_AGENTS
    expected[frozen_agent] = STEPS - len(frozen_range)
    assert block.t.tolist() == expected


# ------------------------------------------------------------------ 3. a frozen agent cannot move
def test_a_frozen_agent_is_bit_identical():
    _, _, stacked, block = reference_and_block()
    for param in stacked:
        param.grad = torch.randn_like(param)
    block.step()  # one step so the moments are non-zero

    frozen = 2
    before = (
        [p[frozen].clone() for p in stacked],
        [m[frozen].clone() for m in block.exp_avg],
        [m[frozen].clone() for m in block.exp_avg_sq],
        int(block.t[frozen]),
    )
    keep = torch.ones(NUM_AGENTS, dtype=torch.bool)
    keep[frozen] = False
    for param in stacked:
        param.grad = torch.randn_like(param) * 1.0e3  # a huge gradient it must ignore
    block.step(keep)

    for param, saved in zip(stacked, before[0]):
        assert torch.equal(param[frozen], saved)
    for moment, saved in zip(block.exp_avg, before[1]):
        assert torch.equal(moment[frozen], saved)
    for moment, saved in zip(block.exp_avg_sq, before[2]):
        assert torch.equal(moment[frozen], saved)
    assert int(block.t[frozen]) == before[3]
    # and the other agents did move
    assert not torch.equal(stacked[0][0], before[0][0])


def test_freezing_every_agent_changes_nothing():
    _, _, stacked, block = reference_and_block()
    for param in stacked:
        param.grad = torch.randn_like(param)
    before = [p.clone() for p in stacked]
    block.step(torch.zeros(NUM_AGENTS, dtype=torch.bool))
    for param, saved in zip(stacked, before):
        assert torch.equal(param, saved)
    assert block.t.tolist() == [0] * NUM_AGENTS


# ------------------------------------------------------------------ 4. state round trip
def test_agent_state_dict_round_trip():
    _, _, stacked, block = reference_and_block()
    for step in range(3):
        for param in stacked:
            param.grad = torch.randn_like(param)
        block.step()
    block.set_lr(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    target_params = [torch.nn.Parameter(torch.zeros_like(p)) for p in stacked]
    target = BlockAdamW(target_params, NUM_AGENTS, lr=LR, weight_decay=WEIGHT_DECAY)
    target.load_agent_state_dict(0, block.agent_state_dict(2))

    for index in range(len(stacked)):
        assert torch.equal(target.exp_avg[index][0], block.exp_avg[index][2])
        assert torch.equal(target.exp_avg_sq[index][0], block.exp_avg_sq[index][2])
        # the other slots are untouched
        assert float(target.exp_avg[index][1].abs().max()) == 0.0
    assert int(target.t[0]) == int(block.t[2])
    assert float(target.lr[0]) == float(block.lr[2])


def test_full_state_dict_round_trip():
    _, _, stacked, block = reference_and_block()
    for param in stacked:
        param.grad = torch.randn_like(param)
    block.step()
    state = copy.deepcopy(block.state_dict())

    fresh_params = [torch.nn.Parameter(torch.zeros_like(p)) for p in stacked]
    fresh = BlockAdamW(fresh_params, NUM_AGENTS, lr=LR, weight_decay=WEIGHT_DECAY)
    fresh.load_state_dict(state)
    assert torch.equal(fresh.t, block.t)
    assert torch.equal(fresh.lr, block.lr)
    for index in range(len(stacked)):
        assert torch.equal(fresh.exp_avg[index], block.exp_avg[index])
        assert torch.equal(fresh.exp_avg_sq[index], block.exp_avg_sq[index])

    with pytest.raises(KeyError):
        fresh.load_state_dict({"t": state["t"]})


# ------------------------------------------------------------------ 5. shape policing
def test_a_parameter_without_the_agent_dimension_raises():
    good = torch.nn.Parameter(torch.zeros(NUM_AGENTS, 3))
    bad = torch.nn.Parameter(torch.zeros(3, 3))
    with pytest.raises(ValueError) as err:
        BlockAdamW([good, bad], NUM_AGENTS, lr=LR)
    message = str(err.value)
    assert "(3, 3)" in message and str(NUM_AGENTS) in message

    with pytest.raises(ValueError):
        BlockAdamW([torch.nn.Parameter(torch.zeros(()))], NUM_AGENTS, lr=LR)
    with pytest.raises(ValueError):
        BlockAdamW([], NUM_AGENTS, lr=LR)


def test_a_missing_gradient_raises():
    params = [torch.nn.Parameter(torch.zeros(NUM_AGENTS, 3))]
    block = BlockAdamW(params, NUM_AGENTS, lr=LR)
    with pytest.raises(ValueError) as err:
        block.step()
    assert "no\ngradient" in str(err.value).replace(" ", "\n") or "gradient" in str(err.value)


def test_a_bad_keep_mask_raises():
    params = [torch.nn.Parameter(torch.zeros(NUM_AGENTS, 3))]
    params[0].grad = torch.zeros_like(params[0])
    block = BlockAdamW(params, NUM_AGENTS, lr=LR)
    with pytest.raises(ValueError):
        block.step(torch.ones(NUM_AGENTS))  # float, not bool
    with pytest.raises(ValueError):
        block.step(torch.ones(NUM_AGENTS + 1, dtype=torch.bool))


def test_set_lr_per_agent_and_scalar():
    params = [torch.nn.Parameter(torch.zeros(NUM_AGENTS, 3))]
    block = BlockAdamW(params, NUM_AGENTS, lr=LR)
    block.set_lr(0.5)
    assert block.lr.tolist() == [0.5] * NUM_AGENTS
    block.set_lr(torch.arange(NUM_AGENTS, dtype=torch.float32))
    assert block.lr.tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        block.set_lr(torch.zeros(NUM_AGENTS + 1))


def test_per_agent_lr_scales_only_that_agent():
    _, _, stacked, block = reference_and_block(weight_decay=0.0)
    block.set_lr(torch.tensor([0.0, LR, LR, LR]))  # agent 0 frozen by a zero LR
    before = stacked[0][0].clone()
    for param in stacked:
        param.grad = torch.randn_like(param)
    block.step()
    assert torch.equal(stacked[0][0], before)  # lr 0 moves nothing (weight_decay is 0)
    assert not torch.equal(stacked[0][1], before)


# ------------------------------------------------------------------ the LR schedule
def test_lr_at_constant_and_cosine():
    assert lr_at(0, 100, 1.0e-3, 1.0e-5, "constant") == 1.0e-3
    assert lr_at(50, 100, 1.0e-3, 1.0e-5, "constant") == 1.0e-3

    assert lr_at(0, 100, 1.0e-3, 1.0e-5, "cosine") == pytest.approx(1.0e-3)
    assert lr_at(100, 100, 1.0e-3, 1.0e-5, "cosine") == pytest.approx(1.0e-5)
    assert lr_at(50, 100, 1.0e-3, 1.0e-5, "cosine") == pytest.approx((1.0e-3 + 1.0e-5) / 2)
    # past the end it stays at the floor, and it never reads the data
    assert lr_at(500, 100, 1.0e-3, 1.0e-5, "cosine") == pytest.approx(1.0e-5)

    with pytest.raises(ValueError):
        lr_at(0, 100, 1.0e-3, 1.0e-5, "linear")
    with pytest.raises(ValueError):
        lr_at(0, 0, 1.0e-3, 1.0e-5, "cosine")


def test_cosine_matches_torchs_closed_form():
    reference = torch.optim.lr_scheduler.CosineAnnealingLR(
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0e-3), T_max=20, eta_min=1.0e-5
    )
    for update in range(21):
        assert lr_at(update, 20, 1.0e-3, 1.0e-5, "cosine") == pytest.approx(
            reference.get_last_lr()[0], rel=1e-9
        )
        reference.step()

"""Per-agent gradient clipping: each agent is clipped by its own norm, nothing shared."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.optim import clip_grad_norm_per_agent

NUM_AGENTS = 3


class BlockOnly(nn.Module):
    """Every parameter carries the leading agent dim."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_agents, 2, 2))
        self.bias = nn.Parameter(torch.zeros(num_agents, 2))


class WithShared(BlockOnly):
    """A parameter shared by every agent: clipping must refuse it."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        super().__init__(num_agents)
        self.shared = nn.Parameter(torch.zeros(5))


def set_grads(module: nn.Module, per_agent_scale) -> None:
    for param in module.parameters():
        grad = torch.ones_like(param)
        for agent, scale in enumerate(per_agent_scale):
            grad[agent] *= scale
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


def test_a_shared_parameter_raises():
    module = WithShared()
    set_grads(module, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError) as err:
        clip_grad_norm_per_agent(module, NUM_AGENTS, max_norm=1.0)
    message = str(err.value)
    assert "shared" in message and "shared parameter" in message

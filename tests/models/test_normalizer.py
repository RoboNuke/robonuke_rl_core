"""BlockRunningNorm must match N separate skrl RunningStandardScalers, one per agent."""

from __future__ import annotations

import pytest
import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

from robonuke_rl_core.models.normalizer import BlockRunningNorm

NUM_AGENTS = 3
SIZE = 4
ROWS = 5


def test_per_agent_stats_match_separate_skrl_scalers():
    block = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    singles = [RunningStandardScaler(size=SIZE, device="cpu") for _ in range(NUM_AGENTS)]

    torch.manual_seed(0)
    for _ in range(4):
        rows = [torch.randn(ROWS, SIZE) * (agent + 1) for agent in range(NUM_AGENTS)]
        block_out = block(torch.cat(rows, dim=0), train=True)
        for agent, single in enumerate(singles):
            single_out = single(rows[agent], train=True)
            assert torch.allclose(
                block_out[agent * ROWS : (agent + 1) * ROWS], single_out, atol=1e-6
            )

    for agent, single in enumerate(singles):
        assert torch.allclose(block.running_mean[agent], single.running_mean, atol=1e-10)
        assert torch.allclose(block.running_variance[agent], single.running_variance, atol=1e-10)
        assert float(block.current_count[agent]) == float(single.current_count)


def test_inverse_round_trip():
    block = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    torch.manual_seed(1)
    data = torch.cat([torch.randn(ROWS, SIZE) * (a + 1) + a for a in range(NUM_AGENTS)], dim=0)
    block(data, train=True)
    normalized = block(data)
    assert torch.allclose(block(normalized, inverse=True), data, atol=1e-4)


def test_stats_are_independent_across_agents():
    block = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    data = torch.zeros(NUM_AGENTS * ROWS, SIZE)
    data[ROWS : 2 * ROWS] = 1000.0  # agent 1 only
    block(data, train=True)
    assert float(block.running_mean[0].abs().max()) == 0.0
    assert float(block.running_mean[2].abs().max()) == 0.0
    assert float(block.running_mean[1].abs().max()) > 1.0


def test_per_agent_save_and_load():
    source = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    torch.manual_seed(2)
    source(torch.randn(NUM_AGENTS * ROWS, SIZE) * 3.0, train=True)

    target = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    target.load_state_dict_into(0, source.state_dict_for(2))
    assert torch.equal(target.running_mean[0], source.running_mean[2])
    assert torch.equal(target.running_variance[0], source.running_variance[2])
    assert torch.equal(target.current_count[0], source.current_count[2])
    # the other slots keep their fresh stats
    assert float(target.running_mean[1].abs().max()) == 0.0

    with pytest.raises(KeyError):
        target.load_state_dict_into(0, {"running_mean": source.running_mean[0]})


def test_bad_shapes_raise():
    block = BlockRunningNorm(NUM_AGENTS, SIZE, device="cpu")
    with pytest.raises(ValueError):
        block(torch.zeros(NUM_AGENTS * ROWS, SIZE + 1), train=True)
    with pytest.raises(ValueError):
        block(torch.zeros(NUM_AGENTS * ROWS + 1, SIZE), train=True)
    with pytest.raises(ValueError):
        BlockRunningNorm(NUM_AGENTS, SIZE, agent_axis=1, device="cpu")

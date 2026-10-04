"""MultiRandomMemory: every agent samples only from its own envs, in block order."""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.memory.multi_random import MultiRandomMemory

NUM_AGENTS = 3
ENVS_PER_AGENT = 4
NUM_ENVS = NUM_AGENTS * ENVS_PER_AGENT


def filled_memory(depth: int = 6) -> MultiRandomMemory:
    memory = MultiRandomMemory(
        memory_size=depth, num_envs=NUM_ENVS, num_agents=NUM_AGENTS, device="cpu"
    )
    memory.create_tensor(name="env_id", size=1, dtype=torch.float32)
    for _ in range(depth):
        # every row carries the env it came from, so sampled rows are traceable
        memory.add_samples(env_id=torch.arange(NUM_ENVS, dtype=torch.float32).unsqueeze(-1))
    return memory


def test_each_agents_rows_come_from_its_own_envs():
    memory = filled_memory()
    torch.manual_seed(0)
    (batch,) = memory.sample(names=["env_id"], batch_size=8)
    # batch_size is PER AGENT: the rows come back as [agent0 | agent1 | ...]
    assert batch[0].shape[0] == 8 * NUM_AGENTS
    env_ids = batch[0].reshape(NUM_AGENTS, 8)
    for agent in range(NUM_AGENTS):
        low, high = agent * ENVS_PER_AGENT, (agent + 1) * ENVS_PER_AGENT
        assert bool(((env_ids[agent] >= low) & (env_ids[agent] < high)).all())


def test_sample_all_partitions_every_row_once_per_agent():
    memory = filled_memory(depth=6)
    batches = memory.sample_all(names=["env_id"], mini_batches=3)
    assert len(batches) == 3
    rows_per_agent = 6 * ENVS_PER_AGENT // 3
    for (batch,) in batches:
        env_ids = batch.reshape(NUM_AGENTS, rows_per_agent)
        for agent in range(NUM_AGENTS):
            low, high = agent * ENVS_PER_AGENT, (agent + 1) * ENVS_PER_AGENT
            assert bool(((env_ids[agent] >= low) & (env_ids[agent] < high)).all())
    # together the minibatches cover the whole buffer exactly once
    seen = torch.cat([batch[0].reshape(-1) for batch in batches])
    assert seen.numel() == 6 * NUM_ENVS
    counts = torch.bincount(seen.long(), minlength=NUM_ENVS)
    assert bool((counts == 6).all())


def test_envs_must_divide_by_agents():
    with pytest.raises(ValueError) as err:
        MultiRandomMemory(memory_size=4, num_envs=7, num_agents=3, device="cpu")
    assert "divisible" in str(err.value)


def test_sampling_stays_inside_the_filled_part():
    memory = MultiRandomMemory(
        memory_size=10, num_envs=NUM_ENVS, num_agents=NUM_AGENTS, device="cpu"
    )
    memory.create_tensor(name="value", size=1, dtype=torch.float32)
    memory.add_samples(value=torch.ones(NUM_ENVS, 1))  # one step only: rows 1..9 are empty
    torch.manual_seed(0)
    (batch,) = memory.sample(names=["value"], batch_size=16)
    assert bool((batch[0] == 1.0).all())

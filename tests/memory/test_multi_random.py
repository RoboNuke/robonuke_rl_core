"""MultiRandomMemory: exact capacity, per-agent isolation, and the time view.

Every agent's slab holds exactly ``capacity`` transitions and nothing an agent samples ever
comes from another agent's rows.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.memory.multi_random import MultiRandomMemory

NUM_AGENTS = 3
ENVS_PER_AGENT = 4
NUM_ENVS = NUM_AGENTS * ENVS_PER_AGENT


def build(capacity: int, **overrides) -> MultiRandomMemory:
    kwargs = dict(
        capacity=capacity, num_envs=NUM_ENVS, num_agents=NUM_AGENTS, device="cpu"
    )
    kwargs.update(overrides)
    return MultiRandomMemory(**kwargs)


def filled_memory(steps: int = 6) -> MultiRandomMemory:
    """A buffer holding ``steps`` env-steps, every row tagged with the env that wrote it."""
    memory = build(steps * ENVS_PER_AGENT)
    memory.create_tensor("env_id", size=1, dtype=torch.float32)
    for _ in range(steps):
        memory.add_samples(env_id=torch.arange(NUM_ENVS, dtype=torch.float32).unsqueeze(-1))
    return memory


# ------------------------------------------------------------------ 1. exact capacity
@pytest.mark.parametrize("capacity", [7, 10, 13, 100, 4])
def test_capacity_is_exactly_what_was_asked_for(capacity):
    """Any positive capacity >= the envs per agent, divisible or not."""
    memory = build(capacity)
    memory.create_tensor("value", size=1, dtype=torch.float32)
    assert memory.get_tensor_by_name("value").shape == (NUM_AGENTS, capacity, 1)

    # write past the end: the slab never grows and never holds another agent's data
    for step in range(10):
        memory.add_samples(value=torch.full((NUM_ENVS, 1), float(step)))
    assert memory.get_tensor_by_name("value").shape == (NUM_AGENTS, capacity, 1)
    assert memory.size == min(10 * ENVS_PER_AGENT, capacity)
    assert memory.filled is (10 * ENVS_PER_AGENT >= capacity)


def test_a_real_sized_non_divisible_capacity_holds_exactly_what_was_asked():
    """The config's case: 1000 transitions per agent with 48 envs per agent."""
    memory = MultiRandomMemory(capacity=1000, num_envs=48 * 4, num_agents=4, device="cpu")
    memory.create_tensor("value", size=3, dtype=torch.float32)
    assert memory.get_tensor_by_name("value").shape == (4, 1000, 3)
    for step in range(25):  # 25 * 48 = 1200 rows written per agent
        memory.add_samples(value=torch.full((192, 3), float(step)))
    assert memory.size == 1000 and memory.filled
    assert memory.pointer == 1200 % 1000
    stored = memory.get_tensor_by_name("value")
    assert not torch.isnan(stored).any()
    # the newest 1000 rows per agent: steps 4 (partially) through 24
    assert set(stored.unique().tolist()) == set(float(s) for s in range(4, 25))


def test_a_capacity_below_one_env_step_raises():
    with pytest.raises(ValueError) as err:
        build(ENVS_PER_AGENT - 1)
    assert "envs per agent" in str(err.value)


def test_envs_must_divide_by_agents():
    with pytest.raises(ValueError) as err:
        MultiRandomMemory(capacity=8, num_envs=7, num_agents=3, device="cpu")
    assert "divisible" in str(err.value)


# ------------------------------------------------------------------ 2. the ring
def test_the_oldest_rows_are_the_ones_overwritten():
    """A non-divisible capacity wraps mid-step; the buffer still holds the newest rows."""
    capacity = 10  # 2.5 env-steps per agent
    memory = build(capacity)
    memory.create_tensor("stamp", size=1, dtype=torch.float32)
    total_steps = 7
    for step in range(total_steps):
        # a unique stamp per (step, env): step * 100 + env
        stamp = 100.0 * step + torch.arange(NUM_ENVS, dtype=torch.float32)
        memory.add_samples(stamp=stamp.unsqueeze(-1))

    stored = memory.get_tensor_by_name("stamp")
    assert not torch.isnan(stored).any()  # a full buffer has no unwritten rows
    for agent in range(NUM_AGENTS):
        expected = {
            100.0 * step + agent * ENVS_PER_AGENT + env
            for step in range(total_steps)
            for env in range(ENVS_PER_AGENT)
            # the newest `capacity` writes of this agent survive
            if step * ENVS_PER_AGENT + env >= total_steps * ENVS_PER_AGENT - capacity
        }
        assert set(stored[agent].flatten().tolist()) == expected


# ------------------------------------------------------------------ 3. routing
def test_each_agents_rows_come_from_its_own_envs():
    memory = filled_memory()
    torch.manual_seed(0)
    (batch,) = memory.sample(names=["env_id"], batch_size=8)
    # batch_size is PER AGENT: the rows come back as [agent0 | agent1 | ...]
    assert batch[0].shape == (8 * NUM_AGENTS, 1)
    env_ids = batch[0].reshape(NUM_AGENTS, 8)
    for agent in range(NUM_AGENTS):
        low, high = agent * ENVS_PER_AGENT, (agent + 1) * ENVS_PER_AGENT
        assert bool(((env_ids[agent] >= low) & (env_ids[agent] < high)).all())


def test_sampling_stays_inside_the_written_part():
    memory = build(40)
    memory.create_tensor("value", size=1, dtype=torch.float32)
    memory.add_samples(value=torch.ones(NUM_ENVS, 1))  # one step only: most rows are NaN
    torch.manual_seed(0)
    (batch,) = memory.sample(names=["value"], batch_size=16)
    assert bool((batch[0] == 1.0).all())
    with pytest.raises(ValueError):
        build(40).sample(names=["value"], batch_size=1)  # empty memory


def test_an_agents_rows_are_never_another_agents():
    """The sharper version: agent i's slab holds only agent i's envs."""
    memory = filled_memory(steps=5)
    stored = memory.get_tensor_by_name("env_id")
    for agent in range(NUM_AGENTS):
        low, high = agent * ENVS_PER_AGENT, (agent + 1) * ENVS_PER_AGENT
        assert bool(((stored[agent] >= low) & (stored[agent] < high)).all())


# ------------------------------------------------------------------ 4. full coverage
def test_sample_all_uses_every_row_exactly_once_per_agent():
    steps = 6
    memory = filled_memory(steps=steps)
    memory.create_tensor("row", size=1, dtype=torch.float32)
    rows = steps * ENVS_PER_AGENT
    # tag every row with its own index so coverage is checkable
    memory.get_tensor_by_name("row").copy_(
        torch.arange(rows, dtype=torch.float32).view(1, rows, 1).expand(NUM_AGENTS, rows, 1)
    )

    torch.manual_seed(0)
    batches = memory.sample_all(names=["env_id", "row"], mini_batches=3)
    assert len(batches) == 3
    per_agent = rows // 3

    seen = []
    for env_id, row in batches:
        assert env_id.shape == (NUM_AGENTS * per_agent, 1)
        ids = env_id.reshape(NUM_AGENTS, per_agent)
        for agent in range(NUM_AGENTS):
            low, high = agent * ENVS_PER_AGENT, (agent + 1) * ENVS_PER_AGENT
            assert bool(((ids[agent] >= low) & (ids[agent] < high)).all())
        seen.append(row.reshape(NUM_AGENTS, per_agent))

    covered = torch.cat(seen, dim=1).long()
    for agent in range(NUM_AGENTS):
        assert sorted(covered[agent].tolist()) == list(range(rows))


def test_sample_all_shuffles_each_agent_independently():
    memory = filled_memory(steps=6)
    memory.create_tensor("row", size=1, dtype=torch.float32)
    rows = 6 * ENVS_PER_AGENT
    memory.get_tensor_by_name("row").copy_(
        torch.arange(rows, dtype=torch.float32).view(1, rows, 1).expand(NUM_AGENTS, rows, 1)
    )
    torch.manual_seed(0)
    (batch,) = memory.sample_all(names=["row"], mini_batches=1)
    order = batch[0].reshape(NUM_AGENTS, rows)
    for agent in range(1, NUM_AGENTS):
        assert not torch.equal(order[0], order[agent])

    # shuffle=False gives the stored order, the same for every agent
    (unshuffled,) = memory.sample_all(names=["row"], mini_batches=1, shuffle=False)
    plain = unshuffled[0].reshape(NUM_AGENTS, rows)
    for agent in range(NUM_AGENTS):
        assert torch.equal(plain[agent], torch.arange(rows, dtype=torch.float32))


def test_mini_batches_must_divide_the_stored_rows():
    memory = filled_memory(steps=5)  # 20 rows per agent
    with pytest.raises(ValueError) as err:
        memory.sample_all(names=["env_id"], mini_batches=3)
    assert "exactly once" in str(err.value)


# ------------------------------------------------------------------ 5. the time view
def test_the_time_view_recovers_the_step_and_env_a_row_came_from():
    steps = 5
    memory = build(steps * ENVS_PER_AGENT)
    memory.create_tensor("stamp", size=1, dtype=torch.float32)
    for step in range(steps):
        stamp = 1000.0 * step + torch.arange(NUM_ENVS, dtype=torch.float32)
        memory.add_samples(stamp=stamp.unsqueeze(-1))

    assert memory.num_steps == steps
    view = memory.time_view("stamp")
    assert view.shape == (NUM_AGENTS, steps, ENVS_PER_AGENT, 1)
    for agent in range(NUM_AGENTS):
        for step in range(steps):
            for env in range(ENVS_PER_AGENT):
                expected = 1000.0 * step + agent * ENVS_PER_AGENT + env
                assert float(view[agent, step, env, 0]) == expected

    # it is a view: a write through it lands in the buffer
    view[0, 0, 0, 0] = -1.0
    assert float(memory.get_tensor_by_name("stamp")[0, 0, 0]) == -1.0


def test_the_time_view_needs_an_evenly_divisible_capacity():
    memory = build(10)  # 10 is not a multiple of 4 envs per agent
    memory.create_tensor("value", size=1, dtype=torch.float32)
    with pytest.raises(ValueError) as err:
        memory.time_view("value")
    assert "multiple of the envs per agent" in str(err.value)
    with pytest.raises(ValueError):
        _ = memory.num_steps


# ------------------------------------------------------------------ 6. tensors and misuse
def test_tensor_creation_sizes_from_spaces_and_ints():
    import gymnasium
    import numpy as np

    memory = build(8)
    space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
    assert memory.create_tensor("observations", size=space, dtype=torch.float32)
    assert memory.get_tensor_by_name("observations").shape == (NUM_AGENTS, 8, 5)
    assert memory.create_tensor("terminated", size=1, dtype=torch.bool)
    assert not memory.create_tensor("terminated", size=1, dtype=torch.bool)  # idempotent
    assert not memory.create_tensor("skipped", size=None)
    assert torch.isnan(memory.get_tensor_by_name("observations")).all()  # unwritten is NaN
    assert not memory.get_tensor_by_name("terminated").any()

    with pytest.raises(ValueError):
        memory.create_tensor("terminated", size=2, dtype=torch.bool)
    with pytest.raises(KeyError):
        memory.get_tensor_by_name("nope")
    assert memory.create_tensor("discrete", size=gymnasium.spaces.Discrete(7), dtype=torch.long)
    assert memory.get_tensor_by_name("discrete").shape == (NUM_AGENTS, 8, 1)


def test_set_tensor_by_name_round_trips_and_checks_the_shape():
    memory = build(8)
    memory.create_tensor("value", size=1, dtype=torch.float32)
    replacement = torch.arange(NUM_AGENTS * 8, dtype=torch.float32).view(NUM_AGENTS, 8, 1)
    memory.set_tensor_by_name("value", replacement)
    assert torch.equal(memory.get_tensor_by_name("value"), replacement)
    with pytest.raises(ValueError):
        memory.set_tensor_by_name("value", torch.zeros(NUM_AGENTS, 7, 1))


def test_add_samples_rejects_the_wrong_row_count_and_width():
    memory = build(8)
    memory.create_tensor("value", size=2, dtype=torch.float32)
    with pytest.raises(ValueError) as err:
        memory.add_samples(value=torch.zeros(NUM_ENVS + 1, 2))
    assert "one per env" in str(err.value)
    with pytest.raises(ValueError) as err:
        memory.add_samples(value=torch.zeros(NUM_ENVS, 3))
    assert "width" in str(err.value)
    with pytest.raises(KeyError):
        memory.add_samples(nope=torch.zeros(NUM_ENVS, 1))
    with pytest.raises(ValueError):
        memory.add_samples()


def test_the_pointer_is_shared_by_every_agent():
    memory = build(9)  # wraps mid-step, so the pointer must be tracked, not derived
    memory.create_tensor("value", size=1, dtype=torch.float32)
    for step in range(5):
        memory.add_samples(value=torch.zeros(NUM_ENVS, 1))
        assert memory.pointer == (step + 1) * ENVS_PER_AGENT % 9
        assert memory.size == min((step + 1) * ENVS_PER_AGENT, 9)

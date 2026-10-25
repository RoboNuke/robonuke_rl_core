"""``normalize_rewards`` / ``reward_clip``: ResiP's running-std reward scaling, per agent.

The scaler itself against ResiP's ``RunningMeanStdClip`` run separately per agent, then the
learners: what the memory stores, what the episode statistics see, the checkpoint, the
independence rule, and the default (off) leaving the stored reward untouched.
"""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner
from robonuke_rl_core.models.normalizer import BlockRewardScaler


class ReferenceRunningMeanStdClip:
    """ResiP's ``RunningMeanStdClip`` (robust-rearrangement ``src/gym/env_rl_wrapper.py``),
    verbatim in spirit: biased batch variance, the count starting at epsilon."""

    def __init__(self, epsilon=1e-4, clip_value=5.0):
        self.mean, self.var, self.count, self.clip_value = torch.zeros(()), torch.ones(()), epsilon, clip_value

    def update(self, x):
        batch_mean, batch_var, batch_count = x.mean(), x.var(unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def __call__(self, x):
        self.update(x)
        return torch.clamp(x / torch.sqrt(self.var + 1e-8), -self.clip_value, self.clip_value)


def test_the_scaler_is_resips_running_mean_std_clip_per_agent():
    torch.manual_seed(0)
    agents, envs = 3, 5
    scaler = BlockRewardScaler(agents, clip_threshold=5.0)
    references = [ReferenceRunningMeanStdClip() for _ in range(agents)]
    scales = [1.0, 100.0, 0.01]  # very different reward scales per agent
    for _ in range(30):
        rewards = torch.randn(agents, envs) * torch.tensor(scales).view(-1, 1) + 2.0
        out = scaler(rewards.reshape(-1, 1), train=True).view(agents, envs)
        for agent in range(agents):
            expected = references[agent](rewards[agent])
            assert torch.allclose(out[agent], expected, atol=1e-5, rtol=1e-4), agent
    # the three agents converged to three different scales: the stats are per agent
    assert scaler.running_variance[1] > 100 * scaler.running_variance[0] > 100 * scaler.running_variance[2]


def test_no_mean_subtraction_and_the_clip():
    scaler = BlockRewardScaler(1, clip_threshold=5.0)
    # a constant positive reward keeps its sign: no centering, and with zero variance the
    # scaled value is huge, so the clip is what comes out
    out = scaler(torch.full((8, 1), 3.0), train=True)
    assert torch.all(out == 5.0)
    scaler = BlockRewardScaler(1, clip_threshold=2.0)
    out = scaler(torch.tensor([[1.0], [-1.0], [10.0], [-10.0]]), train=True)
    assert torch.all(out <= 2.0) and torch.all(out >= -2.0)
    assert float(out[0]) > 0 > float(out[1])
    # eval-style call: no update
    before = scaler.running_variance.clone()
    scaler(torch.randn(4, 1) * 1000, train=False)
    assert torch.equal(before, scaler.running_variance)


def test_the_scaler_refuses_bad_shapes_and_clip():
    with pytest.raises(ValueError, match="clip_threshold"):
        BlockRewardScaler(2, clip_threshold=0.0)
    scaler = BlockRewardScaler(2)
    with pytest.raises(ValueError, match="divisible"):
        scaler(torch.zeros(3, 1), train=True)
    with pytest.raises(ValueError, match="one reward per row"):
        scaler(torch.zeros(4, 2), train=True)


def _step(learner, rewards, timestep: int = 0, truncated=None) -> None:
    n = learner.num_envs
    observations = torch.randn(n, OBS_DIM)
    actions, _ = learner.act(observations, None, timestep=timestep, timesteps=100)
    learner.record_transition(
        observations=observations, states=None, actions=actions, rewards=rewards,
        next_observations=torch.randn(n, OBS_DIM), next_states=None,
        terminated=torch.zeros(n, 1, dtype=torch.bool),
        truncated=torch.zeros(n, 1, dtype=torch.bool) if truncated is None else truncated,
        infos={}, timestep=timestep, timesteps=100,
    )


@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_the_memory_stores_the_scaled_reward_and_the_episode_stats_see_the_raw_one(learner_name):
    learner = build_learner(learner_name, num_agents=2, envs_per_agent=2, normalize_rewards=True, reward_clip=5.0)
    reference = BlockRewardScaler(2, clip_threshold=5.0)
    torch.manual_seed(1)
    raw = [torch.randn(4, 1) * 10.0 for _ in range(3)]
    for step, rewards in enumerate(raw):
        _step(learner, rewards, timestep=step)
    stored = learner.memory.get_tensor_by_name("rewards")[:, :3 * 2].reshape(2, 3, 2)  # (agents, steps, envs)
    for step, rewards in enumerate(raw):
        expected = reference(rewards, train=True).view(2, 2)
        assert torch.allclose(stored[:, step], expected, atol=1e-6), step
    # the episode return is the raw reward's running sum
    assert torch.allclose(learner._episode_return, sum(raw).view(-1))
    assert "rewards" in learner._checkpoint_normalizers()


@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_off_by_default_the_stored_reward_is_the_raw_one(learner_name):
    learner = build_learner(learner_name, num_agents=2, envs_per_agent=2)
    assert learner.reward_normalizer is None
    assert "rewards" not in learner._checkpoint_normalizers()
    rewards = torch.randn(4, 1) * 10.0
    _step(learner, rewards)
    assert torch.equal(learner.memory.get_tensor_by_name("rewards")[:, :2].reshape(-1, 1), rewards)


def test_the_shaper_runs_first_then_the_scaler():
    learner = build_learner("sac", num_agents=1, envs_per_agent=4, normalize_rewards=True, rewards_shaper="test_paths:scale_rewards")
    reference = BlockRewardScaler(1)
    rewards = torch.randn(4, 1)
    _step(learner, rewards)
    expected = reference(rewards * 10.0, train=True)
    assert torch.allclose(learner.memory.get_tensor_by_name("rewards")[0, :4], expected)


def test_ppo_time_limit_bootstrap_adds_to_the_scaled_reward():
    learner = build_learner("ppo", num_agents=1, envs_per_agent=4, normalize_rewards=True, time_limit_bootstrap=True)
    reference = BlockRewardScaler(1)
    rewards = torch.randn(4, 1)
    truncated = torch.tensor([[True], [False], [False], [True]])
    _step(learner, rewards, truncated=truncated)
    stored = learner.memory.get_tensor_by_name("rewards")[0, :4]
    scaled = reference(rewards, train=True)
    assert torch.allclose(stored[~truncated.view(-1)], scaled[~truncated.view(-1)])
    assert not torch.allclose(stored[truncated.view(-1)], scaled[truncated.view(-1)])  # + gamma V(next)


def test_the_scaler_state_rides_the_checkpoint(tmp_path):
    learner = build_learner("sac", num_agents=2, envs_per_agent=2, normalize_rewards=True, run_dirs_list=[tmp_path / "a0", tmp_path / "a1"])
    for step in range(5):
        _step(learner, torch.randn(4, 1) * 7.0, timestep=step)
    paths = learner.save_checkpoints(step=5)
    fresh = build_learner("sac", num_agents=1, envs_per_agent=2, normalize_rewards=True)
    fresh.load_agent(paths[1], slot=0, with_optimizer=False)
    assert torch.equal(fresh.reward_normalizer.running_variance[0], learner.reward_normalizer.running_variance[1])
    assert torch.equal(fresh.reward_normalizer.current_count[0], learner.reward_normalizer.current_count[1])
    # a checkpoint without the scaler does not load into a learner that has one
    plain = build_learner("sac", num_agents=1, envs_per_agent=2)
    with pytest.raises(KeyError, match="normalizers"):
        plain.load_agent(paths[1], slot=0, with_optimizer=False)


def test_agents_scale_independently():
    torch.manual_seed(0)
    a = build_learner("sac", num_agents=3, envs_per_agent=2, normalize_rewards=True)
    torch.manual_seed(0)
    b = build_learner("sac", num_agents=3, envs_per_agent=2, normalize_rewards=True)
    for step in range(4):
        rewards = torch.randn(6, 1)
        extreme = rewards.clone()
        extreme[2:4] *= 1.0e6  # agent 1's envs only
        torch.manual_seed(step)
        _step(a, rewards, timestep=step)
        torch.manual_seed(step)
        _step(b, extreme, timestep=step)
    stored_a = a.memory.get_tensor_by_name("rewards")[:, :8]
    stored_b = b.memory.get_tensor_by_name("rewards")[:, :8]
    for agent in (0, 2):
        assert torch.equal(stored_a[agent], stored_b[agent])
    assert not torch.equal(stored_a[1], stored_b[1])

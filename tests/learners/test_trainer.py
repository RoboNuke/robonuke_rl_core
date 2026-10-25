"""The package trainer: skrl's sequential loop plus the env reset PPO's
``reset_every_iteration`` asks for, driven on CPU against a fake vectorized env."""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, box, build_learner
from robonuke_rl_core.learners.trainer import SequentialTrainerCfg, Trainer


class FakeVecEnv:
    """Enough of skrl's ``IsaacLabWrapper`` for the loop: a one-shot reset guard, a per-env
    clock, auto-resets at the episode length, and observations that spell out which reset
    they came from (column 0) and the clock (column 1), so a test can tell a fresh one."""

    def __init__(self, num_envs: int, episode_length: int = 1000) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.observation_space, self.action_space, self.state_space = box(OBS_DIM), box(ACT_DIM), None
        self.episode_length = episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self._reset_once = True
        self.resets = 0
        self.total_steps = 0
        self.steps_at_reset: list = []

    @property
    def unwrapped(self):
        return self

    def _observations(self) -> torch.Tensor:
        obs = torch.zeros(self.num_envs, OBS_DIM)
        obs[:, 0] = float(self.resets)
        obs[:, 1] = self.episode_length_buf.float()
        return obs

    def reset(self):
        if self._reset_once:
            self.resets += 1
            self.steps_at_reset.append(self.total_steps)
            self.episode_length_buf.zero_()
            self._reset_once = False
        return self._observations(), {}

    def step(self, actions):
        assert actions.shape == (self.num_envs, ACT_DIM)
        self.total_steps += 1
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.episode_length
        self.episode_length_buf[done] = 0  # Isaac Lab auto-resets a finished env in step
        return self._observations(), torch.randn(self.num_envs, 1), done.view(-1, 1), done.view(-1, 1), {}

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        return None


def run(learner, env, timesteps: int):
    Trainer(
        env=env,
        agents=learner,
        cfg=SequentialTrainerCfg(timesteps=timesteps, headless=True, disable_progressbar=True, close_environment_at_exit=False),
    ).train()


def test_reset_every_iteration_resets_every_env_after_each_update_and_the_next_rollout_is_fresh():
    learner = build_learner("ppo", num_agents=2, envs_per_agent=2, rollouts=4, rollout=4, reset_every_iteration=True)
    counts = episodes_emitted(learner)
    env = FakeVecEnv(num_envs=4)
    run(learner, env, timesteps=12)
    assert counts == []  # no episode ended on its own, and a cut one is no episode
    # the trainer's own first reset, then one right after each of the three updates
    assert env.resets == 4
    assert env.steps_at_reset == [0, 4, 8, 12]
    assert int(env.episode_length_buf.max()) == 0
    assert learner.env_reset_requested is False
    # the last rollout (steps 8..11) started from the observations the reset at step 8 returned:
    # reset number 3, clocks 0, 1, 2, 3
    stored = learner.memory.time_view("observations")  # (agents, steps, envs_per_agent, dim)
    assert torch.all(stored[..., 0] == 3.0)
    assert torch.equal(stored[0, :, 0, 1], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # the episodes the resets cut short count for nothing, and the accumulators restarted
    assert float(learner._finished_count.sum()) == 0.0
    assert torch.all(learner._episode_length == 0.0)
    assert torch.all(learner._episode_return == 0.0)


def test_without_the_flag_the_loop_is_skrls():
    learner = build_learner("ppo", num_agents=2, envs_per_agent=2, rollouts=4, rollout=4)
    env = FakeVecEnv(num_envs=4)
    run(learner, env, timesteps=12)
    assert env.resets == 1 and env.steps_at_reset == [0]
    assert learner.env_reset_requested is False
    stored = learner.memory.time_view("observations")
    assert torch.equal(stored[0, :, 0, 1], torch.tensor([8.0, 9.0, 10.0, 11.0]))  # the clock ran on
    # natural episode ends are still counted
    assert float(learner._episode_length[0]) == 12.0


def episodes_emitted(learner) -> list:
    """Counts the ``episode/return`` values that reach the log hook: one per finished episode."""
    counts: list = []
    learner.on_log.append(
        lambda agent, metrics, step: counts.append(metrics["episode/return"].numel()) if "episode/return" in metrics else None
    )
    return counts


def test_a_natural_done_still_counts_with_the_flag_on_and_a_cut_episode_does_not():
    learner = build_learner("ppo", num_agents=1, envs_per_agent=4, rollouts=4, rollout=4, reset_every_iteration=True)
    counts = episodes_emitted(learner)
    env = FakeVecEnv(num_envs=4, episode_length=3)  # the clock ends an episode at steps 3 and 6
    run(learner, env, timesteps=8)
    assert env.resets == 3  # the first, then after the updates at steps 4 and 8
    # 4 envs x 2 natural ends = 8 episodes; the ones the resets cut (at 4 and 8) are not counted
    assert sum(counts) == 8
    assert torch.all(learner._episode_length == 0.0)


def test_the_update_itself_requests_the_reset_and_a_trainer_that_ignores_it_is_refused():
    learner = build_learner("ppo", num_agents=1, envs_per_agent=2, rollouts=4, rollout=4, reset_every_iteration=True)
    n = learner.num_envs

    def step(timestep: int) -> None:
        observations = torch.randn(n, OBS_DIM)
        actions, _ = learner.act(observations, None, timestep=timestep, timesteps=100)
        learner.record_transition(
            observations=observations, states=None, actions=actions, rewards=torch.zeros(n, 1),
            next_observations=torch.randn(n, OBS_DIM), next_states=None,
            terminated=torch.zeros(n, 1, dtype=torch.bool), truncated=torch.zeros(n, 1, dtype=torch.bool),
            infos={}, timestep=timestep, timesteps=100,
        )
        learner.post_interaction(timestep=timestep, timesteps=100)

    for timestep in range(3):
        step(timestep)
        assert learner.env_reset_requested is False
    step(3)  # the rollout is full: the update ran and asked for the reset
    assert learner.env_reset_requested is True
    # skrl's own trainer would just step on; the learner refuses the next transition, naming the fix
    with pytest.raises(RuntimeError, match="Trainer"):
        step(4)
    learner.forced_env_reset()  # what the package trainer does after the reset
    assert learner.env_reset_requested is False
    step(4)


def test_the_trainer_takes_one_learner_only():
    env = FakeVecEnv(num_envs=4)
    a = build_learner("ppo", num_agents=1, envs_per_agent=2)
    b = build_learner("ppo", num_agents=1, envs_per_agent=2)
    trainer = Trainer(env=env, agents=[a, b], scopes=[2, 2], cfg=SequentialTrainerCfg(timesteps=2, headless=True, close_environment_at_exit=False))
    with pytest.raises(ValueError, match="one learner"):
        trainer.train()

"""Every metric channel a wrapper publishes must appear on EVERY step, or on none.

An intermittent channel is invisible in training — the accumulator just averages whatever
arrived — and then fails a whole eval round at the end, in `EvalStateWriter.end_round`, with
nothing to show for the GPU time. The parquet is rectangular, so it has to be.

The rule is easy to break by accident: the natural way to write a "did anything break this
step" metric is behind `if broke.any()`, which is both an intermittent channel and a host
sync. So the fragile wrapper is driven here against a fake env, on CPU, over steps where
something breaks and steps where nothing does, and the key sets are compared.
"""

from __future__ import annotations

import gymnasium
import pytest
import torch

from robonuke_rl_core.envs.cfg import FragileCfg

TASK = "Isaac-Forge-PegInsert-Direct-v0"
NUM_ENVS = 4


class FakeUnwrapped:
    """The handful of attributes the fragile wrapper reads off a Forge env."""

    def __init__(self, num_envs: int = NUM_ENVS):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.force_sensor_smooth = torch.zeros(num_envs, 6)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.in_contact = torch.ones(num_envs, 3, dtype=torch.bool)

    def _get_dones(self):
        flags = torch.zeros(self.num_envs, dtype=torch.bool)
        return flags, flags.clone()


class FakeEnv(gymnasium.Env):
    """A gym env whose `step` reproduces `DirectRLEnv.step`'s ORDER, which is the point.

    Isaac Lab increments `episode_length_buf`, calls `_get_dones` (which the wrapper has
    patched), and then calls `_reset_idx` for the envs that are done -- which zeroes
    `episode_length_buf` for exactly those envs. A wrapper that reads the step count after
    `env.step` returns therefore reads 0 for every env that just broke.
    """

    def __init__(self, unwrapped: FakeUnwrapped):
        self._unwrapped = unwrapped

    @property
    def unwrapped(self):  # gymnasium.Env defines this; the wrapper reads the task object
        return self._unwrapped

    def step(self, action):
        env = self._unwrapped
        env.episode_length_buf += 1
        terminated, time_out = env._get_dones()  # the wrapper's, once it has patched it
        done = terminated | time_out
        env.episode_length_buf[done] = 0  # _reset_idx, inside step, before we return
        zeros = torch.zeros(env.num_envs, 1)
        return {}, zeros.clone(), terminated.reshape(-1, 1), time_out.reshape(-1, 1), {}


def wrapper(**overrides):
    from robonuke_rl_core.envs.forge.fragile import ForgeFragileObjectWrapper

    fields = dict(enabled=True, break_force=[10.0])
    fields.update(overrides)
    env = FakeEnv(FakeUnwrapped())
    return ForgeFragileObjectWrapper(env, FragileCfg(**fields), TASK), env.unwrapped


def run_step(wrapped, force: float) -> dict:
    """Step once with a given wrist force; return the step's infos."""
    wrapped.env.unwrapped.force_sensor_smooth[:, 0] = force
    _, _, _, _, infos = wrapped.step(torch.zeros(NUM_ENVS, 1))
    return infos


def channels(wrapped, force: float) -> tuple:
    """Step once; return (step channel names, episode channel names)."""
    infos = run_step(wrapped, force)
    return (
        tuple(sorted(infos["metrics_to_log"])),
        tuple(sorted(infos["episode_metrics_to_log"])),
    )


# ------------------------------------------------------------------ the invariant
def test_the_fragile_channels_do_not_come_and_go():
    """A broken step and a quiet step must publish exactly the same names."""
    wrapped, _ = wrapper(break_force=[10.0])
    quiet = channels(wrapped, force=1.0)  # well under the threshold: nothing breaks
    broken = channels(wrapped, force=1.0e3)  # every env breaks

    assert quiet == broken, "the channel set changed between a quiet step and a break"
    assert "fragile/break_step" in quiet[1], "break_step must be published on quiet steps too"


def test_a_quiet_step_reports_nan_rather_than_zero():
    """NaN is 'not applicable': a zero would drag the mean break step toward step 0."""
    wrapped, _ = wrapper(break_force=[10.0])
    infos = run_step(wrapped, force=1.0)
    assert torch.isnan(infos["episode_metrics_to_log"]["fragile/break_step"]).all()
    assert float(infos["episode_metrics_to_log"]["fragile/broke"].sum()) == 0.0


def test_the_break_step_survives_the_env_resetting_the_counter():
    """The regression: `_reset_idx` zeroes `episode_length_buf` for the envs that broke,
    inside `env.step`, so reading it afterwards reports 0 for every one of them."""
    wrapped, unwrapped = wrapper(break_force=[10.0])
    for _ in range(4):
        run_step(wrapped, force=1.0)  # four quiet steps: the counter reaches 4
    assert int(unwrapped.episode_length_buf.min()) == 4

    infos = run_step(wrapped, force=1.0e3)  # the fifth step breaks every env
    values = infos["episode_metrics_to_log"]["fragile/break_step"]
    assert torch.equal(values, torch.full((NUM_ENVS,), 5.0)), "read after the reset, not before"
    assert int(unwrapped.episode_length_buf.max()) == 0  # the env really did zero it


@pytest.mark.parametrize("directional", [False, True])
def test_the_cause_channels_are_constant_for_a_given_mode(directional):
    """Which causes exist follows the config, so the names never vary step to step."""
    wrapped, _ = wrapper(
        direction_break_force=directional,
        break_force=[10.0, 20.0] if directional else [10.0],
    )
    if directional:
        # the live peg axis comes from Isaac Lab's quat_apply; the axis itself is not what
        # this test is about, so pin it rather than skip the case off the GPU machine
        wrapped.peg_axis = lambda: torch.tensor([[0.0, 0.0, 1.0]]).expand(NUM_ENVS, 3)

    quiet, broken = channels(wrapped, force=1.0), channels(wrapped, force=1.0e3)
    assert quiet == broken
    causes = {name for name in quiet[1] if name.startswith("fragile/rate_cause_")}
    expected = (
        {"fragile/rate_cause_normal", "fragile/rate_cause_shear"}
        if directional
        else {"fragile/rate_cause_force"}
    )
    assert causes == expected


# ------------------------------------------------------------------ the task-metric wrapper
class FakeTaskEnv(FakeEnv):
    """A Forge-ish env with the attributes `ForgeTaskMetricsWrapper` reads.

    `_get_curr_successes` answers from `self.engaged_flag`, which `_reset_idx` clears — the
    way the real scene state is restored before the wrapper's `step` ever runs.
    """

    def __init__(self, unwrapped: FakeUnwrapped):
        super().__init__(unwrapped)
        u = unwrapped
        u.ep_succeeded = torch.zeros(u.num_envs, dtype=torch.long)
        u.ep_success_times = torch.zeros(u.num_envs, dtype=torch.long)
        u.engaged_flag = torch.zeros(u.num_envs, dtype=torch.bool)
        u.cfg_task = type("TaskCfg", (), {"engage_threshold": 0.01})()
        u._get_curr_successes = lambda success_threshold, check_rot=True: u.engaged_flag.clone()
        u._log_factory_metrics = lambda rew_dict, curr_successes: None
        self.terminated = torch.zeros(u.num_envs, dtype=torch.bool)
        self.truncated = torch.zeros(u.num_envs, dtype=torch.bool)

    def step(self, action):
        u = self._unwrapped
        u.episode_length_buf += 1
        # the env's own order: rewards (where our tap runs) ... then _reset_idx
        u._log_factory_metrics({"reward/term": torch.zeros(u.num_envs)}, u.ep_succeeded.bool())
        done = self.terminated | self.truncated
        u.engaged_flag[done] = False  # _reset_idx restores the initial condition
        zeros = torch.zeros(u.num_envs, 1)
        return (
            {},
            zeros.clone(),
            self.terminated.reshape(-1, 1),
            self.truncated.reshape(-1, 1),
            {},
        )


def task_wrapper():
    from robonuke_rl_core.envs.cfg import TaskMetricsCfg
    from robonuke_rl_core.envs.forge.metrics import ForgeTaskMetricsWrapper

    env = FakeTaskEnv(FakeUnwrapped())
    return ForgeTaskMetricsWrapper(env, TaskMetricsCfg(), TASK), env


def test_engagement_is_latched_so_it_cannot_undercut_success():
    """The regression: engagement was read live, after `_reset_idx` had already run, so it
    came back 0 for every episode that ended -- a 0% engagement rate beside an 80% success
    rate, which is impossible by definition."""
    wrapped, env = task_wrapper()
    u = env.unwrapped

    u.engaged_flag[:] = True  # engaged mid-episode
    wrapped.step(torch.zeros(NUM_ENVS, 1))

    u.ep_succeeded[:] = 1  # and it went on to succeed
    env.truncated[:] = True  # Forge ends on the clock, raising both flags
    env.terminated[:] = True
    _, _, _, _, infos = wrapped.step(torch.zeros(NUM_ENVS, 1))

    episode = infos["episode_metrics_to_log"]
    assert float(episode["episode/engaged"].min()) == 1.0, "engagement was read after the reset"
    assert float(episode["episode/success"].min()) == 1.0
    # the invariant the user is entitled to: engagement >= success, always
    assert float(episode["episode/engaged"].mean()) >= float(episode["episode/success"].mean())


def test_a_break_is_not_counted_as_a_timeout():
    """An early termination is not the clock running out; it belongs to `fragile/broke`."""
    wrapped, env = task_wrapper()
    env.terminated[:] = True  # the fragile wrapper ends the episode through `terminated`
    env.truncated[:] = False  # the clock has NOT run out
    _, _, _, _, infos = wrapped.step(torch.zeros(NUM_ENVS, 1))

    episode = infos["episode_metrics_to_log"]
    assert float(episode["termination/timeout"].max()) == 0.0
    assert float(episode["termination/success"].max()) == 0.0


def test_a_real_timeout_is_counted_as_one():
    wrapped, env = task_wrapper()
    env.terminated[:] = True  # Forge raises BOTH flags at the time limit
    env.truncated[:] = True
    _, _, _, _, infos = wrapped.step(torch.zeros(NUM_ENVS, 1))
    assert float(infos["episode_metrics_to_log"]["termination/timeout"].min()) == 1.0


def test_a_success_wins_over_the_timeout_it_coincides_with():
    wrapped, env = task_wrapper()
    env.unwrapped.ep_succeeded[:] = 1
    env.terminated[:] = True
    env.truncated[:] = True
    _, _, _, _, infos = wrapped.step(torch.zeros(NUM_ENVS, 1))
    episode = infos["episode_metrics_to_log"]
    assert float(episode["termination/success"].min()) == 1.0
    assert float(episode["termination/timeout"].max()) == 0.0

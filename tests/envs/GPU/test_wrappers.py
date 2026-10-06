"""The Forge wrappers on a real env: reset isolation, contact flags, and the fragile break.

Run on the GPU machine: `pytest -m gpu`. The GPU suite shares one env, so each test attaches
its wrapper, exercises it, and restores whatever it patched.

The one that matters most is the efficient reset: its entire purpose is that resetting one env
leaves every other env exactly as it was, so that is checked bit for bit.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.envs.cfg import ContactCfg, EfficientResetCfg, FragileCfg

pytestmark = pytest.mark.gpu


@pytest.fixture
def ready_env(gpu_env):
    """A reset, stepped env — the state tensors only exist after a step."""
    from robonuke_rl_core.evaluation import force_env_reset

    force_env_reset(gpu_env)
    zero = torch.zeros(gpu_env.num_envs, *gpu_env.action_space.shape, device=gpu_env.device)
    gpu_env.step(zero)
    return gpu_env


def physics_snapshot(unwrapped):
    """Everything about an env's physics state that a reset could disturb."""
    state = unwrapped.scene.get_state(is_relative=True)

    def flatten(node, prefix=""):
        out = {}
        for key, value in node.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                out.update(flatten(value, f"{name}."))
            elif torch.is_tensor(value):
                out[name] = value.clone()
        return out

    return flatten(state)


# ------------------------------------------------------------------ efficient reset
def test_a_partial_reset_touches_only_its_own_envs(gpu_cfg, ready_env):
    """The whole point of the wrapper: one env resets, the others do not move."""
    from robonuke_rl_core.envs.forge.reset import ForgeEfficientResetWrapper

    unwrapped = ready_env.unwrapped
    original_reset = unwrapped._reset_idx
    wrapper = ForgeEfficientResetWrapper(unwrapped, EfficientResetCfg(enabled=True), gpu_cfg.task_name)
    try:
        # a full reset first, so the wrapper has donors cached
        every = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        unwrapped._reset_idx(every)
        ready_env.step(torch.zeros(ready_env.num_envs, *ready_env.action_space.shape,
                                   device=ready_env.device))

        before = physics_snapshot(unwrapped)
        bookkeeping = {
            name: getattr(unwrapped, name).clone()
            for name in ("task_prop_gains", "ema_factor", "contact_penalty_thresholds")
            if hasattr(unwrapped, name)
        }

        victim = torch.tensor([1], device=unwrapped.device)
        unwrapped._reset_idx(victim)
        after = physics_snapshot(unwrapped)

        others = [i for i in range(unwrapped.num_envs) if i != 1]
        for name, tensor in before.items():
            if tensor.shape[:1] != (unwrapped.num_envs,):
                continue
            assert torch.equal(tensor[others], after[name][others]), (
                f"{name} changed for an env that was not reset"
            )
        for name, tensor in bookkeeping.items():
            assert torch.equal(tensor[others], getattr(unwrapped, name)[others]), name

        # and the victim really was reset
        assert int(unwrapped.episode_length_buf[1]) == 0
        assert any(
            not torch.equal(tensor[1], after[name][1])
            for name, tensor in before.items()
            if tensor.shape[:1] == (unwrapped.num_envs,)
        )
    finally:
        unwrapped._reset_idx = original_reset


def test_a_full_reset_still_runs_the_env_s_own_chain(gpu_cfg, ready_env):
    from robonuke_rl_core.envs.forge.reset import ForgeEfficientResetWrapper

    unwrapped = ready_env.unwrapped
    original_reset = unwrapped._reset_idx
    calls = []

    def counting_reset(env_ids):
        calls.append(int(env_ids.numel()))
        return original_reset(env_ids)

    unwrapped._reset_idx = counting_reset
    try:
        wrapper = ForgeEfficientResetWrapper(
            unwrapped, EfficientResetCfg(enabled=True), gpu_cfg.task_name
        )
        every = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        unwrapped._reset_idx(every)
        assert calls == [unwrapped.num_envs]  # the full chain ran, once

        unwrapped._reset_idx(torch.tensor([0], device=unwrapped.device))
        assert calls == [unwrapped.num_envs]  # the partial path did NOT call it again
    finally:
        unwrapped._reset_idx = original_reset


# ------------------------------------------------------------------ contact
def test_the_contact_wrapper_publishes_per_axis_flags(gpu_cfg, ready_env):
    """The sensor is installed on the session env by the fixture; the flags come from it."""
    from robonuke_rl_core.envs.forge.contact import SENSOR_KEY, ForgeContactSensorWrapper

    unwrapped = ready_env.unwrapped
    original = unwrapped._get_observations
    wrapper = ForgeContactSensorWrapper(unwrapped, ContactCfg(enabled=True), gpu_cfg.task_name)
    try:
        flags = wrapper.refresh()
        assert flags.shape == (unwrapped.num_envs, 3) and flags.dtype == torch.bool
        assert torch.equal(flags, unwrapped.in_contact)  # published for other wrappers
    finally:
        unwrapped._get_observations = original


# ------------------------------------------------------------------ fragile
def test_a_force_over_the_threshold_terminates_the_env(gpu_cfg, ready_env):
    from robonuke_rl_core.envs.forge.fragile import ForgeFragileObjectWrapper

    unwrapped = ready_env.unwrapped
    original_dones = unwrapped._get_dones
    cfg = FragileCfg(enabled=True, break_force=[1.0e6])  # unbreakable to start
    wrapper = ForgeFragileObjectWrapper(unwrapped, cfg, gpu_cfg.task_name)
    try:
        terminated, _ = unwrapped._get_dones()
        assert not bool(terminated.any())  # nothing breaks at 1e6 N

        # a threshold below the live reading must break every env
        live = float(torch.linalg.norm(wrapper.measured_force(), dim=1).max())
        wrapper.break_force = max(live * 0.5, 1e-6)
        terminated, _ = unwrapped._get_dones()
        assert bool(terminated.any())
    finally:
        unwrapped._get_dones = original_dones


def test_the_directional_mode_uses_the_live_peg_axis(gpu_cfg, ready_env):
    from robonuke_rl_core.envs.forge.fragile import ForgeFragileObjectWrapper, split_axial_shear

    unwrapped = ready_env.unwrapped
    original_dones = unwrapped._get_dones
    cfg = FragileCfg(enabled=True, direction_break_force=True, break_force=[1.0e6, 1.0e6])
    wrapper = ForgeFragileObjectWrapper(unwrapped, cfg, gpu_cfg.task_name)
    try:
        axis = wrapper.peg_axis()
        assert axis.shape == (unwrapped.num_envs, 3)
        assert torch.allclose(
            torch.linalg.norm(axis, dim=1), torch.ones(unwrapped.num_envs, device=axis.device),
            atol=1e-5,
        )
        axial, shear = split_axial_shear(wrapper.measured_force(), axis)
        assert torch.isfinite(axial).all() and torch.isfinite(shear).all()
        assert not bool(wrapper.force_violations().any())  # 1e6 N thresholds
    finally:
        unwrapped._get_dones = original_dones


# ------------------------------------------------------------------ task metrics
def test_the_task_metrics_wrapper_publishes_per_agent_outcomes(gpu_cfg, ready_env):
    """Success, cause, reward terms and prediction quality — per env, not pre-averaged."""
    from robonuke_rl_core.envs.cfg import TaskMetricsCfg
    from robonuke_rl_core.envs.forge.metrics import SUCCESS_PREDICTION, ForgeTaskMetricsWrapper

    unwrapped = ready_env.unwrapped
    originals = (unwrapped._log_factory_metrics, getattr(unwrapped, "_log_forge_metrics", None))
    wrapper = ForgeTaskMetricsWrapper(unwrapped, TaskMetricsCfg(), gpu_cfg.task_name)
    try:
        zero = torch.zeros(ready_env.num_envs, *ready_env.action_space.shape, device=ready_env.device)
        ready_env.step(zero)  # a step fills the taps

        dtype = torch.float32
        step_metrics = wrapper.step_metrics(dtype)
        assert step_metrics, "no reward terms were captured from the env"
        for name, value in step_metrics.items():
            assert name.startswith("reward/")
            assert value.shape == (unwrapped.num_envs,)  # per env, not a scalar
            assert torch.isfinite(value).all()

        episode = wrapper.episode_metrics(
            torch.zeros(unwrapped.num_envs, 1, dtype=torch.bool, device=unwrapped.device),
            torch.ones(unwrapped.num_envs, 1, dtype=torch.bool, device=unwrapped.device),
            dtype,
        )
        assert "episode/success" in episode and "termination/timeout" in episode
        assert episode["episode/success"].shape == (unwrapped.num_envs,)
        # a success that never happened reports NaN, not a zero that would drag the mean down
        assert torch.isnan(episode["episode/success_step"]).any()
        # Forge's success-prediction action is scored under its own prefix
        prediction = [name for name in episode if name.startswith(SUCCESS_PREDICTION)]
        assert prediction, "the Forge success-prediction metrics are missing"
        assert f"{SUCCESS_PREDICTION}/error" in episode
    finally:
        unwrapped._log_factory_metrics = originals[0]
        if originals[1] is not None:
            unwrapped._log_forge_metrics = originals[1]


def test_the_fragile_wrapper_reports_which_break_happened(gpu_cfg, ready_env):
    from robonuke_rl_core.envs.forge.fragile import ForgeFragileObjectWrapper

    unwrapped = ready_env.unwrapped
    original_dones = unwrapped._get_dones
    cfg = FragileCfg(enabled=True, direction_break_force=True, break_force=[1.0e-6, 1.0e-6])
    wrapper = ForgeFragileObjectWrapper(unwrapped, cfg, gpu_cfg.task_name)
    try:
        unwrapped._get_dones()  # thresholds at ~0: everything breaks, on both causes
        assert set(wrapper._cause) == {"normal", "shear"}
        assert bool(wrapper._cause["normal"].any()) or bool(wrapper._cause["shear"].any())
    finally:
        unwrapped._get_dones = original_dones

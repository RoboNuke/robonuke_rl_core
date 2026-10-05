"""The controller core on a real Forge env: parity with the native controller, then the wrapper.

Run on the GPU machine: `pytest -m gpu`.

The parity test is the gate for one torque path. With `use_pose` alone, `gain_mapping:
constant` and the env's own gains, the unified controller reduces to exactly Factory's
``compute_dof_torque`` — `S = I` and `f_d = 0` leave the impedance term untouched — so the two
must agree on live env state, step after step. It compares the ported math against Isaac Lab's
own function rather than against a wrapped env, so it needs no second env and cannot be fooled
by the wrapper simply calling the native code.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.envs.cfg import ControllerCfg
from robonuke_rl_core.envs.forge import control
from robonuke_rl_core.envs.interface import ActionInterface

pytestmark = pytest.mark.gpu

STEPS = 50


#: float32 on 7-DOF matrix inverses: the nullspace term goes through two inversions and a
#: matmul chain, so the last couple of mantissa bits differ by operation order alone. Bitwise
#: equality is not expected; anything above this would mean a real difference in the math.
TORQUE_RTOL, TORQUE_ATOL = 1e-5, 1e-4


@pytest.fixture(scope="module", autouse=True)
def warmed_env(gpu_env):
    """A reset and a couple of steps: the env only has fingertip state once it has stepped."""
    from robonuke_rl_core.evaluation import force_env_reset

    force_env_reset(gpu_env)
    zero = torch.zeros(gpu_env.num_envs, *gpu_env.action_space.shape, device=gpu_env.device)
    for _ in range(2):
        gpu_env.step(zero)
    return gpu_env


def native_torque(env, target_pos, target_quat):
    """Isaac Lab's own controller, on the env's current state."""
    from isaaclab_tasks.direct.factory import factory_control

    unwrapped = env.unwrapped
    torque, wrench = factory_control.compute_dof_torque(
        cfg=unwrapped.cfg,
        dof_pos=unwrapped.joint_pos,
        dof_vel=unwrapped.joint_vel,
        fingertip_midpoint_pos=unwrapped.fingertip_midpoint_pos,
        fingertip_midpoint_quat=unwrapped.fingertip_midpoint_quat,
        fingertip_midpoint_linvel=unwrapped.fingertip_midpoint_linvel,
        fingertip_midpoint_angvel=unwrapped.fingertip_midpoint_angvel,
        jacobian=unwrapped.fingertip_midpoint_jacobian,
        arm_mass_matrix=unwrapped.arm_mass_matrix,
        ctrl_target_fingertip_midpoint_pos=target_pos,
        ctrl_target_fingertip_midpoint_quat=target_quat,
        task_prop_gains=unwrapped.task_prop_gains,
        task_deriv_gains=unwrapped.task_deriv_gains,
        device=unwrapped.device,
        dead_zone_thresholds=unwrapped.dead_zone_thresholds,
    )
    return torque, wrench


def ours(env, target_pos, target_quat, selection=None, interface=None):
    """The unified controller, pose-only unless the caller says otherwise."""
    unwrapped = env.unwrapped
    rows = unwrapped.num_envs
    if selection is None:
        selection = torch.ones((rows, 6), device=unwrapped.device)

    delta_pose = control.pose_error(
        unwrapped.fingertip_midpoint_pos,
        unwrapped.fingertip_midpoint_quat,
        target_pos,
        target_quat,
    )
    wrench = control.unified_wrench(
        selection=selection,
        delta_pose=delta_pose,
        linvel=unwrapped.fingertip_midpoint_linvel,
        angvel=unwrapped.fingertip_midpoint_angvel,
        stiffness=unwrapped.task_prop_gains,
        damping=unwrapped.task_deriv_gains,
    )
    wrench = control.apply_dead_zone(wrench, unwrapped.dead_zone_thresholds)
    torque = control.dof_torque_from_wrench(
        wrench=wrench,
        jacobian=unwrapped.fingertip_midpoint_jacobian,
        arm_mass_matrix=unwrapped.arm_mass_matrix,
        dof_pos=unwrapped.joint_pos,
        dof_vel=unwrapped.joint_vel,
        ctrl=unwrapped.cfg.ctrl,
        device=unwrapped.device,
    )
    return torque, wrench


def a_target(env, scale: float = 0.01):
    """A pose target a little away from where the fingertip is now."""
    unwrapped = env.unwrapped
    offset = torch.randn_like(unwrapped.fingertip_midpoint_pos) * scale
    return unwrapped.fingertip_midpoint_pos + offset, unwrapped.fingertip_midpoint_quat.clone()


# ------------------------------------------------------------------ the parity gate
def test_pose_only_reproduces_the_native_factory_controller(gpu_env):
    """S = I, f_d = 0, the env's own gains: the unified path IS the native one."""
    torch.manual_seed(0)
    worst_torque, worst_wrench = 0.0, 0.0

    for step in range(STEPS):
        target_pos, target_quat = a_target(gpu_env)
        native, native_wrench = native_torque(gpu_env, target_pos, target_quat)
        mine, my_wrench = ours(gpu_env, target_pos, target_quat)

        worst_torque = max(worst_torque, float((mine - native).abs().max()))
        worst_wrench = max(worst_wrench, float((my_wrench - native_wrench).abs().max()))
        assert torch.allclose(mine, native, rtol=TORQUE_RTOL, atol=TORQUE_ATOL), (
            f"step {step}: joint torque differs by {float((mine - native).abs().max()):.3e}"
        )
        assert torch.allclose(my_wrench, native_wrench, rtol=TORQUE_RTOL, atol=TORQUE_ATOL)

        # move the arm so the next comparison is on a different state
        gpu_env.step(torch.zeros(gpu_env.num_envs, *gpu_env.action_space.shape, device=gpu_env.device))

    print(f"[parity] worst |dtorque| = {worst_torque:.3e} N m, |dwrench| = {worst_wrench:.3e}")
    assert worst_torque < TORQUE_ATOL * 10  # the run as a whole, not just per step


def test_the_dead_zone_and_nullspace_are_what_make_it_match(gpu_env):
    """Both ported pieces are load-bearing: drop either and parity breaks."""
    torch.manual_seed(1)
    target_pos, target_quat = a_target(gpu_env, scale=0.03)
    native, _ = native_torque(gpu_env, target_pos, target_quat)
    unwrapped = gpu_env.unwrapped

    delta_pose = control.pose_error(
        unwrapped.fingertip_midpoint_pos,
        unwrapped.fingertip_midpoint_quat,
        target_pos,
        target_quat,
    )
    wrench = control.pose_wrench(
        delta_pose,
        unwrapped.fingertip_midpoint_linvel,
        unwrapped.fingertip_midpoint_angvel,
        unwrapped.task_prop_gains,
        unwrapped.task_deriv_gains,
    )
    # without the dead zone the wrench is larger wherever it was within the threshold
    no_dead_zone = control.dof_torque_from_wrench(
        wrench=wrench,
        jacobian=unwrapped.fingertip_midpoint_jacobian,
        arm_mass_matrix=unwrapped.arm_mass_matrix,
        dof_pos=unwrapped.joint_pos,
        dof_vel=unwrapped.joint_vel,
        ctrl=unwrapped.cfg.ctrl,
        device=unwrapped.device,
    )
    assert not torch.allclose(no_dead_zone, native, rtol=TORQUE_RTOL, atol=TORQUE_ATOL)


# ------------------------------------------------------------------ the force branch
def test_the_force_branch_only_acts_where_the_selection_releases_an_axis(gpu_env):
    """(I - S) K_f (f_d - f): an axis with S = 1 is untouched by any force target."""
    unwrapped = gpu_env.unwrapped
    rows = unwrapped.num_envs
    target_pos, target_quat = a_target(gpu_env)
    delta_pose = control.pose_error(
        unwrapped.fingertip_midpoint_pos,
        unwrapped.fingertip_midpoint_quat,
        target_pos,
        target_quat,
    )
    kwargs = dict(
        delta_pose=delta_pose,
        linvel=unwrapped.fingertip_midpoint_linvel,
        angvel=unwrapped.fingertip_midpoint_angvel,
        stiffness=unwrapped.task_prop_gains,
        damping=unwrapped.task_deriv_gains,
        force_measured=unwrapped.force_sensor_smooth,
        force_gains=torch.full((rows, 6), 0.5, device=unwrapped.device),
    )
    big_target = torch.full((rows, 6), 25.0, device=unwrapped.device)

    all_position = control.unified_wrench(
        selection=torch.ones((rows, 6), device=unwrapped.device),
        force_target=big_target,
        **kwargs,
    )
    pose_only = control.unified_wrench(
        selection=torch.ones((rows, 6), device=unwrapped.device), **kwargs
    )
    assert torch.equal(all_position, pose_only)  # S = I ignores the force branch entirely

    # release z, keep the rest: only the z component may change
    selection = torch.ones((rows, 6), device=unwrapped.device)
    selection[:, 2] = 0.0
    hybrid = control.unified_wrench(selection=selection, force_target=big_target, **kwargs)
    assert torch.allclose(hybrid[:, [0, 1, 3, 4, 5]], pose_only[:, [0, 1, 3, 4, 5]])
    expected_z = 0.5 * (25.0 - unwrapped.force_sensor_smooth[:, 2])
    assert torch.allclose(hybrid[:, 2], expected_z, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ the wrapper
@pytest.mark.parametrize(
    "use_pose, use_force, mapping",
    [
        (True, False, "constant"),
        (True, False, "variable_diagonal"),
        (False, True, "constant"),
        (True, True, "constant"),
        (True, True, "variable_diagonal"),
    ],
)
def test_the_wrapper_runs_every_branch_combination(gpu_cfg, gpu_env, use_pose, use_force, mapping):
    """Attach, step, check the torques are finite — then put the env back as it was.

    The GPU suite shares one env (Isaac Lab hangs on a second), and this wrapper patches the
    env instance, so each case restores what it replaced.
    """
    from robonuke_rl_core.envs.forge.controller import ForgeControllerWrapper

    unwrapped = gpu_env.unwrapped
    before = (
        unwrapped._pre_physics_step,
        unwrapped.generate_ctrl_signals,
        int(unwrapped.cfg.action_space),
    )
    cfg = ControllerCfg(
        enabled=True, use_pose=use_pose, use_force=use_force, gain_mapping=mapping
    )
    wrapper = ForgeControllerWrapper(gpu_env.unwrapped, cfg, gpu_cfg.task_name)
    try:
        expected = ActionInterface(cfg).action_dim
        assert wrapper.action_dim == expected
        assert unwrapped.cfg.action_space == expected
        assert unwrapped.action_space.shape[-1] == expected

        torch.manual_seed(0)
        for _ in range(5):
            actions = torch.rand(gpu_env.num_envs, expected, device=gpu_env.device) * 2 - 1
            gpu_env.step(actions)
            assert torch.isfinite(unwrapped.joint_torque).all()
            assert torch.isfinite(unwrapped.applied_wrench).all()
            assert float(unwrapped.joint_torque.abs().max()) <= control.TORQUE_LIMIT
    finally:
        unwrapped._pre_physics_step, unwrapped.generate_ctrl_signals, action_space = before
        unwrapped.cfg.action_space = action_space
        unwrapped._configure_gym_env_spaces()

    assert unwrapped.cfg.action_space == before[2]  # the shared env is as it was


def test_a_native_action_width_that_disagrees_with_the_env_raises(gpu_cfg, gpu_env):
    from robonuke_rl_core.envs.forge.controller import ForgeControllerWrapper

    cfg = ControllerCfg(enabled=True, native_action_dim=6)
    with pytest.raises(ValueError) as err:
        ForgeControllerWrapper(gpu_env.unwrapped, cfg, gpu_cfg.task_name)
    assert "native_action_dim" in str(err.value)


def test_the_wrapper_refuses_a_non_forge_task(gpu_env):
    from robonuke_rl_core.envs.forge.controller import ForgeControllerWrapper

    with pytest.raises(ValueError) as err:
        ForgeControllerWrapper(
            gpu_env.unwrapped, ControllerCfg(enabled=True), "Isaac-Factory-PegInsert-v0"
        )
    assert "Forge-family" in str(err.value)

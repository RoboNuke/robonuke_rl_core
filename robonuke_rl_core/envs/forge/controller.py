"""`ForgeControllerWrapper`: the unified controller, attached to a Forge-family env.

Reads from the env (its interface contract):

* ``cfg.ctrl`` — ``default_task_prop_gains``, ``default_dof_pos_tensor``, ``kp_null``,
  ``kd_null``, ``pos_action_bounds``, the dead zone. These are Forge's own defaults, read at
  runtime rather than copied into our config, so ``task.cfg.ctrl.*`` tunes them in one place.
* ``fingertip_midpoint_pos`` / ``_quat`` / ``_linvel`` / ``_angvel`` / ``_jacobian``
* ``arm_mass_matrix``, ``joint_pos``, ``joint_vel``, ``ctrl_target_joint_pos``
* ``task_prop_gains`` / ``task_deriv_gains`` / ``dead_zone_thresholds``
* ``force_sensor_smooth`` — only when the force branch is on
* ``_pre_physics_step`` and ``generate_ctrl_signals``, which it replaces on the instance

**What it does not touch:** the pose *targets*. The env's own ``_apply_action`` still turns
the pose action into a target pose — EMA, position bounds, the upright constraint and all —
and the wrapper only replaces the step that turns a target into joint torque. That is what
makes pose-only control bit-comparable with the native controller instead of merely similar.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from ..interface import ActionInterface
from . import control
from .compat import require_forge_env

__all__ = ["ForgeControllerWrapper"]

#: what this wrapper needs the env to expose
ENV_READS = (
    "cfg.ctrl",
    "fingertip_midpoint_pos/_quat/_linvel/_angvel/_jacobian",
    "arm_mass_matrix",
    "joint_pos/joint_vel",
    "task_prop_gains/task_deriv_gains",
    "dead_zone_thresholds",
    "force_sensor_smooth (force branch only)",
    "generate_ctrl_signals",
)


class ForgeControllerWrapper(gym.Wrapper):
    """Replace the env's torque law with the unified controller.

    The action space grows to whatever :class:`~robonuke_rl_core.envs.interface.ActionLayout`
    says; models must therefore be built from the *wrapped* env's spaces.
    """

    def __init__(self, env: Any, cfg: Any, task_name: str) -> None:
        require_forge_env(task_name, type(self).__name__, ENV_READS)
        super().__init__(env)
        unwrapped = env.unwrapped
        self.device = unwrapped.device
        self.num_envs = int(unwrapped.num_envs)
        self.cfg = cfg
        self.interface = ActionInterface(cfg, device=self.device)
        self.layout = self.interface.layout

        native = int(getattr(unwrapped.cfg, "action_space", 6))
        if cfg.use_pose and int(cfg.native_action_dim) != native:
            raise ValueError(
                f"controller.native_action_dim is {cfg.native_action_dim} but {task_name} has a "
                f"native action space of {native}: set controller.native_action_dim: {native}. "
                "That first block is handed to the env's own action pipeline untouched, so the "
                "two must agree."
            )
        self._native_action_dim = native
        if cfg.use_force and not hasattr(unwrapped, "force_sensor_smooth"):
            raise RuntimeError(
                f"{type(self).__name__}: controller.use_force needs the Forge wrist force "
                "sensor (env.force_sensor_smooth), which this env does not expose"
            )

        # per-step buffers the control step reads
        self._selection = torch.ones((self.num_envs, 6), device=self.device)
        self._force_target = torch.zeros((self.num_envs, 6), device=self.device)
        self._force_gains = torch.zeros((self.num_envs, 6), device=self.device)
        self._stiffness: Optional[torch.Tensor] = None

        self._expand_action_space()
        self._install()
        print(f"[controller] {self.layout.describe()}", flush=True)

    # ------------------------------------------------------------------ setup
    def _expand_action_space(self) -> None:
        """Widen the advertised action space without widening the env's own action buffer.

        The space has to be set on the *unwrapped* env, because that is where skrl reads it
        (``IsaacLabWrapper.action_space`` returns ``_unwrapped.single_action_space``). But
        ``_configure_gym_env_spaces`` also re-instantiates ``env.actions`` at the new width,
        and on Forge that buffer is part of the **observation** — the policy sees
        ``prev_actions`` — so leaving it wide silently grows the observation (24 -> 30 here)
        and the first reset dies reshaping it. So the buffer is put back to the env's own
        width, and the env only ever receives an action of that width.
        """
        unwrapped = self.env.unwrapped
        native_actions = unwrapped.actions
        unwrapped.cfg.action_space = self.layout.action_dim
        unwrapped._configure_gym_env_spaces()
        unwrapped.actions = torch.zeros_like(native_actions)
        self.action_space = unwrapped.action_space
        self.single_action_space = getattr(unwrapped, "single_action_space", None)

    def _install(self) -> None:
        """Take over the two instance methods the controller owns."""
        unwrapped = self.env.unwrapped
        self._original_pre_physics_step = unwrapped._pre_physics_step
        self._original_generate = unwrapped.generate_ctrl_signals
        unwrapped._pre_physics_step = self._pre_physics_step
        unwrapped.generate_ctrl_signals = self._generate_ctrl_signals

    # ------------------------------------------------------------------ the step
    def _pre_physics_step(self, action: torch.Tensor) -> None:
        """Keep the controller's blocks, and let the env smooth and read its own.

        The env gets an action of **its own width** — the layout keeps that block at the
        front for exactly this reason — so its EMA, its position bounds, its upright
        constraint and its ``prev_actions`` observation channel all behave as they always did.

        The controller's own blocks are read **raw**, not from the env's smoothed buffer: a
        selection is 0/1 and an EMA of a switch is neither, and gains are already smooth in
        their geometric map.
        """
        action = action.to(self.device)
        pose, selection, force_target, stiffness, force_gains = self.interface.split(action)
        self._selection = selection
        self._force_target = force_target
        self._force_gains = force_gains
        self._stiffness = stiffness if self.cfg.gain_mapping != "constant" else None

        if self.cfg.use_pose:
            forwarded = pose
        else:
            # no pose branch: a zero action keeps the env's target pinned to the current
            # fingertip pose, so nothing in its pipeline fights the force law
            forwarded = action.new_zeros((action.shape[0], self._native_action_dim))
        self._original_pre_physics_step(forwarded)

    def _generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        ctrl_target_gripper_dof_pos: Any,
    ) -> None:
        """The env's own targets in, our torque out."""
        unwrapped = self.env.unwrapped

        delta_pose = control.pose_error(
            unwrapped.fingertip_midpoint_pos,
            unwrapped.fingertip_midpoint_quat,
            ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat,
        )
        stiffness, damping = self._gains()

        wrench = control.unified_wrench(
            selection=self._selection,
            delta_pose=delta_pose,
            linvel=unwrapped.fingertip_midpoint_linvel,
            angvel=unwrapped.fingertip_midpoint_angvel,
            stiffness=stiffness,
            damping=damping,
            force_target=self._force_target if self.cfg.use_force else None,
            force_measured=self.measured_wrench() if self.cfg.use_force else None,
            force_gains=self._force_gains if self.cfg.use_force else None,
        )
        wrench = control.apply_dead_zone(wrench, getattr(unwrapped, "dead_zone_thresholds", None))

        torque = control.dof_torque_from_wrench(
            wrench=wrench,
            jacobian=unwrapped.fingertip_midpoint_jacobian,
            arm_mass_matrix=unwrapped.arm_mass_matrix,
            dof_pos=unwrapped.joint_pos,
            dof_vel=unwrapped.joint_vel,
            ctrl=unwrapped.cfg.ctrl,
            device=self.device,
        )

        # the gripper stays on the env's PD controller, exactly as the native path leaves it
        unwrapped.joint_torque = torque
        unwrapped.applied_wrench = wrench
        unwrapped.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        unwrapped.joint_torque[:, 7:9] = 0.0
        unwrapped._robot.set_joint_position_target(unwrapped.ctrl_target_joint_pos)
        unwrapped._robot.set_joint_effort_target(unwrapped.joint_torque)

    # ------------------------------------------------------------------ pieces
    def _gains(self) -> tuple:
        """``(K, D)`` for this step.

        Under ``gain_mapping: constant`` these are the env's own ``task_prop_gains`` and
        ``task_deriv_gains`` — the Forge defaults, inherited by reading them — so pose-only
        control reproduces the native controller exactly. A variable mapping replaces K with
        the policy's and derives D from it.
        """
        unwrapped = self.env.unwrapped
        if self._stiffness is None:
            return unwrapped.task_prop_gains, unwrapped.task_deriv_gains
        return self._stiffness, self.interface.damping(self._stiffness)

    def measured_wrench(self) -> torch.Tensor:
        """The wrist force/torque the force loop closes on, in the fingertip frame."""
        return self.env.unwrapped.force_sensor_smooth

    @property
    def action_dim(self) -> int:
        return self.layout.action_dim

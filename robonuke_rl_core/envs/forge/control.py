"""The controller core: one torque path for pose, force, and anything between.

    tau = J^T [ (I - S) (K e_pose - D v) + S K_f (f_d - f) ] + nullspace

Ported from Factory's ``factory_control`` (the pose branch, the dead zone and the nullspace,
op for op) and from the hybrid force/position wrapper in
RoboNuke/generalized_hybrid_vic_action_space (the force branch).

**``S`` selects FORCE.** ``S = 1`` on an axis means that axis is force-controlled and ``S = 0``
means position-controlled — one convention, everywhere in this package: the policy's selection
bit, the action vector, the metrics and this matrix all say "1 is force". So ``S = 0`` with
``f_d = 0`` is exactly Factory's own controller, which is what the parity test in
``tests/envs/GPU`` checks, and why pose control does not need a separate path.

Isaac Lab's own quaternion helpers are used for the pose error rather than reimplemented: the
parity gate is only meaningful if the error term is computed the same way, down to the
shortest-path sign convention.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

__all__ = [
    "pose_error",
    "pose_wrench",
    "force_wrench",
    "unified_wrench",
    "apply_dead_zone",
    "dof_torque_from_wrench",
    "TORQUE_LIMIT",
]

#: Factory clamps joint torque to this, and so do we
TORQUE_LIMIT = 100.0


def pose_error(
    fingertip_pos: torch.Tensor,
    fingertip_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """``(num_envs, 6)`` position error and axis-angle orientation error.

    The geometric-Jacobian convention Factory uses: the target quaternion is flipped to the
    shortest path first, so the error never takes the long way round.
    """
    import isaacsim.core.utils.torch as torch_utils
    from isaaclab.utils.math import axis_angle_from_quat

    position = target_pos - fingertip_pos

    quat_dot = (target_quat * fingertip_quat).sum(dim=1, keepdim=True)
    target_quat = torch.where(quat_dot.expand(-1, 4) >= 0, target_quat, -target_quat)
    norm = torch_utils.quat_mul(fingertip_quat, torch_utils.quat_conjugate(fingertip_quat))[:, 0]
    inverse = torch_utils.quat_conjugate(fingertip_quat) / norm.unsqueeze(-1)
    orientation = axis_angle_from_quat(torch_utils.quat_mul(target_quat, inverse))

    return torch.cat((position, orientation), dim=1)


def pose_wrench(
    delta_pose: torch.Tensor,
    linvel: torch.Tensor,
    angvel: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
) -> torch.Tensor:
    """``K e - D v``, the impedance branch, as a ``(num_envs, 6)`` wrench."""
    wrench = torch.zeros_like(delta_pose)
    wrench[:, 0:3] = stiffness[:, 0:3] * delta_pose[:, 0:3] + damping[:, 0:3] * (0.0 - linvel)
    wrench[:, 3:6] = stiffness[:, 3:6] * delta_pose[:, 3:6] + damping[:, 3:6] * (0.0 - angvel)
    return wrench


def force_wrench(
    target: torch.Tensor, measured: torch.Tensor, gains: torch.Tensor
) -> torch.Tensor:
    """``K_f (f_d - f)``, the force branch: a pure proportional law on the wrench error."""
    return gains * (target - measured)


def unified_wrench(
    *,
    force_selection: torch.Tensor,
    delta_pose: torch.Tensor,
    linvel: torch.Tensor,
    angvel: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    force_target: Optional[torch.Tensor] = None,
    force_measured: Optional[torch.Tensor] = None,
    force_gains: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(I - S) (K e - D v) + S K_f (f_d - f)`` for a diagonal ``S``.

    ``force_selection`` is ``S``: ``(num_envs, 6)`` of 0/1 with **1 meaning force-controlled**.
    The whole expression is linear in it, so the degenerate cases are exact rather than
    approximate: ``S = 0`` leaves the impedance term alone, ``S = I`` leaves the force term
    alone. The argument is named rather than positional, and was renamed when the convention
    was fixed, so a caller that still means the old sense fails with a TypeError instead of
    inverting every axis in silence.
    """
    wrench = (1.0 - force_selection) * pose_wrench(
        delta_pose, linvel, angvel, stiffness, damping
    )
    if force_target is not None:
        if force_measured is None or force_gains is None:
            raise ValueError("the force branch needs force_measured and force_gains too")
        wrench = wrench + force_selection * force_wrench(
            force_target, force_measured, force_gains
        )
    return wrench


def apply_dead_zone(wrench: torch.Tensor, thresholds: Optional[torch.Tensor]) -> torch.Tensor:
    """Factory's low-force dead zone: below the threshold the wrench is zero, above it the
    threshold is subtracted. Models the unreliability of small commanded forces."""
    if thresholds is None:
        return wrench
    return torch.where(
        wrench.abs() < thresholds,
        torch.zeros_like(wrench),
        wrench.sign() * (wrench.abs() - thresholds),
    )


def dof_torque_from_wrench(
    *,
    wrench: torch.Tensor,
    jacobian: torch.Tensor,
    arm_mass_matrix: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    ctrl: Any,
    device: Any,
) -> torch.Tensor:
    """``J^T w`` plus the nullspace posture term, clamped — Factory's mapping, ported.

    ``ctrl`` is the env's own ``cfg.ctrl``: ``default_dof_pos_tensor``, ``kp_null`` and
    ``kd_null`` come from there, so the Forge defaults are inherited by reading them.
    """
    num_envs = dof_pos.shape[0]
    torque = torch.zeros((num_envs, dof_pos.shape[1]), device=device)
    jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
    torque[:, 0:7] = (jacobian_t @ wrench.unsqueeze(-1)).squeeze(-1)

    mass_inv = torch.inverse(arm_mass_matrix)
    mass_task = torch.inverse(jacobian @ mass_inv @ jacobian_t)
    jacobian_inv = mass_task @ jacobian @ mass_inv

    default_dof_pos = torch.tensor(ctrl.default_dof_pos_tensor, device=device).repeat(
        (num_envs, 1)
    )
    distance = default_dof_pos - dof_pos[:, :7]
    distance = (distance + math.pi) % (2 * math.pi) - math.pi  # to [-pi, pi]
    u_null = ctrl.kd_null * -dof_vel[:, :7] + ctrl.kp_null * distance
    u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
    identity = torch.eye(7, device=device).unsqueeze(0)
    torque_null = (identity - jacobian_t @ jacobian_inv) @ u_null
    torque[:, 0:7] += torque_null.squeeze(-1)

    return torch.clamp(torque, min=-TORQUE_LIMIT, max=TORQUE_LIMIT)

"""Orientation representations for observations.

A quaternion is a poor thing for a network to regress against: ``q`` and ``-q`` are the same
rotation, so the map from rotation to observation is discontinuous, and the unit-norm
constraint is not something a linear layer respects. The 6-D representation (Zhou et al.,
2019, "On the Continuity of Rotation Representations in Neural Networks") is the first two
columns of the rotation matrix — continuous, and free of the double cover.

The math here is env-agnostic and CPU-tested. Wiring it into a Forge-family env's observation
(rewriting ``obs_order``/``state_order`` and registering the extra dims) lives in
:mod:`robonuke_rl_core.envs.build`, because that part reads Forge's obs machinery.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["MODES", "QUAT", "ROT6D", "ROT6D_DIM", "QUAT_DIM", "quat_to_rot6d", "rot6d_suffix"]

QUAT = "quat"
ROT6D = "6d_rot_mat"
MODES = (QUAT, ROT6D)

QUAT_DIM = 4
ROT6D_DIM = 6

#: observation channels built from a quaternion get this suffix in their 6-D form
SUFFIX = "_rot6d"


def rot6d_suffix(name: str) -> str:
    """The 6-D channel name for a quaternion-valued channel."""
    return f"{name}{SUFFIX}"


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` quaternion ``(w, x, y, z)`` to ``(..., 6)``: R's first two columns.

    ``q`` and ``-q`` give the **same** output, which is the whole point: the representation
    is a function of the rotation, not of the quaternion that happens to encode it.
    """
    if quat.shape[-1] != QUAT_DIM:
        raise ValueError(f"expected a (..., 4) quaternion (w, x, y, z), got {tuple(quat.shape)}")
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # columns of the rotation matrix, in the (w, x, y, z) convention Isaac Lab uses
    first = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )
    second = torch.stack(
        (
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )
    return torch.cat((first, second), dim=-1)

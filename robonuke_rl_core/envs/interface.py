"""The action interface: what the policy's action vector means to the controller.

Pure math — torch only, no env, no Isaac Lab. The layout is **inferred** from the
controller's capabilities rather than configured:

===========  ==========  ======================================  ==========================
``use_pose`` ``use_force``  layout                                fixed blocks
===========  ==========  ======================================  ==========================
yes          no          ``[pose | gains?]``                     ``S = 0``, ``f_d = 0``
no           yes         ``[force | gains?]``                    ``S = I``
yes          yes         ``[pose | selection | force | gains?]``  —
===========  ==========  ======================================  ==========================

The selection block exists exactly when both branches are on, because that is the only case
where the policy has a choice to make. **A selection of 1 means that axis is
force-controlled**, 0 means position-controlled — the one convention this package uses, from
the policy's Bernoulli bit through the action vector and the metrics to ``S`` in the torque
law. So pose-only with ``S = 0`` and ``f_d = 0`` *is* pure impedance control: the degenerate
modes are exact, not approximations, which is why one torque path serves every configuration.

``gains?`` is empty under ``gain_mapping: constant``; under ``variable_diagonal`` it is one
action per axis of each live branch (``[K | K_f]``), so a new mapping is a new entry in
:meth:`ActionLayout.gain_dims`, not a rewrite.

The ``pose`` block is the env's own action vector, passed through untouched —
``controller.native_action_dim`` wide, which is 7 on Forge (6 pose dims plus the success
prediction its reward reads) and 6 on Factory. The env's own pipeline still applies its EMA,
its position bounds and its upright constraint to it; only the torque law is ours.

**How wide is "the force side"?** ``controller.force_axes`` is a length-6 binary mask of the
axes the force branch may take, so the selection, force-target and K_f blocks are all as wide
as its sum: ``[1, 1, 1, 0, 0, 0]`` is the usual 3-D hybrid (force on translation, orientation
always position-controlled), ``[1]*6`` is 6-D, ``[0, 0, 1, 0, 0, 0]`` is force on z alone. An
axis outside the mask is never offered to the policy and keeps ``S = 0``, i.e. stays
position-controlled.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

__all__ = ["ActionLayout", "ActionInterface", "geometric_scale"]

#: axes of every per-axis quantity, in order
AXES = ("x", "y", "z", "Rx", "Ry", "Rz")
#: width of the selection, force-target and per-axis gain blocks
AXIS_DIM = 6


def geometric_scale(
    actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Log-uniform map of actions in [-1, 1] onto ``[low, high]``.

    ``k = low * (high/low) ** ((a + 1) / 2)``: -1 gives ``low``, +1 gives ``high``, and the
    midpoint is the geometric mean, which is the right spacing for a stiffness that spans
    orders of magnitude. Actions are clamped first, because a sampled action can leave the
    tanh range. An axis whose ``low`` is below ``eps`` has no defined geometric map, so it
    returns 0 — a zero lower bound disables that axis instead of producing NaN.
    """
    t = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
    low = torch.as_tensor(low, dtype=actions.dtype, device=actions.device)
    high = torch.as_tensor(high, dtype=actions.dtype, device=actions.device)
    safe_low = torch.clamp(low, min=eps)
    scaled = safe_low * (high / safe_low).pow(t)
    return torch.where(low < eps, torch.zeros_like(scaled), scaled)


class ActionLayout:
    """Where each block sits in the action vector, for one controller config."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.use_pose = bool(cfg.use_pose)
        self.use_force = bool(cfg.use_force)
        self.gain_mapping = str(cfg.gain_mapping)
        self.pose_dim = int(cfg.native_action_dim) if self.use_pose else 0
        #: axes the force branch may take; everything else is position-controlled always
        self.force_axes = [index for index, flag in enumerate(cfg.force_axes) if flag]
        axes = len(self.force_axes) if self.use_force else 0
        #: a selection only exists when the policy has both branches to choose between, and
        #: then only over the force-eligible axes
        self.selection_dim = axes if (self.use_pose and self.use_force) else 0
        self.force_dim = axes
        self.gain_dim = self.gain_dims(cfg)

        offset = 0
        self.pose_slice = slice(offset, offset + self.pose_dim)
        offset += self.pose_dim
        self.selection_slice = slice(offset, offset + self.selection_dim)
        offset += self.selection_dim
        self.force_slice = slice(offset, offset + self.force_dim)
        offset += self.force_dim
        self.gain_slice = slice(offset, offset + self.gain_dim)
        offset += self.gain_dim
        self.action_dim = offset

        # within the gain block: K first, then K_f, one per live branch
        pose_gains = AXIS_DIM if (self.use_pose and self.gain_dim) else 0
        start = self.gain_slice.start
        self.pose_gain_slice = slice(start, start + pose_gains)
        self.force_gain_slice = slice(start + pose_gains, self.gain_slice.stop)

    @staticmethod
    def gain_dims(cfg: Any) -> int:
        """Action dims the gain mapping needs. A new mapping adds a branch here."""
        mapping = str(cfg.gain_mapping)
        if mapping == "constant":
            return 0  # gains come from the config, the policy does not touch them
        if mapping == "variable_diagonal":
            # one gain action per axis of each live branch: K over all six, K_f over the
            # force-eligible ones
            pose = AXIS_DIM if cfg.use_pose else 0
            force = sum(1 for flag in cfg.force_axes if flag) if cfg.use_force else 0
            return pose + force
        raise ValueError(f"unknown controller.gain_mapping {mapping!r}")

    @property
    def selection_indices(self) -> List[int]:
        """Action indices of the selection block — the actor's Bernoulli dims."""
        return list(range(self.selection_slice.start, self.selection_slice.stop))

    @property
    def pos_component_indices(self) -> List[int]:
        """Action index of the **pose** half of each gated pair, in selection order.

        Selection bit ``k`` governs force-eligible axis ``force_axes[k]``, whose two
        candidates are the pose target for that axis (inside the pose block, which is the
        env's own action vector, so the axis index *is* the offset) and force target ``k``
        (inside the force block, which is packed in ``force_axes`` order). MATCH needs both
        lists; they are derived here so no experiment ever hand-writes them, and they are
        empty when there is no selection to condition on.
        """
        if not self.selection_dim:
            return []
        return [self.pose_slice.start + axis for axis in self.force_axes]

    @property
    def force_component_indices(self) -> List[int]:
        """Action index of the **force** half of each gated pair, in selection order."""
        if not self.selection_dim:
            return []
        return list(range(self.force_slice.start, self.force_slice.stop))

    @property
    def selection_axis_names(self) -> List[str]:
        """Axis name per selection bit (``x``, ``z``, ``Rx``, ...), for logging."""
        return [AXES[axis] for axis in self.force_axes] if self.selection_dim else []

    def describe(self) -> str:
        """One line per block, for the wrapper to print when it attaches."""
        blocks = [
            ("pose", self.pose_slice),
            ("selection", self.selection_slice),
            ("force", self.force_slice),
            ("gains", self.gain_slice),
        ]
        parts = [
            f"{name}[{block.start}:{block.stop}]"
            for name, block in blocks
            if block.stop > block.start
        ]
        return f"action_dim={self.action_dim}: " + " | ".join(parts)


class ActionInterface:
    """Splits a policy action into what the controller needs, with the degenerate fills.

    :meth:`split` returns ``(pose_target, force_selection, force_target, K, K_f)`` for every
    configuration: pose-only fills ``S = 0`` and ``f_d = 0``, force-only fills ``S = I``, and
    ``constant`` gains ignore the actions entirely. Everything is ``(num_envs, ...)`` torch,
    built on the device the actions arrive on.
    """

    def __init__(self, cfg: Any, device: Any = None) -> None:
        self.cfg = cfg
        self.layout = ActionLayout(cfg)
        self.device = torch.device(device) if device is not None else None
        self._cache: dict = {}

    # ------------------------------------------------------------------ properties
    @property
    def action_dim(self) -> int:
        return self.layout.action_dim

    @property
    def selection_indices(self) -> List[int]:
        return self.layout.selection_indices

    @property
    def pos_component_indices(self) -> List[int]:
        return self.layout.pos_component_indices

    @property
    def force_component_indices(self) -> List[int]:
        return self.layout.force_component_indices

    @property
    def selection_axis_names(self) -> List[str]:
        return self.layout.selection_axis_names

    def _axis_index(self, like: torch.Tensor) -> torch.Tensor:
        """Indices of the force-eligible axes, as a tensor on the action's device."""
        key = ("force_axes", like.device)
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(
                self.layout.force_axes, dtype=torch.long, device=like.device
            )
        return self._cache[key]

    def _vector(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """A length-6 config vector as a tensor on the action's device, cached."""
        key = (name, like.device, like.dtype)
        if key not in self._cache:
            self._cache[key] = torch.as_tensor(
                list(getattr(self.cfg, name)), dtype=like.dtype, device=like.device
            )
        return self._cache[key]

    # ------------------------------------------------------------------ the split
    def split(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(pose_target, force_selection, force_target, K, K_f)`` from one batch of actions.

        ``force_selection`` is ``(num_envs, 6)`` of 0/1 — **1 means force-controlled** —
        matching ``S`` in ``tau = J^T [ (I - S) (K e - D v) + S K_f (f_d - f) ]``.
        ``K`` and ``K_f`` are ``(num_envs, 6)`` diagonals; the damping ``D`` is derived from
        ``K`` by :meth:`damping`, never commanded.
        """
        if actions.dim() != 2:
            raise ValueError(f"actions must be (num_envs, action_dim), got {tuple(actions.shape)}")
        if actions.shape[1] != self.layout.action_dim:
            raise ValueError(
                f"expected {self.layout.action_dim} action dims for this controller "
                f"({self.layout.describe()}), got {actions.shape[1]}"
            )
        layout = self.layout
        rows = actions.shape[0]
        zeros = actions.new_zeros((rows, AXIS_DIM))

        pose_target = (
            actions[:, layout.pose_slice]
            if layout.use_pose
            else actions.new_zeros((rows, int(self.cfg.native_action_dim)))
        )

        if layout.selection_dim:
            # S = 1 is "this axis is force-controlled". The actor emits selection actions as
            # +/-1 (its Bernoulli convention), so a positive action hands the axis to the
            # force law and a negative one keeps it on position. Axes outside force_axes are
            # not offered to the policy and stay position-controlled (S = 0).
            selection = zeros.clone()
            chosen = (actions[:, layout.selection_slice] > 0.0).to(actions.dtype)
            selection[:, self._axis_index(actions)] = chosen
        elif layout.use_pose:
            selection = zeros.clone()  # pose-only: S = 0, pure impedance everywhere
        else:
            selection = torch.ones_like(zeros)  # force-only: S = I, the force law owns every axis

        if layout.use_force:
            index = self._axis_index(actions)
            bounds = self._vector("force_target_bounds", actions)[index]
            force_target = zeros.clone()
            force_target[:, index] = actions[:, layout.force_slice].clamp(-1.0, 1.0) * bounds
        else:
            force_target = zeros.clone()

        return (pose_target, selection, force_target, *self.gains(actions))

    def gains(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(K, K_f)`` for this batch, per ``gain_mapping``."""
        layout = self.layout
        rows = actions.shape[0]
        mapping = layout.gain_mapping

        if mapping == "constant":
            # pose K comes from the env's own ctrl cfg at runtime; the interface reports None
            # so the wrapper fills it (see ForgeControllerWrapper), and K_f from this config
            pose_gains = None
            force_gains = self._vector("default_force_gains", actions).expand(rows, AXIS_DIM)
        elif mapping == "variable_diagonal":
            pose_gains = (
                geometric_scale(
                    actions[:, layout.pose_gain_slice],
                    self._vector("gain_min", actions),
                    self._vector("gain_max", actions),
                )
                if layout.pose_gain_slice.stop > layout.pose_gain_slice.start
                else None
            )
            if layout.force_gain_slice.stop > layout.force_gain_slice.start:
                index = self._axis_index(actions)
                # off-mask axes keep the constant K_f; they are position-controlled anyway
                # (S = 0 there), so the force term is zeroed whatever K_f says
                force_gains = self._vector("default_force_gains", actions).expand(
                    rows, AXIS_DIM
                ).clone()
                force_gains[:, index] = geometric_scale(
                    actions[:, layout.force_gain_slice],
                    self._vector("force_gain_min", actions)[index],
                    self._vector("force_gain_max", actions)[index],
                )
            else:
                force_gains = self._vector("default_force_gains", actions).expand(rows, AXIS_DIM)
        else:
            raise ValueError(f"unknown controller.gain_mapping {mapping!r}")

        if pose_gains is None:
            pose_gains = self.constant_pose_gains(actions)
        return pose_gains, force_gains

    def constant_pose_gains(self, actions: torch.Tensor) -> torch.Tensor:
        """Pose K when the policy does not command it.

        The wrapper overrides this with the env's ``ctrl.default_task_prop_gains`` so the
        Forge defaults are inherited at runtime; standalone (and in tests) it falls back to
        the configured ``gain_max``, which is the stiffest the config allows.
        """
        return self._vector("gain_max", actions).expand(actions.shape[0], AXIS_DIM)

    def damping(self, pose_gains: torch.Tensor) -> torch.Tensor:
        """``D = 2 * damping_ratio * sqrt(K)`` — never commanded, always derived."""
        ratio = float(getattr(self.cfg, "damping_ratio", 1.0))
        return 2.0 * ratio * torch.sqrt(pose_gains.clamp_min(0.0))

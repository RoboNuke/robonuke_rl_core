"""Env-side configuration: the `controller` and `wrappers` sections.

Plain dataclasses, no Isaac Lab import — ``config.py`` must load without it.

**The controller config holds only what Forge's own ctrl cfg does not.** Everything the
operational-space machinery already defines — ``ema_factor``, the dead zone,
``pos_action_bounds``, ``default_task_prop_gains``, ``kp_null``/``kd_null``,
``default_dof_pos_tensor`` — is read at runtime from ``env.unwrapped.cfg.ctrl``, so the Forge
defaults are inherited by *reading* them rather than by subclassing, and an experiment tunes
them in one place: ``task.cfg.ctrl.*``.

The action layout is **inferred from capabilities**, never configured. ``use_pose`` and
``use_force`` say which branches of

    tau = J^T [ S (K e_pose - D v) + (I - S) K_f (f_d - f) ] + nullspace

are live; the selection block exists exactly when both do, because that is the only case
where the policy has a choice to make. See :mod:`robonuke_rl_core.envs.interface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

__all__ = [
    "GAIN_MAPPINGS",
    "ORIENTATION_MODES",
    "ControllerCfg",
    "FragileCfg",
    "EfficientResetCfg",
    "ContactCfg",
    "OrientationCfg",
    "TaskMetricsCfg",
    "WrappersCfg",
]

#: how the policy's gain actions map to K / K_f
GAIN_MAPPINGS = ("constant", "variable_diagonal")

#: how an orientation is written into the observation
ORIENTATION_MODES = ("quat", "6d_rot_mat")

_AXES = "[x, y, z, Rx, Ry, Rz]"


def _check_six(name: str, value: List[float]) -> None:
    if len(value) != 6:
        raise ValueError(f"controller.{name} must be length 6 {_AXES}, got {value!r}")


def _check_order(low_name: str, low: List[float], high_name: str, high: List[float]) -> None:
    bad = [i for i, (a, b) in enumerate(zip(low, high)) if a > b]
    if bad:
        raise ValueError(
            f"controller requires {low_name} <= {high_name} on every axis; axes {bad} have "
            f"{low} !<= {high}"
        )


@dataclass
class ControllerCfg:
    """The unified controller: which branches run, and how gains come out of the actions.

    ``enabled: false`` (the default) attaches no wrapper at all and leaves the env's own
    controller in charge.
    """

    enabled: bool = False
    #: pose branch: the policy commands a pose target (a delta from the end-effector)
    use_pose: bool = True
    #: force branch: the policy commands a force/torque target
    use_force: bool = False
    #: ``constant`` (gains from the config, no action dims) or ``variable_diagonal``
    #: (per-axis geometric scaling, one gain action per axis of each live branch)
    gain_mapping: str = "constant"

    #: which axes the force branch may take, as a length-6 binary mask
    #: ``[x, y, z, Rx, Ry, Rz]``. This is what makes hybrid control 3-D or 6-D (or z-only):
    #: the selection, force-target and K_f action blocks are as wide as this mask's sum, and
    #: an axis outside it is position-controlled always (S = 0 there). Ignored, and refused,
    #: when ``use_force`` is false.
    #: The default is the **3-D hybrid**: force on the translation axes, orientation always
    #: position-controlled. A wrist wrench's torque channels are the noisy ones and a
    #: regulated torque is rarely what a contact task wants, so 6-D is opt-in.
    force_axes: List[int] = field(default_factory=lambda: [1, 1, 1, 0, 0, 0])

    #: the env's OWN action width, handed through untouched as the first block. Forge is 7
    #: (3 position + 3 rotation + the success prediction its reward reads), Factory is 6.
    #: The wrapper re-checks this against the live env and raises, naming the right value.
    native_action_dim: int = 7

    # ---- pose stiffness K, per axis [x, y, z, Rx, Ry, Rz] ----
    gain_min: List[float] = field(default_factory=lambda: [100.0, 100.0, 100.0, 5.0, 5.0, 5.0])
    gain_max: List[float] = field(
        default_factory=lambda: [2000.0, 2000.0, 2000.0, 100.0, 100.0, 100.0]
    )
    #: D = 2 * damping_ratio * sqrt(K); 1.0 is critical damping, which is what Factory uses
    damping_ratio: float = 1.0

    # ---- force stiffness K_f and the force target ----
    force_gain_min: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    force_gain_max: List[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    )
    #: K_f under ``gain_mapping: constant`` (Forge's ctrl cfg has no force gains of its own)
    default_force_gains: List[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    )
    #: the force action in [-1, 1] scales to +/- these, in N and Nm
    force_target_bounds: List[float] = field(
        default_factory=lambda: [50.0, 50.0, 50.0, 5.0, 5.0, 5.0]
    )

    # ------------------------------------------------------------------ rules
    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return  # the section is inert; nothing it says can take effect
        if not (self.use_pose or self.use_force):
            raise ValueError(
                "controller.enabled is true but both controller.use_pose and "
                "controller.use_force are false: the controller would have no branch to run. "
                "Enable one, or set controller.enabled: false to keep the env's own controller."
            )
        if self.gain_mapping not in GAIN_MAPPINGS:
            raise ValueError(
                f"controller.gain_mapping must be one of {GAIN_MAPPINGS}, got "
                f"{self.gain_mapping!r}"
            )
        if self.native_action_dim < 1:
            raise ValueError(
                f"controller.native_action_dim must be >= 1, got {self.native_action_dim}"
            )
        if self.damping_ratio <= 0.0:
            raise ValueError(
                f"controller.damping_ratio must be > 0 (D = 2*zeta*sqrt(K)), got "
                f"{self.damping_ratio}"
            )
        for name in ("gain_min", "gain_max", "force_gain_min", "force_gain_max",
                     "default_force_gains", "force_target_bounds"):
            _check_six(name, getattr(self, name))
        _check_six("force_axes", self.force_axes)
        if not set(self.force_axes) <= {0, 1}:
            raise ValueError(
                f"controller.force_axes must be binary {_AXES}, got {self.force_axes!r}"
            )
        if self.use_force and sum(self.force_axes) < 1:
            raise ValueError(
                "controller.use_force is true but controller.force_axes selects no axis: the "
                f"force branch would have nothing to act on. Got {self.force_axes!r} — e.g. "
                "[1, 1, 1, 0, 0, 0] for the usual 3-D hybrid, [1, 1, 1, 1, 1, 1] for 6-D."
            )
        if self.use_force and not self.use_pose and sum(self.force_axes) != 6:
            raise ValueError(
                "force-only control (use_pose: false) needs every axis force-eligible, or the "
                f"axes outside controller.force_axes would have no controller at all; got "
                f"{self.force_axes!r} (the default is the 3-D hybrid mask). Set "
                "controller.force_axes: [1, 1, 1, 1, 1, 1], or enable use_pose for a "
                "partial mask."
            )
        _check_order("gain_min", self.gain_min, "gain_max", self.gain_max)
        _check_order("force_gain_min", self.force_gain_min, "force_gain_max", self.force_gain_max)
        if any(value < 0 for value in self.gain_min + self.force_gain_min):
            raise ValueError("controller gain bounds must be >= 0")
        if any(value <= 0 for value in self.force_target_bounds):
            raise ValueError(
                f"controller.force_target_bounds must be > 0 on every axis, got "
                f"{self.force_target_bounds}"
            )

        self._check_force_fields_unused()
        self._check_action_dims(cfg)

    def _check_force_fields_unused(self) -> None:
        """A force setting with the force branch off is a mistake, not a no-op."""
        if self.use_force:
            return
        defaults = ControllerCfg()
        changed = [
            name
            for name in ("force_gain_min", "force_gain_max", "default_force_gains",
                         "force_target_bounds", "force_axes")
            if list(getattr(self, name)) != list(getattr(defaults, name))
        ]
        if changed:
            raise ValueError(
                f"controller.use_force is false but {changed} were set: those fields would be "
                "silently ignored. Enable controller.use_force, or drop them from the config."
            )

    def _check_action_dims(self, cfg: Any) -> None:
        """The selection dims must be exactly the actor's Bernoulli dims (decision 4).

        Selection actions are 0/1 by construction, so the actor's Bernoulli head and the
        controller's selection block have to be the same indices. A shifted index is a silent
        disaster at runtime — a continuous dim treated as a switch — so it fails here.
        """
        from .interface import ActionLayout

        model = getattr(cfg, "model", None) if cfg is not None else None
        actor = getattr(model, "actor", None)
        if actor is None:
            return  # a config without the model section (the config tests' own fixtures)

        layout = ActionLayout(self)
        expected = list(layout.selection_indices)
        declared = sorted(int(dim) for dim in (getattr(actor, "bernoulli_action_dims", None) or []))
        if declared != expected:
            where = (
                f"{expected}"
                if expected
                else "[] — there is no selection block unless use_pose and use_force are "
                "both true"
            )
            raise ValueError(
                f"model.actor.bernoulli_action_dims must be exactly the controller's selection "
                f"block {where}, got {declared}. Selection actions are 0/1 by construction; any "
                "other index would turn a continuous action into a switch."
            )
        forced = sorted(int(dim) for dim in (getattr(actor, "force_zero_action_dims", None) or []))
        owned = set(range(layout.action_dim))
        collision = sorted(set(forced) & owned)
        if collision:
            raise ValueError(
                f"model.actor.force_zero_action_dims {collision} collide with controller-owned "
                f"action dims (0..{layout.action_dim - 1}): those dims drive the controller and "
                "cannot be pinned to zero."
            )


# ------------------------------------------------------------------------------ wrappers
@dataclass
class FragileCfg:
    """Terminate an env when the held object's contact load breaks it."""

    enabled: bool = False
    #: ``[magnitude]`` for the force-magnitude mode, or ``[shear, normal]`` with
    #: ``direction_break_force`` — the components are resolved on the live peg axis
    break_force: List[float] = field(default_factory=lambda: [100.0])
    #: split the load into shear and axial components instead of one magnitude
    direction_break_force: bool = False
    #: also fail an episode that loses contact once it has been made
    require_contact: bool = False
    #: steps at the start of an episode where loss of contact cannot fail it
    require_contact_grace_steps: int = 5
    #: consecutive out-of-contact steps before loss of contact counts as a break
    require_contact_debounce_steps: int = 3

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            return
        expected = 2 if self.direction_break_force else 1
        if len(self.break_force) != expected:
            raise ValueError(
                f"wrappers.fragile.break_force must be length {expected} "
                f"({'[shear, normal]' if self.direction_break_force else '[magnitude]'}) with "
                f"direction_break_force={self.direction_break_force}, got {self.break_force!r}"
            )
        if any(value <= 0 for value in self.break_force):
            raise ValueError(
                f"wrappers.fragile.break_force must be > 0, got {self.break_force!r}"
            )
        for name in ("require_contact_grace_steps", "require_contact_debounce_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"wrappers.fragile.{name} must be >= 0, got {getattr(self, name)}")
        if self.require_contact_debounce_steps < 1:
            raise ValueError(
                "wrappers.fragile.require_contact_debounce_steps must be >= 1 (1 = any single "
                "out-of-contact step breaks)"
            )


@dataclass
class EfficientResetCfg:
    """Reset a finished env by teleporting it onto another env's fresh state.

    A training tool: it reuses initial conditions across envs instead of paying a full
    physics reset. Eval refuses it, because the accounting needs independently sampled
    conditions.
    """

    enabled: bool = False

    def validate(self, cfg: Any) -> None:
        return


@dataclass
class ContactCfg:
    """Per-axis in-contact flags from a contact sensor on the held asset."""

    enabled: bool = False
    #: |f| above this on an end-effector axis counts as contact, in N
    force_threshold: float = 1.0
    #: append the 3 flags to the policy observation (grows the observation space)
    append_to_policy_obs: bool = False
    #: append the 3 flags to the critic state (asymmetric tasks only)
    append_to_critic_state: bool = False
    held_prim_expr: str = "/World/envs/env_.*/HeldAsset"
    fixed_prim_expr: str = "/World/envs/env_.*/FixedAsset"

    def validate(self, cfg: Any) -> None:
        if not self.enabled:
            if self.append_to_policy_obs or self.append_to_critic_state:
                raise ValueError(
                    "wrappers.contact.append_to_policy_obs / append_to_critic_state need "
                    "wrappers.contact.enabled: true — there would be nothing to append"
                )
            return
        if self.force_threshold <= 0:
            raise ValueError(
                f"wrappers.contact.force_threshold must be > 0, got {self.force_threshold}"
            )


@dataclass
class OrientationCfg:
    """How orientations are written into the observation."""

    #: ``quat`` (the env's own (w, x, y, z)) or ``6d_rot_mat`` (the first two columns of R,
    #: Zhou et al. 2019 — continuous and free of the quaternion double cover)
    mode: str = "quat"

    def validate(self, cfg: Any) -> None:
        if self.mode not in ORIENTATION_MODES:
            raise ValueError(
                f"wrappers.orientation.mode must be one of {ORIENTATION_MODES}, got {self.mode!r}"
            )


@dataclass
class TaskMetricsCfg:
    """Per-agent task outcomes: success, termination cause, reward terms, prediction quality.

    On by default. The env logs the same quantities itself, but as scalars already averaged
    over every env, which mixes the agents training side by side; this publishes them per
    agent. Skipped for a task that is not Forge-family.
    """

    enabled: bool = True

    def validate(self, cfg: Any) -> None:
        return


@dataclass
class WrappersCfg:
    """The env wrappers, each off by default."""

    fragile: FragileCfg = field(default_factory=FragileCfg)
    efficient_reset: EfficientResetCfg = field(default_factory=EfficientResetCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    orientation: OrientationCfg = field(default_factory=OrientationCfg)
    task_metrics: TaskMetricsCfg = field(default_factory=TaskMetricsCfg)

    def validate(self, cfg: Any) -> None:
        for group in (self.fragile, self.efficient_reset, self.contact, self.orientation,
                      self.task_metrics):
            group.validate(cfg)
        if self.fragile.enabled and self.fragile.require_contact and not self.contact.enabled:
            raise ValueError(
                "wrappers.fragile.require_contact needs wrappers.contact.enabled: true — the "
                "loss-of-contact failure reads the contact sensor's flags"
            )
        if self.fragile.enabled and not self.efficient_reset.enabled:
            raise ValueError(
                "wrappers.fragile.enabled needs wrappers.efficient_reset.enabled: true. A peg "
                "breaks in one env at a time, so the env resets a SUBSET of envs mid-episode, "
                "and Factory/Forge's reset path is written assuming every env resets together "
                "(randomize_initial_state samples len(env_ids) rows and then assigns them to "
                "the full buffer, which raises, and builds the rest at num_envs). The "
                "efficient-reset wrapper is what makes a partial reset safe, which is the "
                "whole reason it exists — it is not an optimization."
            )

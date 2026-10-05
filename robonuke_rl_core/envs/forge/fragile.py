"""`ForgeFragileObjectWrapper`: break the held object and end the episode.

Reads from the env: ``force_sensor_smooth`` (the wrist wrench), ``held_quat`` (for the peg's
live long axis), ``episode_length_buf``, ``reset_buf``, ``in_contact`` (loss-of-contact mode
only, published by :class:`~robonuke_rl_core.envs.forge.contact.ForgeContactSensorWrapper`),
and ``_get_dones``, which it wraps.

Two failure modes, both ported from the fragile-object wrapper in
RoboNuke/generalized_hybrid_vic_action_space:

* **force** — either the force magnitude exceeds one threshold, or, with
  ``direction_break_force``, the axial and shear components measured against the peg's own
  axis exceed their own thresholds. A thin peg shears long before it crushes, so one
  magnitude number cannot express both.
* **loss of contact** — once the peg has touched down, losing contact on every axis for
  ``require_contact_debounce_steps`` consecutive steps ends the episode. A grace period at
  the start of an episode keeps the reset rebound from counting.

The projection is pure tensor math and is tested on hand-computed cases on CPU; the wrapper
around it needs a live env.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import torch

from .compat import require_forge_env

__all__ = ["ForgeFragileObjectWrapper", "split_axial_shear", "ENV_READS"]

ENV_READS = (
    "force_sensor_smooth",
    "held_quat",
    "episode_length_buf",
    "reset_buf",
    "in_contact (loss-of-contact mode)",
    "_get_dones",
)

#: the held asset's long axis in its own frame — Factory/Forge spawn the peg along +z
PEG_AXIS_LOCAL = (0.0, 0.0, 1.0)


def split_axial_shear(
    force: torch.Tensor, axis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a force into its component along ``axis`` and the magnitude of the rest.

    :param force: ``(num_envs, 3)`` force vector.
    :param axis: ``(num_envs, 3)`` unit axis (renormalized here for safety).
    :returns: ``(axial_magnitude, shear_magnitude)``, each ``(num_envs,)``.
    """
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
    axial = (force * axis).sum(dim=1)
    shear = torch.linalg.norm(force - axial.unsqueeze(-1) * axis, dim=1)
    return axial.abs(), shear


class ForgeFragileObjectWrapper(gym.Wrapper):
    """Terminate envs whose held object breaks."""

    def __init__(self, env: Any, cfg: Any, task_name: str) -> None:
        require_forge_env(task_name, type(self).__name__, ENV_READS)
        super().__init__(env)
        unwrapped = env.unwrapped
        self.cfg = cfg
        self.device = unwrapped.device
        self.num_envs = int(unwrapped.num_envs)

        if cfg.direction_break_force:
            self.shear_force, self.normal_force = (float(v) for v in cfg.break_force)
            self.break_force = None
        else:
            self.break_force = float(cfg.break_force[0])
            self.shear_force = self.normal_force = None
        self.require_contact = bool(cfg.require_contact)
        if self.require_contact and not hasattr(unwrapped, "in_contact"):
            raise RuntimeError(
                f"{type(self).__name__}: wrappers.fragile.require_contact needs the contact "
                "sensor's flags (env.in_contact). Enable wrappers.contact and let build_env "
                "attach it before this wrapper."
            )

        self._axis_local = torch.tensor(PEG_AXIS_LOCAL, device=self.device)
        self._contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_contact = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._broke = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._original_get_dones = unwrapped._get_dones
        unwrapped._get_dones = self._get_dones

    # ------------------------------------------------------------------ the check
    def measured_force(self) -> torch.Tensor:
        """The force the break is judged on: the wrist wrench's linear part."""
        return self.env.unwrapped.force_sensor_smooth[:, :3]

    def peg_axis(self) -> torch.Tensor:
        """The held object's long axis in world frame, from its live orientation."""
        from isaacsim.core.utils.torch import quat_apply

        return quat_apply(
            self.env.unwrapped.held_quat, self._axis_local.expand(self.num_envs, 3)
        )

    def force_violations(self) -> torch.Tensor:
        force = self.measured_force()
        if self.break_force is not None:
            return torch.linalg.norm(force, dim=1) >= self.break_force
        axial, shear = split_axial_shear(force, self.peg_axis())
        return (axial >= self.normal_force) | (shear >= self.shear_force)

    def contact_violations(self) -> torch.Tensor:
        """Loss of contact, once contact has been made and the grace period is over."""
        unwrapped = self.env.unwrapped
        in_contact = unwrapped.in_contact.any(dim=1)
        self._contacted |= in_contact
        self._out_of_contact = torch.where(
            in_contact,
            torch.zeros_like(self._out_of_contact),
            self._out_of_contact + 1,
        )
        past_grace = unwrapped.episode_length_buf > self.cfg.require_contact_grace_steps
        debounced = self._out_of_contact >= self.cfg.require_contact_debounce_steps
        return self._contacted & past_grace & debounced

    def _get_dones(self):
        terminated, time_out = self._original_get_dones()
        broke = self.force_violations()
        if self.require_contact:
            broke = broke | self.contact_violations()
        self._broke = broke
        return terminated | broke, time_out

    # ------------------------------------------------------------------ the step
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if isinstance(infos, dict):
            force = self.measured_force()
            metrics = infos.setdefault("metrics_to_log", {})
            metrics["fragile/force"] = torch.linalg.norm(force, dim=1)
            episode = infos.setdefault("episode_metrics_to_log", {})
            episode["fragile/broke"] = self._broke.to(force.dtype)
        # a reset clears the contact latch, so the next episode has to earn contact again
        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1))
        self._contacted &= ~done
        self._out_of_contact = torch.where(
            done, torch.zeros_like(self._out_of_contact), self._out_of_contact
        )
        return observations, rewards, terminated, truncated, infos

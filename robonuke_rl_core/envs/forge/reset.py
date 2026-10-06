"""`ForgeEfficientResetWrapper`: reset one env without making every other env pay.

Reads from the env: ``_reset_idx`` (which it wraps), ``scene.get_state`` /
``scene.reset_to``, ``episode_length_buf``, the Factory and Forge per-env bookkeeping tensors
listed below, and ``_compute_intermediate_values``.

Factory and Forge write ``_reset_idx`` assuming every env resets together: it re-randomizes
assets and then lets the whole scene settle for a fixed number of physics steps. With
per-env termination — which a fragile peg guarantees — that cost is paid by all the envs that
were still running, every time one of them breaks.

Ported from the efficient-reset wrapper in RoboNuke/generalized_hybrid_vic_action_space:

* a **full** reset (every env, or the first one) runs the normal chain and then caches the
  post-reset scene state and bookkeeping;
* a **partial** reset runs only ``DirectRLEnv._reset_idx`` (scene/event/noise, no settling)
  and then *teleports* each finished env onto a randomly chosen donor env's cached state.

The donor state is read and written **relative** to the env origin, so a donor's pose lands
correctly on the recipient's own origin. Everything else the lightweight path skips — the
per-env bookkeeping tensors, Forge's per-episode randomization, the force-sensor smoothing —
is copied or cleared explicitly.

**Required whenever an env can end early**, i.e. whenever ``wrappers.fragile`` is on
(``WrappersCfg.validate`` enforces it): Factory/Forge's own reset path is written assuming
every env resets together, so a partial reset through it corrupts the envs still mid-episode
— or raises outright. Making a partial reset safe is the whole reason this wrapper exists.

A teleported episode shares its initial condition with a donor, so it is **not**
independently sampled. That does not disturb eval: an env stops being tracked the moment its
first episode of the round closes (``EvalAccounting.valid``), so every teleported episode is
already masked out of the returns, the metrics, the trace and the video. Each round still
begins with a global ``force_env_reset``, which is where eval's independent conditions come
from.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from .compat import require_forge_env

__all__ = ["ForgeEfficientResetWrapper", "FACTORY_ATTRS", "FORGE_ATTRS", "ENV_READS"]

ENV_READS = (
    "_reset_idx",
    "scene.get_state/reset_to",
    "episode_length_buf",
    "_compute_intermediate_values",
    "the Factory/Forge per-env bookkeeping tensors",
)

#: Factory per-env bookkeeping the lightweight reset path does not restore. All of it is
#: either env-relative or frame-independent, so a donor's row can be copied as it stands.
FACTORY_ATTRS = (
    "fixed_pos_obs_frame",
    "init_fixed_pos_obs_noise",
    "prev_joint_pos",
    "prev_fingertip_pos",
    "prev_fingertip_quat",
    "actions",
    "prev_actions",
    "ee_angvel_fd",
    "ee_linvel_fd",
)

#: Forge's per-episode randomization, re-sampled for every env on a full reset
FORGE_ATTRS = (
    "ema_factor",
    "task_prop_gains",
    "task_deriv_gains",
    "pos_threshold",
    "rot_threshold",
    "dead_zone_thresholds",
    "flip_quats",
    "contact_penalty_thresholds",
)

#: smoothing state that must not survive a reset, or a stale force rides into the new episode
FORCE_SMOOTHING = ("force_sensor_world_smooth", "force_sensor_smooth")


class ForgeEfficientResetWrapper(gym.Wrapper):
    """Teleport-based partial reset."""

    def __init__(self, env: Any, cfg: Any, task_name: str) -> None:
        require_forge_env(task_name, type(self).__name__, ENV_READS)
        super().__init__(env)
        unwrapped = env.unwrapped
        self.cfg = cfg
        self.device = unwrapped.device
        self.num_envs = int(unwrapped.num_envs)

        self._cached_state: Optional[dict] = None
        self._cached_attrs: dict = {}
        self._full_reset_idx = unwrapped._reset_idx
        self._direct_reset_idx = self._find_direct_reset(unwrapped)
        unwrapped._reset_idx = self._reset_idx

    @staticmethod
    def _find_direct_reset(unwrapped: Any):
        """``DirectRLEnv._reset_idx`` — the lightweight path, without Factory's settling."""
        for klass in type(unwrapped).__mro__:
            if klass.__name__ == "DirectRLEnv" and "_reset_idx" in klass.__dict__:
                return klass._reset_idx.__get__(unwrapped, type(unwrapped))
        raise RuntimeError(
            "ForgeEfficientResetWrapper could not find DirectRLEnv._reset_idx in the env's MRO; "
            "the partial reset needs the base class's lightweight reset."
        )

    # ------------------------------------------------------------------ the reset
    def _reset_idx(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if env_ids.numel() >= self.num_envs or self._cached_state is None:
            self._full_reset_idx(env_ids)
            self._cache()
            return
        self._direct_reset_idx(env_ids)
        self.env.unwrapped.episode_length_buf[env_ids] = 0
        self._teleport(env_ids)

    def _cache(self) -> None:
        """Snapshot a freshly reset scene: the donors every later partial reset draws from."""
        unwrapped = self.env.unwrapped
        self._cached_state = unwrapped.scene.get_state(is_relative=True)
        self._cached_attrs = {
            name: getattr(unwrapped, name).clone()
            for name in (*FACTORY_ATTRS, *FORGE_ATTRS)
            if hasattr(unwrapped, name) and torch.is_tensor(getattr(unwrapped, name))
        }

    def _teleport(self, env_ids: torch.Tensor) -> None:
        unwrapped = self.env.unwrapped
        count = int(env_ids.numel())
        donors = torch.randint(0, self.num_envs, (count,), device=self.device)

        unwrapped.scene.reset_to(
            self._donor_state(donors), env_ids=env_ids, is_relative=True
        )
        for name, cached in self._cached_attrs.items():
            getattr(unwrapped, name)[env_ids] = cached[donors]
        for name in FORCE_SMOOTHING:
            if hasattr(unwrapped, name):
                getattr(unwrapped, name)[env_ids] = 0.0

        self._refresh_derived(env_ids)

    def _donor_state(self, donors: torch.Tensor):
        """The cached scene state with every per-env row replaced by its donor's."""

        def pick(value):
            if torch.is_tensor(value) and value.shape[:1] == (self.num_envs,):
                return value[donors]
            if isinstance(value, dict):
                return {key: pick(item) for key, item in value.items()}
            return value

        return pick(self._cached_state)

    def _refresh_derived(self, env_ids: torch.Tensor) -> None:
        """Recompute the env's cached kinematics after the teleport.

        ``scene.reset_to`` wrote new poses to sim, but the env's own fingertip/asset tensors
        still hold the pre-teleport ones, and the next control step runs before the env
        recomputes them — so the first post-reset action would chase the pose the episode
        ended at. Recomputing touches every env's finite-difference velocity base, so the
        still-running envs' rows are put back: only the reset envs should see a fresh (zero)
        velocity.
        """
        unwrapped = self.env.unwrapped
        recompute = getattr(unwrapped, "_compute_intermediate_values", None)
        if not callable(recompute):
            return
        survivors = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        survivors[env_ids] = False
        saved = {
            name: getattr(unwrapped, name).clone()
            for name in ("ee_linvel_fd", "ee_angvel_fd", "prev_fingertip_pos",
                         "prev_fingertip_quat", "prev_joint_pos")
            if hasattr(unwrapped, name) and torch.is_tensor(getattr(unwrapped, name))
        }
        recompute(dt=unwrapped.physics_dt)
        for name, value in saved.items():
            getattr(unwrapped, name)[survivors] = value[survivors]

"""Every action-noise draw of a deterministic actor (DDPG), in one place.

Two phases, one set of options each (:class:`~robonuke_rl_core.learners.cfg.NoiseCfg`): ``warmup`` (until the learner
starts learning, and the target smoothing of its critic warm-up burst) and ``train`` (from then on, and every later
critic update's target smoothing). Eval acts without noise. One draw, for either phase::

    d     = N(0, 1) | U(-1, 1)                     distribution (held for k steps per agent with ``repeat``)
    noise = clip(d * scale, +-clip)                scale: std / half width; clip on the scaled draw
    a     = actor(s) + noise  |  mid + noise       center

``scale`` is the config's (one number for every action dim or a list, one per dim; a linear schedule with
``scale_end``), or with ``scale: registered`` a project's per-row, per-dim function of the raw observation and the phase
(:attr:`ActionNoise.scale_fn`, set by the trainer from ``register_noise_scale``), ``(rows, A)`` in r units. Every clip is
one number or one per dim.

Target smoothing (TD3's) in a critic update: ``a' = mu'(s') + clip(N(0, target_scale), +-target_clip)``. Its numbers
live in device buffers set before each update (:meth:`ActionNoise.load_smoothing`), so one captured CUDA graph serves
both phases. ``target_scale: registered``: the std per row is the registered scale of the next observation ``s'`` for the
``train`` phase (the training noise's own std there), so the smoothing is the acting noise, as with one number for both.

:class:`HeldDraw` is the correlated-noise hold (FlashSAC's), shared with SAC's ``noise_repeat``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

__all__ = ["HeldDraw", "ActionNoise", "PHASES", "REGISTERED"]

PHASES = ("warmup", "train")
#: ``NoiseCfg.scale``'s value for a scale computed per row by a project's function (``register_noise_scale``)
REGISTERED = "registered"


class HeldDraw:
    """FlashSAC's ``_sample_flashsac_actions`` per agent: each agent holds one draw for ``k`` steps, ``k`` ~ a Zeta(``s``)
    truncated at ``max_len`` (``P(k) ~ k^-s``). Every call makes a fresh draw and a fresh length; an agent whose held
    draw has been used ``k`` times (or never drawn) takes them, the others keep theirs. Rows are agent-major (agent
    ``i`` owns rows ``[i * rows / num_agents, (i + 1) * rows / num_agents)``). Episode ends do not cut a hold short."""

    def __init__(self, num_agents: int, zeta_s: float, max_len: int, device: Any) -> None:
        self.num_agents = int(num_agents)
        self.device = device
        lengths = torch.arange(1, int(max_len) + 1, dtype=torch.float32, device=device)
        pmf = lengths ** (-float(zeta_s))
        #: FlashSAC's _build_truncated_zeta_cdf
        self.cdf = torch.cumsum(pmf / pmf.sum(), dim=0)
        self.held: Optional[torch.Tensor] = None
        self.count = torch.zeros(self.num_agents, dtype=torch.long, device=device)
        self.length = torch.ones(self.num_agents, dtype=torch.long, device=device)

    def __call__(self, fresh: torch.Tensor) -> torch.Tensor:
        """``fresh`` ``(rows, width)``: this step's new draw; returns each row's held one."""
        rows = fresh.shape[0]
        if self.held is None or self.held.shape != fresh.shape:
            self.held = torch.zeros_like(fresh)
            self.count.zero_()
        redraw = (self.count == 0) | (self.count >= self.length)  # (agents,)
        uniform = torch.rand(self.num_agents, 1, device=self.device)
        length = torch.argmax((uniform < self.cdf).to(torch.int32), dim=1) + 1
        per_row = redraw.repeat_interleave(rows // self.num_agents).unsqueeze(-1)
        self.held = torch.where(per_row, fresh, self.held)
        self.length = torch.where(redraw, length, self.length)
        self.count = torch.where(redraw, torch.zeros_like(self.count), self.count) + 1
        return self.held


class ActionNoise:
    """The two phases' noise (``warmup``, ``train``: :class:`~robonuke_rl_core.learners.cfg.NoiseCfg`) for a
    deterministic actor over ``action_range`` ``(lo, hi)``. ``scale_fn`` is read when a phase's ``scale`` is
    ``registered``: ``(observations, phase) -> (rows, A)``, the std / half width per row and dim in r units from the raw
    observations (the trainer sets it from ``register_noise_scale``), or None."""

    def __init__(self, warmup: Any, train: Any, *, action_range, num_agents: int, device: Any, width: int = 1) -> None:
        self.cfgs = {"warmup": warmup, "train": train}
        self.width = int(width)
        for phase, cfg in self.cfgs.items():
            for name in ("scale", "scale_end", "clip", "target_scale", "target_clip"):
                value = getattr(cfg, name)
                if value is None or (name in ("scale", "target_scale") and value == REGISTERED):
                    continue
                if not isinstance(value, (int, float)) and len(value) not in (1, self.width):
                    raise ValueError(f"ddpg.{phase}_noise.{name} has {len(value)} numbers; the action is {self.width} wide")
        self._lists: Dict[tuple, torch.Tensor] = {}
        lo, hi = (float(v) for v in action_range)
        self.mid, self.half = (lo + hi) / 2.0, (hi - lo) / 2.0
        self.num_agents = int(num_agents)
        self.device = device
        self.scale_fn: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None
        #: each phase's last registered scale ``(rows, A)`` (None before its first draw), for logging
        self.last_scale: Dict[str, Optional[torch.Tensor]] = {phase: None for phase in PHASES}
        self.held = {phase: HeldDraw(num_agents, cfg.repeat_zeta_s, cfg.repeat_max, device) if cfg.repeat else None
                     for phase, cfg in self.cfgs.items()}
        #: target smoothing, set per update (:meth:`load_smoothing`): on (0/1), std, clip
        self.smoothing_used = any(cfg.target_smoothing for cfg in self.cfgs.values())
        #: a phase smooths with the registered scale (``target_scale: registered``): every critic target then reads it,
        #: selected per phase by the device flag ``registered`` (one captured graph serves both phases)
        self.smoothing_registered = any(cfg.target_smoothing and cfg.target_scale == REGISTERED for cfg in self.cfgs.values())
        self._smooth = {"on": torch.zeros((), device=device), "registered": torch.zeros((), device=device)}
        # per action dim (a per-dim target_scale / target_clip), filled before each update
        self._smooth.update({k: torch.zeros((self.width,), device=device) for k in ("std", "clip")})
        self._smooth["clip"].fill_(1.0)

    # ------------------------------------------------------------------ scales
    @staticmethod
    def _floats(value: Any) -> List[float]:
        return [float(value)] if isinstance(value, (int, float)) else [float(v) for v in value]

    def registered(self, phase: str) -> bool:
        """Phase ``phase``'s scale is a project's per-row function (``scale: registered``)."""
        return self.cfgs[phase].scale == REGISTERED

    def scales(self, phase: str, timestep: int) -> List[float]:
        """Phase ``phase``'s std / half width at env step ``timestep``, one number (every dim) or one per dim (``scale``
        null: the range's half width). A registered scale has no number: refused."""
        cfg = self.cfgs[phase]
        if cfg.scale == REGISTERED:
            raise ValueError(f"ddpg.{phase}_noise.scale is registered: a per-row function of the observation, no number")
        if cfg.scale is None:
            return [self.half]
        start = self._floats(cfg.scale)
        if cfg.scale_end is None or int(cfg.scale_steps) <= 0:
            return start
        end = self._floats(cfg.scale_end)
        n = max(len(start), len(end))
        start, end = (start * n if len(start) == 1 else start), (end * n if len(end) == 1 else end)
        frac = min(max(timestep, 0) / int(cfg.scale_steps), 1.0)
        return [a + (b - a) * frac for a, b in zip(start, end)]

    def scale(self, phase: str, timestep: int) -> float:
        """Phase ``phase``'s std / half width at env step ``timestep``, averaged over the dims (what is logged)."""
        values = self.scales(phase, timestep)
        return sum(values) / len(values)

    def _tensor(self, values: List[float]) -> torch.Tensor:
        """``values`` as a device tensor: a scalar, or ``(width,)`` per dim -- made once per list (a tensor from a Python
        list is a host-to-device copy, a synchronization every step)."""
        key = tuple(values)
        found = self._lists.get(key)
        if found is None:
            found = torch.tensor(values[0] if len(values) == 1 else values, dtype=torch.float32, device=self.device)
            self._lists[key] = found
        return found

    def registered_scale(self, observations: torch.Tensor, phase: str, rows: int, width: int) -> torch.Tensor:
        """Phase ``phase``'s registered scale for every row, ``(rows, width)`` in r units (:attr:`scale_fn`)."""
        if self.scale_fn is None:
            raise RuntimeError(
                f"ddpg.{phase}_noise.scale is registered: a per-row, per-dim function of the observation, but no builder "
                "answered (register_noise_scale in the training script's setup())"
            )
        scale = self.scale_fn(observations, phase)
        if tuple(scale.shape) != (rows, width):
            raise ValueError(f"the registered noise scale returned shape {tuple(scale.shape)}; the draw is {(rows, width)}")
        return scale

    # ------------------------------------------------------------------ acting
    def draw(self, phase: str, mean: Optional[torch.Tensor], observations: torch.Tensor, rows: int, width: int,
             timestep: int) -> torch.Tensor:
        """Phase ``phase``'s action for every row ``(rows, width)``, unclamped: its center plus its noise. ``mean`` the
        actor's action (None when the phase is centered on the midpoint, which never reads it)."""
        cfg = self.cfgs[phase]
        if cfg.distribution == "gaussian":
            unit = torch.randn((rows, width), device=self.device)
        else:
            unit = torch.rand((rows, width), device=self.device) * 2.0 - 1.0
        if self.held[phase] is not None:
            unit = self.held[phase](unit)
        if cfg.scale == REGISTERED:
            scale = self.registered_scale(observations, phase, rows, width)
            self.last_scale[phase] = scale
        else:
            scale = self._tensor(self.scales(phase, timestep))
        noise = unit * scale
        if cfg.clip is not None:
            clip = self._tensor(self._floats(cfg.clip))
            noise = torch.maximum(torch.minimum(noise, clip), -clip)
        if cfg.center == "mid":
            return self.mid + noise
        if mean is None:
            raise ValueError(f"ddpg.{phase}_noise.center: actor needs the actor's action")
        return mean + noise

    def needs_actor(self, phase: str) -> bool:
        return self.cfgs[phase].center == "actor"

    # ------------------------------------------------------------------ target smoothing
    def load_smoothing(self, phase: str, timestep: int) -> None:
        """Set the device buffers the next critic updates' target smoothing reads to phase ``phase``'s (device fills, no
        host read; a captured graph sees them)."""
        cfg = self.cfgs[phase]
        registered = cfg.target_scale == REGISTERED
        if registered:  # per row, from the next observations (smooth)
            std = [0.0]
        elif cfg.target_scale is not None:
            std = self._floats(cfg.target_scale)
        elif cfg.target_smoothing:
            std = self.scales(phase, timestep)
        else:  # no smoothing in this phase: the std is never used
            std = [0.0]
        self._smooth["on"].fill_(1.0 if cfg.target_smoothing else 0.0)
        self._smooth["registered"].fill_(1.0 if registered else 0.0)
        self._smooth["std"].copy_(self._tensor(std).expand(self.width))
        self._smooth["clip"].copy_(self._tensor(self._floats(cfg.target_clip)).expand(self.width))

    def smooth(self, next_actions: torch.Tensor, next_observations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The target's next action plus the loaded phase's smoothing noise (unclamped); as it is when no phase smooths.
        ``next_observations``: the raw next observations, read when a phase smooths with the registered scale."""
        if not self.smoothing_used:
            return next_actions
        clip = self._smooth["clip"]
        std = self._smooth["std"]
        if self.smoothing_registered:
            if next_observations is None:
                raise RuntimeError("target_scale: registered reads the next observations' registered scale, and none were given")
            per_row = self.registered_scale(next_observations, "train", next_actions.shape[0], next_actions.shape[1])
            std = torch.where(self._smooth["registered"] > 0, per_row, std)
        noise = torch.maximum(torch.minimum(torch.randn_like(next_actions) * std, clip), -clip)
        return next_actions + noise * self._smooth["on"]

    def smoothing_std(self) -> torch.Tensor:
        """The loaded phase's smoothing std, a device scalar (for logging; the mean over the dims)."""
        return self._smooth["std"].mean()

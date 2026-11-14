"""Every action-noise draw of a deterministic actor (DDPG), in one place.

Two phases, one set of options each (:class:`~robonuke_rl_core.learners.cfg.NoiseCfg`): ``warmup`` (until the learner
starts learning, and the target smoothing of its critic warm-up burst) and ``train`` (from then on, and every later
critic update's target smoothing). Eval acts without noise. One draw, for either phase::

    d     = N(0, 1) | U(-1, 1)                     distribution (held for k steps per agent with ``repeat``)
    noise = clip(d * scale(t), +-clip)             scale: std / half width, linear schedule; clip on the scaled draw
    noise = noise * factor(obs)                    adaptive: the env's per-env, per-dim factor, from the observation
    a     = actor(s) + noise  |  mid + noise       center

Target smoothing (TD3's) in a critic update: ``a' = mu'(s') + clip(N(0, target_scale), +-target_clip)``, times
``factor(s')`` with ``target_adaptive``. Its numbers live in device buffers set before each update (:meth:`ActionNoise.
load_smoothing`), so one captured CUDA graph serves both phases.

:class:`HeldDraw` is the correlated-noise hold (FlashSAC's), shared with SAC's ``noise_repeat``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

__all__ = ["HeldDraw", "ActionNoise", "PHASES"]

PHASES = ("warmup", "train")


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
    deterministic actor over ``action_range`` ``(lo, hi)``. ``factor`` is read when needed: ``obs -> (rows, A)``, the
    adaptive factor from raw observations (the trainer sets it from ``register_exploration_scale``), or None."""

    def __init__(self, warmup: Any, train: Any, *, action_range, num_agents: int, device: Any) -> None:
        self.cfgs = {"warmup": warmup, "train": train}
        lo, hi = (float(v) for v in action_range)
        self.mid, self.half = (lo + hi) / 2.0, (hi - lo) / 2.0
        self.num_agents = int(num_agents)
        self.device = device
        self.factor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.held = {phase: HeldDraw(num_agents, cfg.repeat_zeta_s, cfg.repeat_max, device) if cfg.repeat else None
                     for phase, cfg in self.cfgs.items()}
        #: target smoothing, set per update (:meth:`load_smoothing`): on (0/1), std, clip, adaptive (0/1)
        self.smoothing_used = any(cfg.target_smoothing for cfg in self.cfgs.values())
        self.smoothing_adaptive = any(cfg.target_smoothing and cfg.target_adaptive for cfg in self.cfgs.values())
        self._smooth = {k: torch.zeros((), device=device) for k in ("on", "std", "clip", "adaptive")}
        self._smooth["clip"].fill_(1.0)

    # ------------------------------------------------------------------ scales
    def scale(self, phase: str, timestep: int) -> float:
        """Phase ``phase``'s std / half width at env step ``timestep`` (``scale`` null: the range's half width)."""
        cfg = self.cfgs[phase]
        if cfg.scale is None:
            return self.half
        start, end, steps = float(cfg.scale), cfg.scale_end, int(cfg.scale_steps)
        if end is None or steps <= 0:
            return start
        return start + (float(end) - start) * min(max(timestep, 0) / steps, 1.0)

    def _factor(self, observations: torch.Tensor, what: str) -> torch.Tensor:
        if self.factor is None:
            raise RuntimeError(
                f"{what} is adaptive: it scales by a per-env, per-dim factor from the observation, but no builder "
                "answered (register_exploration_scale in the training script's setup())"
            )
        return self.factor(observations)

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
        noise = unit * self.scale(phase, timestep)
        if cfg.clip is not None:
            noise = noise.clamp(-float(cfg.clip), float(cfg.clip))
        if cfg.adaptive:
            noise = noise * self._factor(observations, f"ddpg.{phase}_noise.adaptive")
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
        std = float(cfg.target_scale) if cfg.target_scale is not None else self.scale(phase, timestep)
        self._smooth["on"].fill_(1.0 if cfg.target_smoothing else 0.0)
        self._smooth["std"].fill_(std)
        self._smooth["clip"].fill_(float(cfg.target_clip))
        self._smooth["adaptive"].fill_(1.0 if cfg.target_smoothing and cfg.target_adaptive else 0.0)

    def smooth(self, next_actions: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        """The target's next action plus the loaded phase's smoothing noise (unclamped); as it is when no phase smooths."""
        if not self.smoothing_used:
            return next_actions
        clip = self._smooth["clip"]
        noise = torch.maximum(torch.minimum(torch.randn_like(next_actions) * self._smooth["std"], clip), -clip)
        if self.smoothing_adaptive:
            factor = self._factor(next_observations, "the target smoothing (target_adaptive)")
            noise = noise * (1.0 + self._smooth["adaptive"] * (factor - 1.0))
        return next_actions + noise * self._smooth["on"]

    def smoothing_std(self) -> torch.Tensor:
        """The loaded phase's smoothing std, a device scalar (for logging)."""
        return self._smooth["std"]

"""Per-phase logging for routed residuals (``sac.phase_logging``): every phase its own episode tracking, step counter
and publish cadence.

A *phase attempt* is an env's stay in one ``sac.std_label_names`` phase: it starts when the label the env publishes
(``sac.std_by_label``) enters the phase and ends when the label changes or the episode ends. It succeeded if the label
moved to a later phase, or if the episode ended in it with the task's success (``episode/success``: only the last
phase can be in progress when the task succeeds). Each attempt records its steps, its return (the rewards of its steps)
and whether its episode ended out of bounds (``episode/out_of_bounds``, when the env publishes it).

Every phase in ``sac.phase_logging_phases`` (default: every label) counts its env steps, ``<phase>_step``, per agent.

* A phase with a residual (``sac.route_labels``) is published every ``trainer.write_interval`` of **its own** steps, as
  ``<phase>_episode/<metric>`` (``success`` -- over the last ``trainer.success_window`` attempts when set --,
  ``best_success_rate``, ``length``, ``length_min``, ``length_max``, ``return``, ``return_min``, ``return_max``,
  ``count``, ``out_of_bounds``), its residual's learner metrics as ``<phase>_<family>/<name>`` (``insert_q/target_mean``,
  ``insert_loss/critic``, ...), the env's per-phase residual metrics (``sac.metrics_by_label``) as
  ``<phase>_residual/<name>``, and ``<phase>_step`` with every point (each run plots ``<phase>_*`` against it).
* A phase without one is published with the episodes, every ``trainer.write_interval`` env steps: ``episode/<phase>_success``,
  ``episode/<phase>_length`` (steps per attempt), ``episode/<phase>_steps`` (its step count).

The env's ``phase/*`` episode metrics leave the ``phase`` family: ``phase/reached`` becomes ``episode/phase_reached``,
``phase/end``, ``phase/passed_*`` and ``phase/<name>_step`` are dropped (the attempts above replace them), any other
``phase/<name>`` becomes ``episode/<name>``. Every metric name has one ``/``: the rest of it is joined with ``_``
(``episode/length/max`` -> ``episode/length_max``).

The labels come to the host once per step (one ``(num_envs,)`` copy), so this is for the few-env runs routing is used
with.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

__all__ = ["PhaseLogger", "flat_name"]

#: the env's episode metrics this replaces (dropped): the per-phase pass flags and step indices, and the end code
_DROPPED_PHASE = ("phase/end",)


def flat_name(name: str) -> str:
    """``family/a/b`` -> ``family/a_b``: one ``/`` (wandb groups by the first; a second one does not nest)."""
    family, sep, rest = name.partition("/")
    return f"{family}/{rest.replace('/', '_')}" if sep else name


class _Attempts:
    """One (agent, phase)'s finished attempts since its last publish, its success window and best rate."""

    def __init__(self, window: Optional[int]) -> None:
        self.success: List[float] = []
        self.length: List[float] = []
        self.ret: List[float] = []
        self.oob: List[float] = []
        self.window = deque(maxlen=int(window)) if window else None
        self.best = float("nan")

    def add(self, success: float, length: float, ret: float, oob: float) -> None:
        self.success.append(success)
        self.length.append(length)
        self.ret.append(ret)
        self.oob.append(oob)
        if self.window is not None:
            self.window.append(success)

    def rate(self) -> Optional[float]:
        values = list(self.window) if self.window is not None else self.success
        return sum(values) / len(values) if values else None

    def clear(self) -> None:
        self.success, self.length, self.ret, self.oob = [], [], [], []


class PhaseLogger:
    """The per-phase layer between a :class:`~robonuke_rl_core.learners.routed.RoutedSAC` and its
    :class:`~robonuke_rl_core.logging.WandbLogger` (module docstring)."""

    def __init__(
        self,
        logger: Any,
        *,
        names: Sequence[str],
        phases: Sequence[str],
        routes: Sequence[str],
        num_agents: int,
        envs_per_agent: int,
        write_interval: int,
        success_window: Optional[int] = None,
        distributions: Sequence[str] = (),
    ) -> None:
        from robonuke_rl_core.logging import MetricAccumulator

        self.logger = logger
        self.names = list(names)
        self.phases = list(phases)
        self.routes = list(routes)
        unknown = [p for p in self.phases + self.routes if p not in self.names]
        if unknown:
            raise ValueError(f"phase logging: {unknown} are not std_label_names {self.names}")
        missing = [r for r in self.routes if r not in self.phases]
        if missing:
            raise ValueError(f"phase logging: routed phases {missing} are not in phase_logging_phases {self.phases}")
        if int(write_interval) <= 0:
            raise ValueError("phase logging publishes every trainer.write_interval phase steps: set it > 0")
        self.num_agents = int(num_agents)
        self.envs_per_agent = int(envs_per_agent)
        self.write_interval = int(write_interval)
        self.index = {p: self.names.index(p) for p in self.phases}
        #: per (agent, phase) env steps, and the finished attempts since the phase's last publish
        self.steps = {(a, p): 0 for a in range(self.num_agents) for p in self.phases}
        self.attempts = {(a, p): _Attempts(success_window) for a in range(self.num_agents) for p in self.phases}
        #: a routed phase's learner and per-phase residual metrics, held until its own publish
        self.held = {
            p: MetricAccumulator(self.num_agents, device=getattr(getattr(logger, "accumulator", None), "device", None),
                                 distributions=[self._route_name(p, d) for d in distributions], sep="_")
            for p in self.routes
        }
        #: per env: its current attempt's steps and return (reset when the attempt ends)
        self._attempt_steps: Optional[torch.Tensor] = None
        self._attempt_return: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ names
    @staticmethod
    def _route_name(route: str, name: str) -> str:
        return f"{route}_{flat_name(name)}"

    def _split_label(self, name: str) -> Optional[tuple]:
        """``<base>_<phase>`` (``sac.metrics_by_label`` / ``std_by_label``'s per-phase names) -> (base, phase)."""
        for phase in self.names:
            if name.endswith(f"_{phase}"):
                return name[: -len(phase) - 1], phase
        return None

    def _env_name(self, name: str) -> Optional[str]:
        """An env / episode metric's published name, or None to drop it."""
        if name in _DROPPED_PHASE or name.startswith("phase/passed_"):
            return None
        if name.startswith("phase/"):
            rest = name[len("phase/"):]
            if rest.endswith("_step") and rest[: -len("_step")] in self.names:
                return None  # the step a phase was done at: the attempts' lengths replace it
            return "episode/phase_reached" if rest == "reached" else flat_name(f"episode/{rest}")
        return flat_name(name)

    # ------------------------------------------------------------------ the routed residuals' on_log hooks
    def hook_for(self, route: str, first: bool) -> Callable[[int, Dict[str, torch.Tensor], int], None]:
        """``route``'s residual's ``on_log`` hook: its learner metrics held for its phase's publish, except its progress
        (``route/*``: its phase's steps, replay rows and updates), published every env interval as ``route/{route}_*``
        (outside ``{route}_*``, so on the env-step axis) and so every phase's count is current; the env's metrics (from the first residual only)
        renamed into ``episode/`` and published with the episodes, the per-phase residual metrics held for their phase
        (dropped for a phase without a residual)."""
        from robonuke_rl_core.learners.routed import ENV_METRIC_PREFIXES

        def hook(agent: int, metrics: Dict[str, torch.Tensor], step: int) -> None:
            now: Dict[str, torch.Tensor] = {}
            held: Dict[str, Dict[str, torch.Tensor]] = {}
            for name, value in metrics.items():
                if name.startswith(ENV_METRIC_PREFIXES):
                    if not first:
                        continue
                    if name.startswith("residual/"):
                        split = self._split_label(name)
                        if split is not None and split[1] in self.held:
                            base, phase = split
                            held.setdefault(phase, {})[f"{phase}_{flat_name(base)}"] = value
                        continue  # the unsplit residual metrics, and a phase without a residual's
                    renamed = self._env_name(name)
                    if renamed is not None:
                        now[renamed] = value
                    continue
                if name.startswith("route/"):
                    # route/<route>_<metric>: outside <route>_*, so it plots against the env steps, not <route>_step
                    now[f"route/{route}_{flat_name(name.split('/', 1)[1])}"] = value
                    continue
                split = self._split_label(name) if name.startswith("policy/") else None
                if split is not None:
                    # std_by_label's per-phase std: only the residual's own phase
                    if split[1] == route:
                        held.setdefault(route, {})[self._route_name(route, split[0])] = value
                    continue
                held.setdefault(route, {})[self._route_name(route, name)] = value
            if now:
                self.logger(agent, now, step)
            for phase, values in held.items():
                self.held[phase].add(agent, values)

        return hook

    # ------------------------------------------------------------------ the env steps
    def observe(
        self,
        labels: Optional[torch.Tensor],
        next_labels: Optional[torch.Tensor],
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        step: int,
    ) -> None:
        """One env step: ``labels`` the phase each env's step started in (the routed learner's), ``next_labels`` the
        one it ended in (this step's ``infos``)."""
        if labels is None or next_labels is None:
            return
        n = labels.numel()
        if self._attempt_steps is None:
            self._attempt_steps = torch.zeros(n, dtype=torch.float64)
            self._attempt_return = torch.zeros(n, dtype=torch.float64)
        now = labels.reshape(-1).to("cpu", torch.long)
        nxt = next_labels.reshape(-1).to("cpu", torch.long)
        done = (terminated.reshape(-1) | truncated.reshape(-1)).to("cpu", torch.bool)
        self._attempt_steps += 1
        self._attempt_return += rewards.reshape(-1).to("cpu", torch.float64)
        episode = (infos or {}).get("episode_metrics_to_log") or {} if isinstance(infos, dict) else {}
        success = episode.get("episode/success")
        oob = episode.get("episode/out_of_bounds")
        ended = done | (nxt != now)
        crossed = []
        for env in range(n):
            label = int(now[env])
            phase = self.names[label] if 0 <= label < len(self.names) else None
            if phase not in self.index:
                if bool(ended[env]):
                    self._attempt_steps[env] = 0.0
                    self._attempt_return[env] = 0.0
                continue
            agent = env // self.envs_per_agent
            before = self.steps[(agent, phase)]
            self.steps[(agent, phase)] = before + 1
            if (before + 1) % self.write_interval == 0:
                crossed.append((agent, phase))
            if bool(ended[env]):
                if bool(done[env]):
                    passed = float(success.reshape(-1)[env]) if success is not None else 0.0
                    out = float(oob.reshape(-1)[env]) if oob is not None else float("nan")
                else:
                    passed, out = float(int(nxt[env]) > label), 0.0
                self.attempts[(agent, phase)].add(passed, float(self._attempt_steps[env]), float(self._attempt_return[env]), out)
                self._attempt_steps[env] = 0.0
                self._attempt_return[env] = 0.0
        for agent, phase in crossed:
            if phase in self.held:
                self._publish_phase(agent, phase, step)

    def _publish_phase(self, agent: int, phase: str, step: int) -> None:
        """A routed phase's point: its attempts, its held metrics and its step count, at the global ``step``."""
        values = dict(self.held[phase].flush(agents=[agent])[agent])
        values.update(self._attempt_values(agent, phase, f"{phase}_episode/"))
        values[f"{phase}_step"] = float(self.steps[(agent, phase)])
        self.logger.log_now(agent, values, step)

    def _attempt_values(self, agent: int, phase: str, prefix: str) -> Dict[str, float]:
        a = self.attempts[(agent, phase)]
        out: Dict[str, float] = {}
        rate = a.rate()
        if rate is not None:
            out[f"{prefix}success"] = rate
            a.best = rate if not a.best == a.best else max(a.best, rate)  # NaN before the first
            out[f"{prefix}best_success_rate"] = a.best
        if a.length:
            out[f"{prefix}count"] = float(len(a.length))
            for key, values in (("length", a.length), ("return", a.ret)):
                out[f"{prefix}{key}"] = sum(values) / len(values)
                out[f"{prefix}{key}_min"] = min(values)
                out[f"{prefix}{key}_max"] = max(values)
            oob = [v for v in a.oob if v == v]
            if oob:
                out[f"{prefix}out_of_bounds"] = sum(oob) / len(oob)
        a.clear()
        return out

    # ------------------------------------------------------------------ the episodes' interval
    def on_global_flush(self, step: int) -> None:
        """Every ``trainer.write_interval`` env steps, before the logger's flush: each phase without a residual's
        attempts and step count into the ``episode`` family."""
        for agent in range(self.num_agents):
            values: Dict[str, torch.Tensor] = {}
            for phase in self.phases:
                if phase in self.held:
                    continue
                a = self.attempts[(agent, phase)]
                if a.success:
                    values[f"episode/{phase}_success"] = torch.tensor(sum(a.success) / len(a.success))
                    values[f"episode/{phase}_length"] = torch.tensor(sum(a.length) / len(a.length))
                values[f"episode/{phase}_steps"] = torch.tensor(float(self.steps[(agent, phase)]))
                a.clear()
            if values:
                self.logger(agent, values, step)

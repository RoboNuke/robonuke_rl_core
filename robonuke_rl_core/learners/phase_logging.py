"""Per-phase logging for routed residuals (``sac.phase_logging``): every phase its own episode tracking, step counter
and publish cadence.

A *phase attempt* is an env's stay in one ``sac.std_label_names`` phase: it starts when the label the env publishes
(``sac.std_by_label``) enters the phase and ends when the label changes or the episode ends. It succeeded if the label
moved to a later phase, or if the episode ended in it with the task's success (``episode/success``: only the last
phase can be in progress when the task succeeds). Each attempt records its steps, its return (the rewards of its steps)
and whether its episode ended out of bounds (``episode/out_of_bounds``, when the env publishes it).

Every phase in ``sac.phase_logging_phases`` (default: every label) counts its env steps, ``<phase>_step``, per agent:
the steps the agent's envs started in the phase, which is what its residual's memory receives rows for. While a
residual pools its rows into every agent's memory (DDPG ``shared_warmup``), every agent's counter takes every agent's
envs' steps in that phase, so ``<phase>_step`` stays the count of phase steps behind the agent's memory (and equals the
residual's ``route/steps``). Counted on the labels at step time: ``n_step`` delaying the rows and the memory's capacity
play no part.

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

Nothing is read to the host per step: the attempt counters, the per-(agent, phase) step counts and attempt sums,
the success windows and the publish-cadence flags all live on the labels' device, and a phase's point is read once
when it is published (every ``write_interval`` of its steps) or at the global flush. Whether a phase crossed its
publish cadence this step is a device tensor; :meth:`PhaseLogger.observe` reads it itself, or leaves it for a
caller that reads every learner's host values in one copy (``defer=True`` and :meth:`PhaseLogger.publish_crossed`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from .windows import push_window

__all__ = ["PhaseLogger", "flat_name"]

#: the env's episode metrics this replaces (dropped): the per-phase pass flags and step indices, and the end code
_DROPPED_PHASE = ("phase/end",)


def flat_name(name: str) -> str:
    """``family/a/b`` -> ``family/a_b``: one ``/`` (wandb groups by the first; a second one does not nest)."""
    family, sep, rest = name.partition("/")
    return f"{family}/{rest.replace('/', '_')}" if sep else name


class _Attempts:
    """Per (agent, phase), the finished attempts since the phase's last publish, its success window and best rate: every
    sum a ``(num_agents, num_phases)`` float64 device tensor, the window ``(num_agents, num_phases, window)``."""

    def __init__(self, num_agents: int, num_phases: int, window: Optional[int], device) -> None:
        shape = (num_agents, num_phases)
        z = lambda: torch.zeros(shape, dtype=torch.float64, device=device)  # noqa: E731
        self.count, self.success, self.length, self.ret, self.oob, self.oob_count = z(), z(), z(), z(), z(), z()
        self.length_min, self.length_max = torch.full(shape, float("inf"), dtype=torch.float64, device=device), torch.full(shape, float("-inf"), dtype=torch.float64, device=device)
        self.return_min, self.return_max = torch.full(shape, float("inf"), dtype=torch.float64, device=device), torch.full(shape, float("-inf"), dtype=torch.float64, device=device)
        self.window = torch.full((num_agents * num_phases, int(window)), float("nan"), dtype=torch.float64, device=device) if window else None
        self.window_next = torch.zeros((num_agents * num_phases,), dtype=torch.long, device=device) if window else None
        self.best = [[float("nan")] * num_phases for _ in range(num_agents)]
        self.num_phases = num_phases

    def add(self, agent: torch.Tensor, phase: torch.Tensor, finished: torch.Tensor, passed: torch.Tensor,
            length: torch.Tensor, ret: torch.Tensor, oob: torch.Tensor) -> None:
        """Fold this step's finished attempts (per env: ``finished`` bool, the others float64) into the sums, on the
        device. Envs that finished nothing add zero; the min / max see +-inf for them."""
        index = (agent, phase)
        w = finished.to(torch.float64)
        self.count.index_put_(index, w, accumulate=True)
        self.success.index_put_(index, w * passed, accumulate=True)
        self.length.index_put_(index, w * length, accumulate=True)
        self.ret.index_put_(index, w * ret, accumulate=True)
        finite = finished & torch.isfinite(oob)
        self.oob.index_put_(index, finite.to(torch.float64) * torch.nan_to_num(oob), accumulate=True)
        self.oob_count.index_put_(index, finite.to(torch.float64), accumulate=True)
        flat = agent * self.num_phases + phase
        inf = torch.full_like(length, float("inf"))
        for target, values, reduce in ((self.length_min, length, "amin"), (self.length_max, length, "amax"),
                                       (self.return_min, ret, "amin"), (self.return_max, ret, "amax")):
            masked = torch.where(finished, values, inf if reduce == "amin" else -inf)
            target.view(-1).index_reduce_(0, flat, masked, reduce, include_self=True)
        if self.window is not None:
            push_window(self.window, self.window_next, flat, finished, passed)

    def read(self, agent: int, phase: int) -> Dict[str, float]:
        """One (agent, phase)'s values as floats: a single device read."""
        cells = [self.count, self.success, self.length, self.ret, self.oob, self.oob_count,
                 self.length_min, self.length_max, self.return_min, self.return_max]
        values = torch.stack([c[agent, phase] for c in cells])
        if self.window is not None:
            values = torch.cat((values, self.window[agent * self.num_phases + phase]))
        values = values.tolist()
        keys = ["count", "success", "length", "ret", "oob", "oob_count", "length_min", "length_max", "return_min", "return_max"]
        out = dict(zip(keys, values))
        out["window"] = values[len(keys):]
        return out

    def rate(self, agent: int, phase: int, read: Dict[str, float]) -> Optional[float]:
        values = [v for v in read["window"] if v == v] if self.window is not None else None
        if values is None:
            return read["success"] / read["count"] if read["count"] > 0 else None
        return sum(values) / len(values) if values else None

    def clear(self, agent: int, phase: int) -> None:
        for cell in (self.count, self.success, self.length, self.ret, self.oob, self.oob_count):
            cell[agent, phase] = 0.0
        self.length_min[agent, phase] = float("inf")
        self.return_min[agent, phase] = float("inf")
        self.length_max[agent, phase] = float("-inf")
        self.return_max[agent, phase] = float("-inf")


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
        self.success_window = int(success_window) if success_window else None
        #: per (agent, phase) env steps, and the finished attempts since the phase's last publish: on the labels'
        #: device, created at the first step (:meth:`_state`)
        self.steps: Optional[torch.Tensor] = None
        self.attempts: Optional[_Attempts] = None
        #: label index -> position in ``phases`` (-1: not tracked), and env -> agent, on the device
        self._phase_of_label: Optional[torch.Tensor] = None
        self._env_agent: Optional[torch.Tensor] = None
        #: a routed phase's learner and per-phase residual metrics, held until its own publish
        self.held = {
            p: MetricAccumulator(self.num_agents, device=getattr(getattr(logger, "accumulator", None), "device", None),
                                 distributions=[self._route_name(p, d) for d in distributions], sep="_")
            for p in self.routes
        }
        #: per env: its current attempt's steps and return (reset when the attempt ends)
        self._attempt_steps: Optional[torch.Tensor] = None
        self._attempt_return: Optional[torch.Tensor] = None

    def _state(self, labels: torch.Tensor) -> None:
        """The device state, created at the first step on the labels' device."""
        if self.steps is not None:
            return
        n, device = labels.numel(), labels.device
        if n != self.num_agents * self.envs_per_agent:
            raise ValueError(f"phase logging: {n} labels for {self.num_agents} x {self.envs_per_agent} envs")
        self.steps = torch.zeros((self.num_agents, len(self.phases)), dtype=torch.float64, device=device)
        self.attempts = _Attempts(self.num_agents, len(self.phases), self.success_window, device)
        lookup = torch.full((len(self.names),), -1, dtype=torch.long, device=device)
        for position, phase in enumerate(self.phases):
            lookup[self.index[phase]] = position
        self._phase_of_label = lookup
        self._env_agent = torch.arange(n, device=device) // self.envs_per_agent
        self._attempt_steps = torch.zeros(n, dtype=torch.float64, device=device)
        self._attempt_return = torch.zeros(n, dtype=torch.float64, device=device)

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
        defer: bool = False,
        pooled: Sequence[str] = (),
    ) -> Optional[torch.Tensor]:
        """One env step: ``labels`` the phase each env's step started in (the routed learner's), ``next_labels`` the
        one it ended in (this step's ``infos``). ``pooled``: the routed phases whose residual wrote this step's rows
        into every agent's memory; their step counters grow by every env's step in the phase, for every agent. Device
        work only, except the publish check: which (agent, phase) crossed its publish cadence this step, a
        ``(num_agents, num_phases)`` bool tensor, read here (one host synchronization) and acted on, or, with
        ``defer``, returned for the caller to read with its other host values and hand to :meth:`publish_crossed`."""
        if labels is None or next_labels is None:
            return None
        self._state(labels)
        now = labels.reshape(-1).to(torch.long)
        nxt = next_labels.reshape(-1).to(now.device, torch.long)
        done = (terminated.reshape(-1) | truncated.reshape(-1)).to(now.device)
        self._attempt_steps += 1
        self._attempt_return += rewards.reshape(-1).to(now.device, torch.float64)
        episode = (infos or {}).get("episode_metrics_to_log") or {} if isinstance(infos, dict) else {}
        success = episode.get("episode/success")
        oob = episode.get("episode/out_of_bounds")
        known = (now >= 0) & (now < len(self.names))
        phase = self._phase_of_label[now.clamp(0, len(self.names) - 1)]
        tracked = known & (phase >= 0)
        phase = phase.clamp(min=0)  # an untracked env's weight is 0 below, its position irrelevant
        ended = done | (nxt != now)
        agent = self._env_agent
        before = self.steps.clone()
        own = tracked
        for name in pooled:
            if name not in self.held:
                raise ValueError(f"phase logging: pooled phase {name!r} is not a routed phase {list(self.held)}")
            position = self.phases.index(name)
            in_phase = tracked & (phase == position)
            own = own & in_phase.logical_not()
            self.steps[:, position] += in_phase.sum().to(torch.float64)  # every env's step, to every agent
        self.steps.index_put_((agent, phase), own.to(torch.float64), accumulate=True)
        crossed = (self.steps // self.write_interval) > (before // self.write_interval)
        finished = tracked & ended
        zero = torch.zeros_like(self._attempt_return)
        passed_done = success.reshape(-1).to(now.device, torch.float64) if success is not None else zero
        out_done = oob.reshape(-1).to(now.device, torch.float64) if oob is not None else torch.full_like(zero, float("nan"))
        passed = torch.where(done, passed_done, (nxt > now).to(torch.float64))
        out = torch.where(done, out_done, zero)
        self.attempts.add(agent, phase, finished, passed, self._attempt_steps, self._attempt_return, out)
        self._attempt_steps = torch.where(ended, zero, self._attempt_steps)
        self._attempt_return = torch.where(ended, zero, self._attempt_return)
        if defer:
            return crossed
        self.publish_crossed(crossed.reshape(-1).tolist(), step)
        return None

    def publish_crossed(self, crossed: Sequence[float], step: int) -> None:
        """Publish every routed (agent, phase) whose flag in ``crossed`` (``num_agents * num_phases`` host values, row
        major, as :meth:`observe` returned them) is set."""
        width = len(self.phases)
        for flat, flag in enumerate(crossed):
            if flag:
                agent, position = divmod(flat, width)
                phase = self.phases[position]
                if phase in self.held:
                    self._publish_phase(agent, phase, step)

    def _publish_phase(self, agent: int, phase: str, step: int) -> None:
        """A routed phase's point: its attempts, its held metrics and its step count, at the global ``step``."""
        values = dict(self.held[phase].flush(agents=[agent])[agent])
        values.update(self._attempt_values(agent, phase, f"{phase}_episode/"))
        values[f"{phase}_step"] = float(self.steps[agent, self.phases.index(phase)].item())
        self.logger.log_now(agent, values, step)

    def _attempt_values(self, agent: int, phase: str, prefix: str) -> Dict[str, float]:
        position = self.phases.index(phase)
        a = self.attempts
        read = a.read(agent, position)
        out: Dict[str, float] = {}
        rate = a.rate(agent, position, read)
        if rate is not None:
            out[f"{prefix}success"] = rate
            best = a.best[agent][position]
            a.best[agent][position] = rate if not best == best else max(best, rate)  # NaN before the first
            out[f"{prefix}best_success_rate"] = a.best[agent][position]
        count = read["count"]
        if count > 0:
            out[f"{prefix}count"] = float(count)
            out[f"{prefix}length"] = read["length"] / count
            out[f"{prefix}length_min"] = read["length_min"]
            out[f"{prefix}length_max"] = read["length_max"]
            out[f"{prefix}return"] = read["ret"] / count
            out[f"{prefix}return_min"] = read["return_min"]
            out[f"{prefix}return_max"] = read["return_max"]
            if read["oob_count"] > 0:
                out[f"{prefix}out_of_bounds"] = read["oob"] / read["oob_count"]
        a.clear(agent, position)
        return out

    # ------------------------------------------------------------------ the episodes' interval
    def on_global_flush(self, step: int) -> None:
        """Every ``trainer.write_interval`` env steps, before the logger's flush: each phase without a residual's
        attempts and step count into the ``episode`` family."""
        if self.steps is None:
            return
        steps = self.steps.tolist()  # one read for every agent and phase
        for agent in range(self.num_agents):
            values: Dict[str, torch.Tensor] = {}
            for position, phase in enumerate(self.phases):
                if phase in self.held:
                    continue
                read = self.attempts.read(agent, position)
                if read["count"] > 0:
                    values[f"episode/{phase}_success"] = torch.tensor(read["success"] / read["count"])
                    values[f"episode/{phase}_length"] = torch.tensor(read["length"] / read["count"])
                values[f"episode/{phase}_steps"] = torch.tensor(float(steps[agent][position]))
                self.attempts.clear(agent, position)
            if values:
                self.logger(agent, values, step)

"""The logging layer: accumulate metrics on the GPU, publish one wandb run per agent.

Two pieces, so the arithmetic is testable without wandb installed:

* :class:`MetricAccumulator` — pure torch. Sums and counts per (agent, name) live on the
  device between flushes, so the rollout and update paths never pay a GPU->CPU sync for a
  metric. :meth:`MetricAccumulator.flush` converts the whole interval in one go.
* :class:`WandbLogger` — owns one wandb run per agent plus an accumulator, and is itself the
  learner's ``on_log`` hook. ``wandb`` is imported lazily, so importing this module (and the
  package) works without it.

The learners stay dumb routers: they slice per agent, mask the episode channel, and forward.
Every aggregation decision lives here. A metric whose interval count is zero publishes
nothing — an interval with no finished episodes leaves a gap in the chart rather than a zero.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

__all__ = ["MetricAccumulator", "WandbLogger", "MIN_WANDB_VERSION", "DISTRIBUTIONS"]

#: first wandb release whose ``reinit="create_new"`` keeps several runs alive in one process
MIN_WANDB_VERSION = (0, 19, 10)

#: allowed values of ``wandb.mode``
MODES = ("online", "offline", "disabled")

#: metrics published with ``/min`` and ``/max`` beside their mean. For these the spread
#: across envs and episodes is the interesting part: an average return hides a run where one
#: env does all the work, and an average episode length hides the split between a quick
#: success and a timeout.
DISTRIBUTIONS = (
    "episode/return",
    "episode/length",
    "reward/step",
    "q/q1_mean",
    "q/q2_mean",
    "q/target_mean",
    # a peak is the whole question with memory: a mean that fits says nothing about the
    # spike that did not
    "stats/gpu_used_mb",
    "stats/ram_mb",
)


class MetricAccumulator:
    """Running per-agent sums and counts, converted to Python only on :meth:`flush`.

    Values arrive as tensors of any shape: ``(envs_per_agent,)`` for a per-step env metric,
    ``(k,)`` for the episodes that ended this step (``k`` may be 0), 0-d for a learner metric.
    Each contributes its finite entries to a running sum and count, so the flushed value is
    the mean over everything the interval saw.

    **A NaN means "not applicable", not "zero".** An episode that did not break has no break
    step; one that never succeeded has no time-to-success. Those entries are NaN and are
    skipped -- by the sum, the count and the min/max alike -- so a metric that only some
    episodes have still reports the right average over the episodes that have it. The count
    therefore lives on the device too, and is converted with everything else at flush.

    **Batched, because the step is launch-bound.** The sums and counts are one ``(num_agents,
    names)`` table each on the device. :meth:`add` only stacks the call's values per shape (one
    copy per shape) and queues them; every ``COMMIT_EVERY`` queued stacks, and at :meth:`flush`,
    the queue is folded into the tables in one pass per value width (one ``isfinite``, one masked
    sum, one count, two ``index_add_``). Folding each metric of each agent as it arrived cost
    ~8 kernel launches per metric and ~1500 per training step -- more than the replay memory
    write it sits next to. No host synchronization at any point before :meth:`flush`: the row
    indices of a fold reach the device through pinned memory.
    """

    #: queued stacks that trigger a fold (bounds the queue when nothing flushes for a long time)
    COMMIT_EVERY = 64

    def __init__(
        self,
        num_agents: int,
        device: Optional[Any] = None,
        distributions: Iterable[str] = (),
        sep: str = "/",
    ) -> None:
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        self.num_agents = int(num_agents)
        self.device = torch.device(device) if device is not None else None
        #: names that also publish ``<name>/min`` and ``<name>/max`` -- for a metric whose
        #: spread across envs says as much as its average (a return, an episode length, a Q)
        self.distributions = set(distributions)
        #: what joins a distribution's name and ``min`` / ``max`` (``"_"``: one ``/`` per name, phase logging's)
        self.sep = str(sep)
        #: name -> column of the tables
        self._columns: Dict[str, int] = {}
        #: ``(num_agents, columns)`` float64 on the device: sums, finite counts, and for the distributions the
        #: running min and max (+-inf where nothing was seen); None until the first value fixes the device
        self._sum: Optional[torch.Tensor] = None
        self._count: Optional[torch.Tensor] = None
        self._low: Optional[torch.Tensor] = None
        self._high: Optional[torch.Tensor] = None
        #: queued (agent, names, stacked ``(len(names), width)`` float64 values), folded by :meth:`_commit`
        self._pending: List[tuple] = []

    # ------------------------------------------------------------------ in
    def add(self, agent: int, metrics: Dict[str, torch.Tensor]) -> None:
        """Fold one agent's metrics into the interval. No host sync."""
        if not 0 <= agent < self.num_agents:
            raise ValueError(f"agent {agent} is out of range for {self.num_agents} agents")
        by_width: Dict[int, tuple] = {}
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"metric '{name}' must be a tensor, got {type(value).__name__}; the "
                    "accumulator sums on the device and converts once per flush"
                )
            if value.numel() == 0:
                continue  # an interval with nothing in it publishes nothing
            if name not in self._columns:
                self._add_column(name)
            names, values = by_width.setdefault(value.numel(), ([], []))
            names.append(name)
            values.append(value.detach().reshape(-1))
        for width, (names, values) in by_width.items():
            stacked = torch.stack(values).to(torch.float64)
            if self.device is None:
                self.device = stacked.device
            if self._sum is None:
                self._tables_to(self.device)
            if stacked.device != self.device:
                # a value from elsewhere (phase logging's host-side counts): every sum lives on one device
                stacked = stacked.to(self.device)
            self._pending.append((agent, names, stacked))
        if len(self._pending) >= self.COMMIT_EVERY:
            self._commit()

    def _add_column(self, name: str) -> None:
        self._commit()  # the queued stacks index the tables as they are
        self._columns[name] = len(self._columns)
        if self._sum is not None:
            self._grow(1)

    def _grow(self, by: int) -> None:
        n, device = self.num_agents, self._sum.device
        self._sum = torch.cat((self._sum, torch.zeros(n, by, dtype=torch.float64, device=device)), dim=1)
        self._count = torch.cat((self._count, torch.zeros(n, by, dtype=torch.float64, device=device)), dim=1)
        self._low = torch.cat((self._low, torch.full((n, by), float("inf"), dtype=torch.float64, device=device)), dim=1)
        self._high = torch.cat((self._high, torch.full((n, by), -float("inf"), dtype=torch.float64, device=device)), dim=1)

    def _tables_to(self, device) -> None:
        n, k = self.num_agents, len(self._columns)
        self._sum = torch.zeros(n, k, dtype=torch.float64, device=device)
        self._count = torch.zeros(n, k, dtype=torch.float64, device=device)
        self._low = torch.full((n, k), float("inf"), dtype=torch.float64, device=device)
        self._high = torch.full((n, k), -float("inf"), dtype=torch.float64, device=device)

    def _commit(self) -> None:
        """Fold the queued stacks into the tables: one pass per value width, no host sync."""
        if not self._pending:
            return
        from robonuke_rl_core.learners.windows import stage_to_device

        pending, self._pending = self._pending, []
        k = len(self._columns)
        by_width: Dict[int, list] = {}
        for agent, names, stacked in pending:
            by_width.setdefault(stacked.shape[1], []).append((agent, names, stacked))
        for width, group in by_width.items():
            values = torch.cat([stacked for _, _, stacked in group])
            flat = [agent * k + self._columns[name] for agent, names, _ in group for name in names]
            index = torch.empty(len(flat), dtype=torch.long, device=values.device)
            stage_to_device(index, flat)
            finite = torch.isfinite(values)
            total = torch.where(finite, values, torch.zeros((), dtype=values.dtype, device=values.device)).sum(dim=1)
            self._sum.view(-1).index_add_(0, index, total)
            self._count.view(-1).index_add_(0, index, finite.sum(dim=1).to(torch.float64))
            spread = self._spread_rows(group)
            if spread:
                pick = torch.empty(len(spread), dtype=torch.long, device=values.device)
                stage_to_device(pick, spread)
                where = index[pick]
                infinity = torch.full((), float("inf"), dtype=values.dtype, device=values.device)
                low = torch.where(finite, values, infinity).amin(dim=1)[pick]
                high = torch.where(finite, values, -infinity).amax(dim=1)[pick]
                self._low.view(-1).scatter_reduce_(0, where, low, reduce="amin")
                self._high.view(-1).scatter_reduce_(0, where, high, reduce="amax")

    def _spread_rows(self, group) -> List[int]:
        """Rows of a width group's stacked values whose name is a distribution."""
        rows, at = [], 0
        for _, names, _ in group:
            for name in names:
                if name in self.distributions:
                    rows.append(at)
                at += 1
        return rows

    # ------------------------------------------------------------------ out
    def flush(self, agents: Optional[Iterable[int]] = None) -> List[Dict[str, float]]:
        """Mean per (agent, name) for the interval, then clear. One sync for everything. ``agents``: only these agents'
        (the others keep accumulating; their entries in the result are empty)."""
        chosen = list(range(self.num_agents)) if agents is None else [int(a) for a in agents]
        self._commit()
        out: List[Dict[str, float]] = [{} for _ in range(self.num_agents)]
        if self._sum is None or not self._columns or not chosen:
            self._clear(chosen)
            return out
        rows = torch.as_tensor(chosen, dtype=torch.long)
        if self._sum.device.type != "cpu":
            rows = rows.to(self._sum.device, non_blocking=True)
        # one sync for everything: the sums, the counts, the extremes of the chosen agents
        values = torch.stack((self._sum[rows], self._count[rows], self._low[rows], self._high[rows])).tolist()
        names = sorted(self._columns, key=self._columns.get)
        for i, agent in enumerate(chosen):
            sums, counts, lows, highs = values[0][i], values[1][i], values[2][i], values[3][i]
            for name, column in self._columns.items():
                if counts[column] > 0:  # every entry was NaN (or none came): the interval has nothing to say about it
                    out[agent][name] = sums[column] / counts[column]
            for name in names:
                column = self._columns[name]
                if name in self.distributions and counts[column] > 0:
                    out[agent][f"{name}{self.sep}min"] = lows[column]
                    out[agent][f"{name}{self.sep}max"] = highs[column]
        self._clear(chosen)
        return out

    def _clear(self, agents: Optional[Iterable[int]] = None) -> None:
        chosen = list(range(self.num_agents)) if agents is None else list(agents)
        self._pending = [entry for entry in self._pending if entry[0] not in chosen]
        if self._sum is None or not chosen:
            return
        rows = torch.as_tensor(chosen, dtype=torch.long)
        if self._sum.device.type != "cpu":
            rows = rows.to(self._sum.device, non_blocking=True)
        self._sum[rows] = 0.0
        self._count[rows] = 0.0
        self._low[rows] = float("inf")
        self._high[rows] = -float("inf")


def _version_tuple(raw: Any) -> tuple:
    """Leading integer components of a version string, e.g. ``"0.19.10rc1"`` -> (0, 19, 10)."""
    parts: List[int] = []
    for chunk in str(raw).split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class WandbLogger:
    """One wandb run per agent, fed by a shared :class:`MetricAccumulator`.

    Instances are the learner's ``on_log`` hook (``logger(agent, metrics, step)``) and
    :meth:`flush` is its ``on_flush`` hook, so the publish cadence is the learner's
    ``trainer.write_interval``. The x-axis of every point is the global env timestep.

    :param backend: the wandb module to use; ``None`` imports ``wandb`` lazily. Tests pass a
        fake backend, so none of this needs wandb installed.
    """

    def __init__(
        self,
        *,
        run_names: Sequence[str],
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Iterable[str] = (),
        mode: str = "online",
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Any] = None,
        device: Optional[Any] = None,
        distributions: Iterable[str] = (),
        backend: Any = None,
        flat_names: bool = False,
    ) -> None:
        names = list(run_names)
        if not names:
            raise ValueError("WandbLogger needs at least one run name")
        # The environment wins over the config. The HPC job script exports WANDB_MODE=offline
        # when the job has no usable key; an explicit `mode=` to wandb.init would beat that
        # env var and send an unattended job online to die, so the env var is applied here.
        env_mode = os.environ.get("WANDB_MODE", "")
        if env_mode and env_mode != mode:
            print(
                f"[wandb] WANDB_MODE={env_mode} is set in the environment; it overrides "
                f"wandb.mode={mode!r}",
                flush=True,
            )
            mode = env_mode
        if mode not in MODES:
            raise ValueError(f"wandb mode must be one of {MODES}, got {mode!r}")
        if backend is None:
            import wandb  # lazy: the package imports without wandb installed

            backend = wandb
        self._backend = backend
        self._require_concurrent_runs(len(names))

        self.run_names = names
        self.accumulator = MetricAccumulator(
            len(names), device=device, distributions=distributions or DISTRIBUTIONS, sep="_" if flat_names else "/"
        )
        self._last_step = 0
        #: per agent, a row logged with :meth:`log_now` and not yet committed: ``(step, values)``
        self._pending: List[Optional[tuple]] = [None] * len(names)
        self.runs = [
            backend.init(
                entity=entity,
                project=project,
                group=group,
                name=name,
                tags=list(tags),
                config=config,
                mode=mode,
                # independent handles: `wandb.run` and the global `wandb.log` are NOT used,
                # because with several live runs they would publish to whichever ran last
                reinit="create_new",
            )
            for name in names
        ]
        if config_path is not None:
            # the resolved config as a plain run file, so an eval can fetch it back verbatim
            self.save_file(config_path)

    # ------------------------------------------------------------------ files
    def save_file(self, path: Any, agent: Optional[int] = None, base_path: Optional[Any] = None) -> None:
        """Add a plain file to one run's Files tab, or to every run's when ``agent`` is None.

        Ordinary run files (``run.save``), never the Artifacts API: an eval downloads these
        back by name, so a file is stored under its base name and nothing else (or, with
        ``base_path``, under its path relative to that folder: a routed SAC's
        ``<phase>/checkpoints/<file>``, whose phases share file names).
        """
        from pathlib import Path

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"cannot upload {source}: not a file")
        runs = self.runs if agent is None else [self.runs[agent]]
        for run in runs:
            run.save(str(source), base_path=str(base_path if base_path is not None else source.parent), policy="now")

    def checkpoint(self, agent: int, step: int, path: Any) -> None:
        """The learner's ``on_checkpoint`` hook: mirror that agent's checkpoint file."""
        self.save_file(path, agent=agent)

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[Any] = None,
        device: Optional[Any] = None,
        backend: Any = None,
        flat_names: bool = False,
    ) -> "WandbLogger":
        """Build from a resolved config: one run per name in ``cfg.derived['run_names']``."""
        return cls(
            run_names=cfg.derived["run_names"],
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
            config=config,
            config_path=config_path,
            device=device,
            backend=backend,
            flat_names=flat_names,
        )

    def _require_concurrent_runs(self, runs: int) -> None:
        """Refuse to start unless this wandb can keep ``runs`` runs alive at once."""
        raw = getattr(self._backend, "__version__", None)
        version = _version_tuple(raw)
        if version < MIN_WANDB_VERSION:
            minimum = ".".join(str(part) for part in MIN_WANDB_VERSION)
            raise RuntimeError(
                f"wandb {raw} cannot hold {runs} concurrent runs in one process: one run per "
                f"agent needs reinit='create_new', added in wandb {minimum}. Upgrade wandb, or "
                "set wandb.mode=disabled to run without logging. Merging the agents into one "
                "run is not an option: their metrics would overwrite each other."
            )

    # ------------------------------------------------------------------ hooks
    def __call__(self, agent: int, metrics: Dict[str, torch.Tensor], step: int) -> None:
        """The ``on_log`` hook: accumulate, publish nothing yet."""
        self._last_step = max(self._last_step, int(step))
        self.accumulator.add(agent, metrics)

    def flush(self, step: int) -> None:
        """The ``on_flush`` hook: publish the interval's means at ``step`` (with a :meth:`log_now` row of the same step
        merged in)."""
        for agent, values in enumerate(self.accumulator.flush()):
            pending = self._pending[agent]
            if pending is not None and pending[0] < int(step):
                self._commit(agent)
                pending = None
            if pending is not None:
                values = {**pending[1], **values}
                self._pending[agent] = None
            if values:
                # commit now: wandb otherwise holds a row logged at an explicit step until a
                # later step arrives, so every point would show up one interval late
                self.runs[agent].log(values, step=int(step), commit=True)

    def log_now(self, agent: int, values: Dict[str, float], step: int) -> None:
        """Publish ``values`` for ``agent`` at ``step`` outside the interval (phase logging's per-phase points): held
        until a later step or the next :meth:`flush`, so a flush at the same step publishes one row."""
        self._last_step = max(self._last_step, int(step))
        pending = self._pending[agent]
        if pending is not None and pending[0] < int(step):
            self._commit(agent)
            pending = None
        if pending is None:
            self._pending[agent] = (int(step), dict(values))
        else:
            pending[1].update(values)

    def _commit(self, agent: int) -> None:
        step, values = self._pending[agent]
        self._pending[agent] = None
        if values:
            self.runs[agent].log(values, step=step, commit=True)

    def define_phase_axes(self, phases: Iterable[str]) -> None:
        """Plot each phase's ``<phase>_*`` metrics against its own ``<phase>_step`` in every run."""
        for run in self.runs:
            for phase in phases:
                run.define_metric(f"{phase}_step")
                run.define_metric(f"{phase}_*", step_metric=f"{phase}_step")

    def close(self) -> None:
        """Publish whatever is still pending (at the last step seen) and finish every run."""
        self.flush(self._last_step)
        for agent in range(len(self.runs)):
            if self._pending[agent] is not None:
                self._commit(agent)
        for run in self.runs:
            run.finish()

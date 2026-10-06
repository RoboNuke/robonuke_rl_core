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
)


class MetricAccumulator:
    """Running per-agent sums and counts, converted to Python only on :meth:`flush`.

    Values arrive as tensors of any shape: ``(envs_per_agent,)`` for a per-step env metric,
    ``(k,)`` for the episodes that ended this step (``k`` may be 0), 0-d for a learner metric.
    Each contributes its finite entries to a running sum and count, so the flushed value is
    the mean over everything the interval saw.

    **A NaN means "not applicable", not "zero".** An episode that did not break has no break
    step; one that never succeeded has no time-to-success. Those entries are NaN and are
    skipped — by the sum, the count and the min/max alike — so a metric that only some
    episodes have still reports the right average over the episodes that have it. The count
    therefore lives on the device too, and is converted with everything else at flush.
    """

    def __init__(
        self,
        num_agents: int,
        device: Optional[Any] = None,
        distributions: Iterable[str] = (),
    ) -> None:
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        self.num_agents = int(num_agents)
        self.device = torch.device(device) if device is not None else None
        #: names that also publish ``<name>/min`` and ``<name>/max`` — for a metric whose
        #: spread across envs says as much as its average (a return, an episode length, a Q)
        self.distributions = set(distributions)
        self._sums: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.num_agents)]
        #: finite entries seen, as a device tensor: counting them needs the NaN mask
        self._counts: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.num_agents)]
        self._extremes: List[Dict[str, tuple]] = [{} for _ in range(self.num_agents)]

    def add(self, agent: int, metrics: Dict[str, torch.Tensor]) -> None:
        """Fold one agent's metrics into the interval. No host sync."""
        if not 0 <= agent < self.num_agents:
            raise ValueError(f"agent {agent} is out of range for {self.num_agents} agents")
        sums, counts = self._sums[agent], self._counts[agent]
        for name, value in metrics.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"metric '{name}' must be a tensor, got {type(value).__name__}; the "
                    "accumulator sums on the device and converts once per flush"
                )
            if value.numel() == 0:
                continue  # an interval with nothing in it publishes nothing

            detached = value.detach().to(torch.float64)
            finite = torch.isfinite(detached)
            zero = torch.zeros((), dtype=detached.dtype, device=detached.device)
            total = torch.where(finite, detached, zero).sum()
            count = finite.sum().to(torch.float64)
            if name in sums:
                sums[name] = sums[name] + total
                counts[name] = counts[name] + count
            else:
                sums[name] = total
                counts[name] = count
            if name in self.distributions:
                infinity = torch.full((), float("inf"), dtype=detached.dtype, device=detached.device)
                low = torch.where(finite, detached, infinity).min()
                high = torch.where(finite, detached, -infinity).max()
                seen = self._extremes[agent].get(name)
                self._extremes[agent][name] = (
                    (low, high)
                    if seen is None
                    else (torch.minimum(seen[0], low), torch.maximum(seen[1], high))
                )

    def flush(self) -> List[Dict[str, float]]:
        """Mean per (agent, name) for the interval, then clear. One sync for everything."""
        keys = [(agent, name) for agent in range(self.num_agents) for name in self._sums[agent]]
        if not keys:
            self._clear()
            return [{} for _ in range(self.num_agents)]
        # one sync for everything: the sums, then the counts, then each distribution's extremes
        tensors = [self._sums[agent][name] for agent, name in keys]
        tensors += [self._counts[agent][name] for agent, name in keys]
        spread = [(agent, name) for agent, name in keys if name in self._extremes[agent]]
        for agent, name in spread:
            tensors.extend(self._extremes[agent][name])
        values = torch.stack(tensors).tolist()

        out: List[Dict[str, float]] = [{} for _ in range(self.num_agents)]
        counts = {}
        for index, (agent, name) in enumerate(keys):
            total, count = values[index], values[len(keys) + index]
            counts[(agent, name)] = count
            if count > 0:  # every entry was NaN: the interval has nothing to say about it
                out[agent][name] = total / count
        rest = values[2 * len(keys) :]
        for index, (agent, name) in enumerate(spread):
            if counts[(agent, name)] > 0:
                out[agent][f"{name}/min"] = rest[2 * index]
                out[agent][f"{name}/max"] = rest[2 * index + 1]
        self._clear()
        return out

    def _clear(self) -> None:
        for agent in range(self.num_agents):
            self._sums[agent] = {}
            self._counts[agent] = {}
            self._extremes[agent] = {}


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
    ) -> None:
        names = list(run_names)
        if not names:
            raise ValueError("WandbLogger needs at least one run name")
        if mode not in MODES:
            raise ValueError(f"wandb mode must be one of {MODES}, got {mode!r}")
        if backend is None:
            import wandb  # lazy: the package imports without wandb installed

            backend = wandb
        self._backend = backend
        self._require_concurrent_runs(len(names))

        self.run_names = names
        self.accumulator = MetricAccumulator(
            len(names), device=device, distributions=distributions or DISTRIBUTIONS
        )
        self._last_step = 0
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
    def save_file(self, path: Any, agent: Optional[int] = None) -> None:
        """Add a plain file to one run's Files tab, or to every run's when ``agent`` is None.

        Ordinary run files (``run.save``), never the Artifacts API: an eval downloads these
        back by name, so a file is stored under its base name and nothing else.
        """
        from pathlib import Path

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"cannot upload {source}: not a file")
        runs = self.runs if agent is None else [self.runs[agent]]
        for run in runs:
            run.save(str(source), base_path=str(source.parent), policy="now")

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
        """The ``on_flush`` hook: publish the interval's means at ``step``."""
        for agent, values in enumerate(self.accumulator.flush()):
            if values:
                self.runs[agent].log(values, step=int(step))

    def close(self) -> None:
        """Publish whatever is still pending (at the last step seen) and finish every run."""
        self.flush(self._last_step)
        for run in self.runs:
            run.finish()

"""Shared base for the block-parallel learners.

``LearnerBase`` holds everything that is independent of the RL algorithm: the env
partition, env-metric forwarding, episode statistics, the two hook lists, and per-agent
checkpoints. It subclasses skrl's ``Agent`` so skrl's ``SequentialTrainer`` drives it.

Agent ``i`` owns envs ``[i*envs_per_agent, (i+1)*envs_per_agent)`` and nothing it computes
may depend on another agent's data; see CLAUDE.md's independence rule.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from skrl.agents.torch import Agent
from skrl.agents.torch.base import AgentCfg
from skrl.agents.torch.base import ExperimentCfg as SkrlExperimentCfg
from skrl.models.torch import Model

from ..losses.losses import LossContext
from ..memory.multi_random import MultiRandomMemory
from ..models.normalizer import BlockRunningNorm
from .cfg import TrainerCfg

__all__ = [
    "LearnerBase",
    "run_dirs",
    "CHECKPOINT_EXTENSION",
    "CHECKPOINT_BEST",
    "checkpoint_name",
    "SUCCESS_METRIC",
    "BEST_SUCCESS_METRIC",
]

#: one place for the checkpoint file naming, so eval, wandb and the loader cannot disagree
CHECKPOINT_EXTENSION = "ckpt"
CHECKPOINT_BEST = f"ckpt_best.{CHECKPOINT_EXTENSION}"

#: the env episode channel the best-checkpoint success rate is measured from
SUCCESS_METRIC = "episode/success"
#: the success rate of the interval that produced the current best checkpoint. Training only:
#: it is published from the write-interval flush, which eval never runs.
BEST_SUCCESS_METRIC = "episode/best_success_rate"


def checkpoint_name(step: int) -> str:
    """The file name for a periodic checkpoint at ``step``."""
    return f"ckpt_{int(step)}.{CHECKPOINT_EXTENSION}"


@dataclass
class _SkrlCfg(AgentCfg):
    """Minimal cfg for skrl's ``Agent``. Our own section cfg lives in ``self.cfg``."""

    experiment: SkrlExperimentCfg = field(default_factory=SkrlExperimentCfg)


class _UpdateTimer:
    """Times the block it wraps and records it on the learner."""

    def __init__(self, learner: "LearnerBase") -> None:
        self.learner = learner

    def __enter__(self) -> "_UpdateTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.learner.update_ms = (time.perf_counter() - self._start) * 1.0e3
        return None


#: bytes per memory page, for reading /proc
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def process_rss_mb() -> float:
    """This process's resident set size in MiB, from ``/proc/self/statm``.

    Read from /proc rather than through psutil so the package keeps no dependency for one
    number; Isaac Sim is Linux-only, so /proc is always there. RSS is the figure that
    answers "is this run going to be OOM-killed" — it counts what is actually resident,
    including everything Kit and PhysX hold, which a torch-side number cannot see.
    """
    with open("/proc/self/statm", "r") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * _PAGE_SIZE / float(1 << 20)


def run_dirs(cfg: Any) -> List[Path]:
    """``{output_dir}/{project}/{group}/{run_name}`` for each agent, in agent order."""
    return [
        Path(cfg.trainer.output_dir) / cfg.wandb.project / cfg.wandb.group / name
        for name in cfg.derived["run_names"]
    ]


class LearnerBase(Agent):
    def __init__(
        self,
        *,
        models: Dict[str, Model],
        memory: MultiRandomMemory | None,
        observation_space,
        action_space,
        state_space=None,
        device=None,
        cfg,
        trainer_cfg: TrainerCfg,
        num_agents: int,
        num_envs: int,
        run_dirs: List[Path] | None = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_agents ({num_agents}): each "
                "agent owns a contiguous block of envs"
            )
        if run_dirs is not None and len(run_dirs) != num_agents:
            raise ValueError(f"expected {num_agents} run dirs, got {len(run_dirs)}")

        skrl_cfg = _SkrlCfg(
            experiment=SkrlExperimentCfg(
                write_interval=0,  # we publish through the on_log hooks, not skrl's writer
                checkpoint_interval=0,  # we write per-agent checkpoints ourselves
            )
        )
        super().__init__(
            cfg=skrl_cfg,
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        self.cfg = cfg  # our section cfg (SACCfg or PPOCfg)
        self.trainer_cfg = trainer_cfg
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.envs_per_agent = num_envs // num_agents
        self.run_dirs = [Path(p) for p in run_dirs] if run_dirs is not None else None

        #: ``fn(agent_idx, metrics, step)``; env metrics arrive as ``(envs_per_agent,)``
        #: tensors, learner metrics as 0-d tensors. Empty by default: nothing is logged.
        self.on_log: List[Callable[[int, Dict[str, torch.Tensor], int], None]] = []
        #: ``fn(step)``; called once per ``write_interval``, after the episode flush. The
        #: logger's publish point: metrics accumulate through ``on_log`` and leave here.
        self.on_flush: List[Callable[[int], None]] = []
        #: ``fn(agent_idx, step, path)``; called for every checkpoint file written, so a
        #: logger can mirror it. Empty by default: checkpoints are local files only.
        self.on_checkpoint: List[Callable[[int, int, Path], None]] = []
        #: ``fn(ctx) -> Tensor | None``; a returned tensor is added to that loss. Empty by default.
        self.aux_loss: List[Callable[[LossContext], Optional[torch.Tensor]]] = []

        # Episode statistics: per-env running return/length, and per-agent sums over the
        # write interval. All on-device, so a step with a finished env costs no GPU->CPU sync;
        # the only sync is one read per agent at the interval flush.
        self._episode_return = torch.zeros(num_envs, device=self.device)
        self._episode_length = torch.zeros(num_envs, device=self.device)
        self._finished_returns = torch.zeros(num_agents, device=self.device)
        self._finished_lengths = torch.zeros(num_agents, device=self.device)
        self._finished_count = torch.zeros(num_agents, device=self.device)
        self._best_return: List[float] = [float("-inf")] * num_agents
        self._mean_return: List[float] = [float("nan")] * num_agents
        #: per-agent success rate over the interval, from the env's episode/success channel
        self._success_total = torch.zeros(num_agents, device=self.device)
        self._success_count = torch.zeros(num_agents, device=self.device)
        #: the success rate of the interval that produced the current best checkpoint, and
        #: which agents produced a new best in the interval just flushed
        self._best_success: List[float] = [float("nan")] * num_agents
        self._new_best: List[bool] = [False] * num_agents

        self._rewards_shaper = self._resolve_rewards_shaper()
        #: wall-clock ms of the last update, filled by :meth:`update_timer`
        self.update_ms = 0.0

    # ------------------------------------------------------------------ setup
    def _resolve_rewards_shaper(self) -> Optional[Callable]:
        ref = getattr(self.cfg, "rewards_shaper", None)
        if ref is None:
            return None
        from ..config import resolve_callable

        return resolve_callable(ref)

    def init(self, *, trainer_cfg: Dict[str, Any] | None = None) -> None:
        """Create the memory tensors. No writers: metrics leave through ``on_log``."""
        self._create_memory_tensors()

    def _create_memory_tensors(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ aux-loss memory
    def aux_memory_keys(self) -> Dict[str, int]:
        """Per-transition tensors the ``aux_loss`` hooks need: name -> width, merged.

        Read once, in ``_create_memory_tensors``, which the trainer triggers through
        :meth:`init`. A hook appended after that contributes nothing and the loss raises by
        name when it goes looking for its key — better than a silently absent supervision
        signal.
        """
        keys: Dict[str, int] = {}
        for hook in self.aux_loss:
            for key, width in (getattr(hook, "memory_keys", None) or {}).items():
                width = int(width)
                if keys.get(key, width) != width:
                    raise ValueError(
                        f"two aux_loss hooks want memory key '{key}' at different widths: "
                        f"{keys[key]} and {width}"
                    )
                keys[key] = width
        return keys

    def create_aux_memory_tensors(self) -> List[str]:
        """Create those tensors in the memory; returns their names for ``_tensors_names``."""
        keys = self.aux_memory_keys()
        if keys and self.memory is None:
            raise ValueError(
                f"an aux loss needs the per-transition tensors {sorted(keys)} but this "
                "learner has no memory"
            )
        self._aux_memory_keys = keys
        for name, width in sorted(keys.items()):
            self.memory.create_tensor(name=name, size=width, dtype=torch.float32)
        return sorted(keys)

    def aux_memory_values(self, infos: Any) -> Dict[str, torch.Tensor]:
        """This step's values for those tensors, read from ``infos`` and shape-checked.

        The env publishes each one under its own top-level ``infos`` key, as
        ``(num_envs, width)`` (or ``(num_envs,)`` when the width is 1). Anything else is a
        mismatch between the loss and the env, so it raises rather than training on a
        broadcast.
        """
        keys = getattr(self, "_aux_memory_keys", None) or {}
        if not keys:
            return {}
        out: Dict[str, torch.Tensor] = {}
        for name, width in keys.items():
            value = infos.get(name) if isinstance(infos, dict) else None
            if value is None:
                raise RuntimeError(
                    f"an aux loss needs infos[{name!r}] every step, and this step did not "
                    f"carry it (infos keys: {sorted(infos) if isinstance(infos, dict) else type(infos).__name__}). "
                    "The wrapper that publishes it has to be enabled."
                )
            if not torch.is_tensor(value):
                raise TypeError(
                    f"infos[{name!r}] must be a tensor, got {type(value).__name__}"
                )
            shaped = value.reshape(self.num_envs, -1) if value.dim() > 1 else value.reshape(-1, 1)
            if shaped.shape != (self.num_envs, width):
                raise ValueError(
                    f"infos[{name!r}] must be ({self.num_envs}, {width}), got "
                    f"{tuple(value.shape)}"
                )
            out[name] = shaped.to(dtype=torch.float32, device=self.device)
        return out

    def make_normalizer(self, space) -> BlockRunningNorm:
        """A per-agent input normalizer sized to ``space``."""
        from skrl.utils.spaces.torch import compute_space_size

        return BlockRunningNorm(
            self.num_agents,
            compute_space_size(space, occupied_size=True),
            clip_threshold=self.cfg.normalizer_clip,
            device=self.device,
        )

    # ------------------------------------------------------------------ hooks
    def emit(self, agent_idx: int, metrics: Dict[str, torch.Tensor], step: int) -> None:
        """Hand one agent's metrics to every ``on_log`` hook."""
        if not self.on_log:
            return
        for hook in self.on_log:
            hook(agent_idx, metrics, step)

    def emit_per_agent(self, metrics: Dict[str, torch.Tensor], step: int) -> None:
        """Emit learner metrics given as ``(num_agents,)`` tensors: one 0-d value per agent."""
        if not self.on_log:
            return
        for name, value in metrics.items():
            if not torch.is_tensor(value) or value.shape != (self.num_agents,):
                raise TypeError(
                    f"learner metric '{name}' must be a tensor of shape ({self.num_agents},), got "
                    f"{tuple(value.shape) if torch.is_tensor(value) else type(value).__name__}"
                )
        for agent in range(self.num_agents):
            self.emit(agent, {name: value[agent] for name, value in metrics.items()}, step)

    def emit_selection(self, outputs: Any, step: int) -> None:
        """``selection/p_force_<axis>``: how likely each hybrid axis is to be force-controlled.

        The selection bit is 1 for force (``models/simba.SELECTION_BIT_IS_FORCE``), so the
        actor's ``selection_prob`` *is* this probability, with no complement to take.
        The policy's own probability, averaged over the agent's envs — not the realized
        choice, which the controller wrapper publishes as ``selection/force_<axis>``. The
        probability is the smoother of the two and is what the supervised selection loss
        moves, so it is the one to watch when asking whether a policy is learning *when* to
        push. A policy without selection dims emits nothing.

        The axis names come from the model, which the factory names from the controller's
        force-eligible axes; a model built without one falls back to the bit's index.
        """
        if not self.on_log or not isinstance(outputs, dict):
            return
        probability = outputs.get("selection_prob")
        if probability is None:
            return
        per_agent = probability.view(self.num_agents, self.envs_per_agent, -1).mean(dim=1)
        names = getattr(self.policy, "selection_names", None) or [
            str(index) for index in range(per_agent.shape[-1])
        ]
        self.emit_per_agent(
            {f"selection/p_force_{name}": per_agent[:, index] for index, name in enumerate(names)},
            step,
        )

    def compute_aux_loss(self, ctx: LossContext) -> Optional[torch.Tensor]:
        """Sum of what the ``aux_loss`` hooks return for ``ctx.target``, or None.

        ``losses.build_aux_losses`` builds the usual hook; a project can append its own.
        """
        total = None
        for hook in self.aux_loss:
            value = hook(ctx)
            if value is None:
                continue
            if not torch.is_tensor(value):
                raise TypeError(
                    f"an aux_loss hook returned {type(value).__name__}; return a tensor or None"
                )
            total = value if total is None else total + value
        return total

    # ------------------------------------------------------------------ env metrics
    def observe_step(
        self,
        *,
        infos: Any,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        step: int,
    ) -> None:
        """Everything the base does with one env step, in one place.

        Forwards both env metric channels and keeps the episode bookkeeping that feeds both
        ``episode/*`` and the best-checkpoint rule. The learners call this once from
        ``record_transition``, before they touch the memory.
        """
        self._forward_env_metrics(infos, step)
        if self.on_log:
            # the instantaneous reward, per env: its spread across envs is published beside
            # the mean (see logging.DISTRIBUTIONS), which an episode return cannot show
            flat = rewards.reshape(-1)
            for agent in range(self.num_agents):
                lo = agent * self.envs_per_agent
                self.emit(agent, {"reward/step": flat[lo : lo + self.envs_per_agent]}, step)

        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1))
        self._episode_return += rewards.reshape(-1)
        self._episode_length += 1.0

        # The compaction below is the step's only host sync, and only when someone is
        # listening: with no ``on_log`` hook the episode channels are skipped entirely.
        slots = self._episode_slots(done) if self.on_log else None
        if slots is not None:
            self._emit_finished_episodes(slots, step)
        self._forward_episode_metrics(infos, slots, step)
        self._fold_finished_episodes(done)

    def _forward_env_metrics(self, infos: Any, step: int) -> None:
        """Slice ``infos["metrics_to_log"]`` to each agent's envs and emit it as is."""
        metrics = self._env_metrics(infos, "metrics_to_log")
        if metrics is None:
            return
        for agent in range(self.num_agents):
            lo = agent * self.envs_per_agent
            hi = lo + self.envs_per_agent
            self.emit(agent, {name: value[lo:hi] for name, value in metrics.items()}, step)

    def _forward_episode_metrics(
        self, infos: Any, slots: Optional[List[torch.Tensor]], step: int
    ) -> None:
        """Forward ``infos["episode_metrics_to_log"]`` for the envs that finished this step.

        Each agent gets the compacted ``(k,)`` values of its own finished envs; an agent with
        none gets nothing, so an interval without episodes publishes no point. Slots of envs
        that did not finish are never read — an env may leave anything there.
        """
        metrics = self._env_metrics(infos, "episode_metrics_to_log")
        if metrics is None or slots is None:
            return
        success = metrics.get(SUCCESS_METRIC)
        for agent, index in enumerate(slots):
            if index.numel() == 0:
                continue
            lo = agent * self.envs_per_agent
            hi = lo + self.envs_per_agent
            if success is not None:
                # kept on device: the interval's single sync is in _flush_episode_stats
                mine = success[lo:hi].index_select(0, index)
                self._success_total[agent] += mine.sum()
                self._success_count[agent] += mine.numel()
            self.emit(
                agent,
                {
                    name: value[lo:hi].index_select(0, index)
                    for name, value in metrics.items()
                },
                step,
            )

    def _env_metrics(self, infos: Any, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """The validated ``infos[key]`` dict, or None when the env does not provide it."""
        if not isinstance(infos, dict):
            return None
        metrics = infos.get(key)
        if metrics is None:
            return None  # nothing to forward
        if not isinstance(metrics, dict):
            raise TypeError(f"infos['{key}'] must be a dict, got {type(metrics).__name__}")
        for name, value in metrics.items():
            if not torch.is_tensor(value) or value.shape != (self.num_envs,):
                raise TypeError(
                    f"infos['{key}']['{name}'] must be a tensor of shape "
                    f"({self.num_envs},), got "
                    f"{tuple(value.shape) if torch.is_tensor(value) else type(value).__name__}"
                )
        return metrics

    # ------------------------------------------------------------------ episode statistics
    def _episode_slots(self, done: torch.Tensor) -> Optional[List[torch.Tensor]]:
        """Local env indices of the envs that finished this step, per agent, or None if none."""
        finished = torch.nonzero(done, as_tuple=False).flatten().tolist()
        if not finished:
            return None
        per_agent: List[List[int]] = [[] for _ in range(self.num_agents)]
        for env in finished:
            per_agent[env // self.envs_per_agent].append(env % self.envs_per_agent)
        return [
            torch.tensor(local, dtype=torch.long, device=self.device) for local in per_agent
        ]

    def _emit_finished_episodes(self, slots: List[torch.Tensor], step: int) -> None:
        """Emit the return and length of every episode that ended this step, per agent."""
        for agent, index in enumerate(slots):
            if index.numel() == 0:
                continue
            lo = agent * self.envs_per_agent
            hi = lo + self.envs_per_agent
            self.emit(
                agent,
                {
                    "episode/return": self._episode_return[lo:hi].index_select(0, index),
                    "episode/length": self._episode_length[lo:hi].index_select(0, index),
                },
                step,
            )

    def _fold_finished_episodes(self, done: torch.Tensor) -> None:
        """Fold finished envs into their agent's interval sums, then reset those envs.

        On-device and sync-free: this is what the best-checkpoint rule reads, so it runs
        whether or not anything is logging.
        """
        done_f = done.to(self._episode_return.dtype)
        blocks = (self.num_agents, self.envs_per_agent)
        self._finished_count += done_f.view(blocks).sum(dim=1)
        self._finished_returns += (self._episode_return * done_f).view(blocks).sum(dim=1)
        self._finished_lengths += (self._episode_length * done_f).view(blocks).sum(dim=1)
        keep = 1.0 - done_f
        self._episode_return *= keep
        self._episode_length *= keep

    def _flush_episode_stats(self, step: int) -> None:
        """Close the interval: refresh the per-agent mean return and emit the episode count.

        The returns and lengths themselves left through :meth:`_emit_finished_episodes` as the
        episodes ended; what is left here is the one value the best-checkpoint rule needs (a
        single sync per interval) and the interval's episode count.
        """
        counts = self._finished_count.tolist()  # the interval's single sync
        returns = self._finished_returns.tolist()
        success_total = self._success_total.tolist()
        success_count = self._success_count.tolist()
        for agent, count in enumerate(counts):
            if count > 0:
                self._mean_return[agent] = returns[agent] / count
                self.emit(
                    agent,
                    {"episode/count": torch.tensor(count, device=self.device)},
                    step,
                )
        self._update_best(step, success_total, success_count)
        self._finished_count.zero_()
        self._finished_returns.zero_()
        self._finished_lengths.zero_()
        self._success_total.zero_()
        self._success_count.zero_()

    def _update_best(self, step: int, success_total: List[float], success_count: List[float]) -> None:
        """Advance the best-return record, and publish the best checkpoint's success rate.

        ``episode/best_success_rate`` is the success rate measured over the interval that
        produced the current ``ckpt_best``, republished every interval so the line is flat
        between improvements. It answers "how good is the policy I would actually ship",
        which the live success rate does not: that one wanders with exploration.

        **It can go down.** The best checkpoint is chosen by mean episode return, so a new
        best-by-return may have had a worse success rate than the one it replaced. Making it
        monotonic would mean reporting a number that no checkpoint on disk ever achieved.

        The decision lives here, not in :meth:`_write_best_checkpoints`, because the flush
        hooks publish before checkpoints are written; deciding later would delay the metric
        by a whole interval.
        """
        for agent in range(self.num_agents):
            mean_return = self._mean_return[agent]
            improved = mean_return == mean_return and mean_return > self._best_return[agent]
            self._new_best[agent] = improved
            if improved:
                self._best_return[agent] = mean_return
                if success_count[agent] > 0:
                    self._best_success[agent] = success_total[agent] / success_count[agent]
            rate = self._best_success[agent]
            if rate == rate:  # not NaN: an interval with episodes has been seen
                self.emit(
                    agent,
                    {BEST_SUCCESS_METRIC: torch.tensor(rate, device=self.device)},
                    step,
                )

    @property
    def mean_returns(self) -> List[float]:
        """Per-agent mean episode return over the last flushed interval (nan before the first)."""
        return list(self._mean_return)

    # ------------------------------------------------------------------ trainer plumbing
    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Update, then flush episode stats and write checkpoints on their intervals."""
        if self.training:
            self._update_if_ready(timestep=timestep, timesteps=timesteps)

        step = timestep + 1
        write_interval = self.trainer_cfg.write_interval
        if write_interval > 0 and step % write_interval == 0:
            self._flush_episode_stats(timestep)
            for hook in self.on_flush:
                hook(timestep)
            if self.training:
                self._write_best_checkpoints(timestep)
        checkpoint_interval = self.trainer_cfg.checkpoint_interval
        if self.training and checkpoint_interval > 0 and step % checkpoint_interval == 0:
            self.save_checkpoints(timestep)

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        raise NotImplementedError

    def stats_metrics(self) -> Dict[str, torch.Tensor]:
        """``stats/*``: what the last update cost, in wall time and in memory.

        Every value is **process-wide**, so each agent gets the same number — they share one
        process, one GPU and one Isaac Sim. They are still emitted per agent so that one
        agent's wandb run is self-contained.

        ``gpu_used_mb`` is the whole device (``mem_get_info``: total minus free), which is
        the number that answers "will this fit", because it counts what Kit and PhysX hold as
        well as us. ``gpu_torch_reserved_mb`` is our allocator's share of it, so a growing
        gap between the two says the leak is not in the learner.
        """
        values = {"update_time_ms": self.update_ms, "ram_mb": process_rss_mb()}
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            values["gpu_used_mb"] = (total - free) / float(1 << 20)
            values["gpu_torch_reserved_mb"] = torch.cuda.memory_reserved(self.device) / float(
                1 << 20
            )
        return {
            f"stats/{name}": torch.full((self.num_agents,), float(value), device=self.device)
            for name, value in values.items()
        }

    def update_timer(self):
        """Context manager timing one update; the elapsed ms land in ``self.update_ms``.

        Wall time around the update, the way skrl's own "algorithm update time" is measured.
        No explicit CUDA sync: the next rollout step depends on these weights, so the queue
        drains anyway, and a sync here would cost more than it measures.
        """
        return _UpdateTimer(self)

    def random_actions(self, rows: int) -> torch.Tensor:
        """Uniform actions on [-1, 1], the squashed policy's support.

        skrl's default would sample the env's advertised Box, which Isaac Lab reports as
        (-inf, +inf), filling the buffer with actions the policy can never produce.
        """
        return torch.rand(rows, *self.action_space.shape, device=self.device) * 2.0 - 1.0

    def shape_rewards(self, rewards: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        if self._rewards_shaper is None:
            return rewards
        return self._rewards_shaper(rewards, timestep, timesteps)

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_model_keys(self) -> List[str]:
        """Attribute names of block-parallel models to slice per agent."""
        raise NotImplementedError

    def _checkpoint_optimizer_keys(self) -> List[str]:
        """Attribute names of optimizers whose state is sliced per agent."""
        raise NotImplementedError

    def _checkpoint_normalizers(self) -> Dict[str, BlockRunningNorm]:
        """Name -> per-agent normalizer to save. ``{}`` when normalization is off."""
        return {}

    def _checkpoint_extras(self, agent: int) -> Dict[str, Any]:
        """Learner-specific per-agent state (e.g. SAC's entropy coefficient)."""
        return {}

    def _load_extras(self, agent: int, extras: Dict[str, Any], path: Path) -> None:
        """Load what :meth:`_checkpoint_extras` wrote into slot ``agent``."""
        if extras:
            raise KeyError(
                f"{path} holds learner extras {sorted(extras)} but {type(self).__name__} does not "
                "load any"
            )

    def _build_checkpoint(self, agent: int, step: int) -> Dict[str, Any]:
        return {
            "step": int(step),
            "num_agents": int(self.num_agents),
            "agent_idx": int(agent),
            "mean_return": float(self._mean_return[agent]),
            "models": {
                key: getattr(self, key).agent_state_dict(agent)
                for key in self._checkpoint_model_keys()
            },
            "optimizers": {
                key: getattr(self, key).agent_state_dict(agent)
                for key in self._checkpoint_optimizer_keys()
            },
            "normalizers": {
                name: norm.state_dict_for(agent)
                for name, norm in self._checkpoint_normalizers().items()
            },
            "extras": self._checkpoint_extras(agent),
        }

    def checkpoint_dir(self, agent: int) -> Path:
        if self.run_dirs is None:
            raise RuntimeError(
                f"{type(self).__name__} has no run_dirs, so it cannot write checkpoints; pass "
                "run_dirs=run_dirs(cfg) to the constructor"
            )
        return self.run_dirs[agent] / "checkpoints"

    def save_checkpoints(self, step: int) -> List[Path]:
        """Write one checkpoint per agent (see :func:`checkpoint_name`); return the paths."""
        paths = []
        for agent in range(self.num_agents):
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / checkpoint_name(step)
            torch.save(self._build_checkpoint(agent, step), path)
            self._announce_checkpoint(agent, step, path)
            paths.append(path)
        return paths

    def _announce_checkpoint(self, agent: int, step: int, path: Path) -> None:
        for hook in self.on_checkpoint:
            hook(agent, int(step), path)

    def _write_best_checkpoints(self, step: int) -> None:
        """Rewrite the best checkpoint for each agent whose mean episode return improved."""
        if self.trainer_cfg.checkpoint_interval <= 0 or self.run_dirs is None:
            return
        for agent in range(self.num_agents):
            if not self._new_best[agent]:
                continue  # no finished episode yet, or no improvement (decided in the flush)
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / CHECKPOINT_BEST
            torch.save(self._build_checkpoint(agent, step), path)
            self._announce_checkpoint(agent, step, path)

    def load_agent(self, path: str | Path, slot: int, *, with_optimizer: bool = True) -> Dict[str, Any]:
        """Load one agent's checkpoint file into slot ``slot``; return its metadata.

        Model weights, normalizer stats, learner extras and (unless ``with_optimizer`` is
        False) the optimizer moments for that one slot. BlockAdamW keeps its state per agent,
        so this works at any ``num_agents``.
        """
        path = Path(path)
        if not 0 <= slot < self.num_agents:
            raise ValueError(f"slot {slot} is out of range for num_agents={self.num_agents}")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing = {"step", "num_agents", "agent_idx", "models", "optimizers", "normalizers", "extras"} - set(ckpt)
        if missing:
            raise KeyError(f"{path} is missing required checkpoint keys: {sorted(missing)}")

        model_keys = self._checkpoint_model_keys()
        if set(ckpt["models"]) != set(model_keys):
            raise KeyError(
                f"{path} holds models {sorted(ckpt['models'])} but {type(self).__name__} expects "
                f"{sorted(model_keys)}"
            )
        for key in model_keys:
            getattr(self, key).load_agent_state_dict(slot, ckpt["models"][key])

        normalizers = self._checkpoint_normalizers()
        if set(ckpt["normalizers"]) != set(normalizers):
            raise KeyError(
                f"{path} holds normalizers {sorted(ckpt['normalizers'])} but "
                f"{type(self).__name__} expects {sorted(normalizers)}"
            )
        for name, norm in normalizers.items():
            norm.load_state_dict_into(slot, ckpt["normalizers"][name])

        self._load_extras(slot, ckpt["extras"], path)

        if with_optimizer:
            expected = set(self._checkpoint_optimizer_keys())
            if set(ckpt["optimizers"]) != expected:
                raise KeyError(
                    f"{path} holds optimizer state {sorted(ckpt['optimizers'])} but "
                    f"{type(self).__name__} expects {sorted(expected)}"
                )
            for key in expected:
                getattr(self, key).load_agent_state_dict(slot, ckpt["optimizers"][key])

        return {k: ckpt[k] for k in ("step", "num_agents", "agent_idx", "mean_return")}

    @staticmethod
    def latest_checkpoint(directory: str | Path) -> Path:
        """The highest-step periodic checkpoint in ``directory``."""
        extension = CHECKPOINT_EXTENSION
        candidates = glob.glob(os.path.join(str(directory), f"ckpt_*.{extension}"))
        pattern = re.compile(rf"ckpt_(\d+)\.{extension}$")
        steps = [(int(m.group(1)), p) for p in candidates if (m := pattern.search(p))]
        if not steps:
            raise FileNotFoundError(f"no ckpt_<step>.{extension} files in {directory}")
        return Path(max(steps)[1])

    # ------------------------------------------------------------------ small helpers
    def expand_per_agent(self, per_agent: torch.Tensor, rows: int) -> torch.Tensor:
        """``(N, 1) -> (N*rows, 1)`` to broadcast against a flat batch."""
        return per_agent.repeat_interleave(rows, dim=0)

    def per_agent_mean(self, flat: torch.Tensor, rows: int) -> torch.Tensor:
        """``(N*rows, *) -> (N,)``: each agent's mean over its own rows only."""
        return flat.view(self.num_agents, rows, -1).mean(dim=(1, 2))

    def __str__(self) -> str:  # skrl's __str__ assumes its own cfg dataclass
        return f"{type(self).__name__}(num_agents={self.num_agents}, cfg={dataclasses.asdict(self.cfg)})"

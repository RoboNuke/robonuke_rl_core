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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from skrl.agents.torch import Agent
from skrl.agents.torch.base import AgentCfg
from skrl.agents.torch.base import ExperimentCfg as SkrlExperimentCfg
from skrl.memories.torch import Memory
from skrl.models.torch import Model

from ..models.block_utils import (
    assign_block_slice,
    merge_optimizer_states,
    slice_block_state_dict,
    slice_optimizer_state,
)
from ..losses.losses import LossContext
from ..models.normalizer import BlockRunningNorm
from .cfg import TrainerCfg

__all__ = ["LearnerBase", "run_dirs"]


@dataclass
class _SkrlCfg(AgentCfg):
    """Minimal cfg for skrl's ``Agent``. Our own section cfg lives in ``self.cfg``."""

    experiment: SkrlExperimentCfg = field(default_factory=SkrlExperimentCfg)


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
        memory: Memory | None,
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
        self.cfg = cfg  # our section cfg (SACCfg / PPOCfg / FlashSACCfg)
        self.trainer_cfg = trainer_cfg
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.envs_per_agent = num_envs // num_agents
        self.run_dirs = [Path(p) for p in run_dirs] if run_dirs is not None else None

        #: ``fn(agent_idx, metrics, step)``; env metrics arrive as ``(envs_per_agent,)``
        #: tensors, learner metrics as 0-d tensors. Empty by default: nothing is logged.
        self.on_log: List[Callable[[int, Dict[str, torch.Tensor], int], None]] = []
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

        self._rewards_shaper = self._resolve_rewards_shaper()

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
    def _forward_env_metrics(self, infos: Any, step: int) -> None:
        """Slice ``infos["metrics_to_log"]`` to each agent's envs and emit it as is."""
        if not isinstance(infos, dict):
            return
        metrics = infos.get("metrics_to_log")
        if metrics is None:
            return  # nothing to forward
        if not isinstance(metrics, dict):
            raise TypeError(
                f"infos['metrics_to_log'] must be a dict, got {type(metrics).__name__}"
            )
        for name, value in metrics.items():
            if not torch.is_tensor(value) or value.shape != (self.num_envs,):
                raise TypeError(
                    f"infos['metrics_to_log']['{name}'] must be a tensor of shape "
                    f"({self.num_envs},), got "
                    f"{tuple(value.shape) if torch.is_tensor(value) else type(value).__name__}"
                )
        for agent in range(self.num_agents):
            lo = agent * self.envs_per_agent
            hi = lo + self.envs_per_agent
            self.emit(agent, {name: value[lo:hi] for name, value in metrics.items()}, step)

    # ------------------------------------------------------------------ episode statistics
    def _track_episodes(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Running per-env return/length; a finished env folds into its agent's sums."""
        self._episode_return += rewards.reshape(-1)
        self._episode_length += 1.0
        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1)).to(
            self._episode_return.dtype
        )
        blocks = (self.num_agents, self.envs_per_agent)
        self._finished_count += done.view(blocks).sum(dim=1)
        self._finished_returns += (self._episode_return * done).view(blocks).sum(dim=1)
        self._finished_lengths += (self._episode_length * done).view(blocks).sum(dim=1)
        keep = 1.0 - done
        self._episode_return *= keep
        self._episode_length *= keep

    def _flush_episode_stats(self, step: int) -> None:
        """Emit per-agent mean episode return/length for this interval, then clear."""
        counts = self._finished_count.tolist()  # the interval's single sync
        returns = self._finished_returns.tolist()
        lengths = self._finished_lengths.tolist()
        for agent, count in enumerate(counts):
            if count > 0:
                self._mean_return[agent] = returns[agent] / count
                self.emit(
                    agent,
                    {
                        "episode/return": torch.tensor(self._mean_return[agent], device=self.device),
                        "episode/length": torch.tensor(lengths[agent] / count, device=self.device),
                        "episode/count": torch.tensor(count, device=self.device),
                    },
                    step,
                )
        self._finished_count.zero_()
        self._finished_returns.zero_()
        self._finished_lengths.zero_()

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
            if self.training:
                self._write_best_checkpoints(timestep)
        checkpoint_interval = self.trainer_cfg.checkpoint_interval
        if self.training and checkpoint_interval > 0 and step % checkpoint_interval == 0:
            self.save_checkpoints(timestep)

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        raise NotImplementedError

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
                key: slice_block_state_dict(getattr(self, key), agent, self.num_agents)
                for key in self._checkpoint_model_keys()
            },
            "optimizers": {
                key: slice_optimizer_state(getattr(self, key).state_dict(), agent, self.num_agents)
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
        """Write ``ckpt_{step}.pt`` for every agent; return the paths."""
        paths = []
        for agent in range(self.num_agents):
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"ckpt_{step}.pt"
            torch.save(self._build_checkpoint(agent, step), path)
            paths.append(path)
        return paths

    def _write_best_checkpoints(self, step: int) -> None:
        """Rewrite ``ckpt_best.pt`` for each agent whose mean episode return improved."""
        if self.trainer_cfg.checkpoint_interval <= 0 or self.run_dirs is None:
            return
        for agent in range(self.num_agents):
            mean_return = self._mean_return[agent]
            if mean_return != mean_return or mean_return <= self._best_return[agent]:
                continue  # no finished episode yet, or no improvement
            self._best_return[agent] = mean_return
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self._build_checkpoint(agent, step), directory / "ckpt_best.pt")

    def load_agent(self, path: str | Path, slot: int, *, with_optimizer: bool = False) -> Dict[str, Any]:
        """Load one agent's checkpoint file into block slot ``slot``; return its metadata.

        Model weights, normalizer stats and learner extras always load. Optimizer state only
        loads with ``with_optimizer=True``, which needs ``num_agents == 1``: one file cannot
        fill an N-agent optimizer's block-shaped moments.
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
            assign_block_slice(getattr(self, key), slot, self.num_agents, ckpt["models"][key])

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
            if self.num_agents != 1:
                raise ValueError(
                    f"with_optimizer=True needs num_agents == 1 (this learner has "
                    f"{self.num_agents}): one agent's file cannot fill a block-shaped optimizer "
                    "state. Load without it to start that slot with fresh moments."
                )
            for key in self._checkpoint_optimizer_keys():
                getattr(self, key).load_state_dict(merge_optimizer_states([ckpt["optimizers"][key]], 1))

        return {k: ckpt[k] for k in ("step", "num_agents", "agent_idx", "mean_return")}

    @staticmethod
    def latest_checkpoint(directory: str | Path) -> Path:
        """The highest-step ``ckpt_<step>.pt`` in ``directory``."""
        candidates = glob.glob(os.path.join(str(directory), "ckpt_*.pt"))
        steps = [(int(m.group(1)), p) for p in candidates if (m := re.search(r"ckpt_(\d+)\.pt$", p))]
        if not steps:
            raise FileNotFoundError(f"no ckpt_<step>.pt files in {directory}")
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

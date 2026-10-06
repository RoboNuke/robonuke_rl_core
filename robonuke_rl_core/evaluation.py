"""Evaluation: the `eval` config section, the episode accounting, and the state writer.

The accounting rule, which everything here enforces, is **first episode per round**. A round
is a global reset with fresh random initial conditions followed by exactly
``max_episode_length`` steps, and each env contributes exactly **one** episode to the data:
its first of the round, which either hit a terminal condition or got the whole step budget.
Isaac Lab keeps auto-resetting envs mid-round; every step after an env's first done is masked
out of the counts, the returns, the metrics, the per-step state and the video.

Imports stay torch-only, so this module (and therefore ``config.py``) loads without Isaac Lab.
The runner that needs an env imports Isaac Lab lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = [
    "EvalCfg",
    "SINGLE_AGENT_OVERRIDE",
    "run_eval",
    "EvalResult",
    "force_env_reset",
    "resolved_config_path",
    "find_checkpoint",
    "checkpoint_file",
    "fetch_wandb_run",
    "upload_eval_results",
    "build_eval_policy",
    "eval_rounds",
    "per_env_channel",
    "EvalAccounting",
    "EvalStateWriter",
]


#: The config-layer override every eval and debug run injects, whatever the trained run said.
#:
#: Eval loads one agent's checkpoint into a one-agent shell and gives it every env, so the
#: trained run's ``experiment.num_agents`` is not just unused here — left in place it is
#: actively wrong twice over. The divisibility rule (``task.cfg.scene.num_envs`` must divide
#: by ``experiment.num_agents``) would reject a perfectly good eval env count for a reason
#: that does not apply, and the eval's own ``resolved_config.yaml`` would claim an agent
#: count that never ran. Injected as a CLI-layer override, so it is recorded like any other.
SINGLE_AGENT_OVERRIDE = "experiment.num_agents=1"


# --------------------------------------------------------------------------------- config
@dataclass
class EvalCfg:
    """How to evaluate a trained policy. Video defaults match the recorder this ports from.

    ``num_rollouts`` is valid episodes, not steps: the runner takes
    ``ceil(num_rollouts / num_envs)`` rounds and the last round uses only as many envs as are
    still needed. The env count is ``task.cfg.scene.num_envs``: the eval config usually sets
    it (recording wants few envs), and when it does not, the value from the run being
    evaluated is used, like every other field.
    """

    num_rollouts: int = 64
    #: act on the distribution's mean (Bernoulli dims thresholded) instead of sampling
    deterministic: bool = True
    #: capture the full per-step state (observations, actions, rewards, metrics, eval_state)
    save_state: bool = True
    #: write one mp4 per (round, env); ``--record`` forces it on
    record: bool = False
    #: registered overlay names, drawn in this order
    overlays: List[str] = field(default_factory=lambda: ["hud"])
    video_fps: int = 30
    video_height: int = 180
    video_width: int = 240

    def validate(self, cfg: Any) -> None:
        if self.num_rollouts < 1:
            raise ValueError(f"eval.num_rollouts must be >= 1, got {self.num_rollouts}")
        for name in ("video_fps", "video_height", "video_width"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"eval.{name} must be >= 1, got {value}")
        duplicates = sorted({name for name in self.overlays if self.overlays.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"eval.overlays lists {duplicates} more than once; each overlay draws once"
            )
        from .recording import OVERLAYS  # cheap: recording's heavy imports are all lazy

        unknown = [name for name in self.overlays if name not in OVERLAYS]
        if unknown:
            raise ValueError(
                f"eval.overlays names unknown overlays {unknown}; registered: "
                f"{sorted(OVERLAYS)}. Projects register their own with @register_overlay "
                "before the config is loaded."
            )


def eval_rounds(num_rollouts: int, num_envs: int) -> List[int]:
    """Envs to use in each round: full rounds of ``num_envs``, then the remainder.

    One env contributes one episode per round, so the rounds sum to exactly
    ``num_rollouts`` — no round collects episodes that would be thrown away.
    """
    if num_rollouts < 1:
        raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}")
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    rounds = math.ceil(num_rollouts / num_envs)
    return [min(num_envs, num_rollouts - index * num_envs) for index in range(rounds)]


def per_env_channel(
    infos: Any, key: str, num_envs: int
) -> Optional[Dict[str, torch.Tensor]]:
    """The validated ``infos[key]`` dict of per-env tensors, or None when absent.

    Same contract as the learner's metric channels: a dict of tensors whose first dimension
    is ``num_envs``. Unlike those, trailing dimensions are allowed — ``eval_state`` carries
    whole vectors per env, not one number. A wrong shape raises naming the key.
    """
    if not isinstance(infos, dict):
        return None
    channel = infos.get(key)
    if channel is None:
        return None
    if not isinstance(channel, dict):
        raise TypeError(f"infos['{key}'] must be a dict, got {type(channel).__name__}")
    for name, value in channel.items():
        if not torch.is_tensor(value) or value.shape[:1] != (num_envs,):
            shown = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise TypeError(
                f"infos['{key}']['{name}'] must be a tensor whose first dimension is "
                f"num_envs ({num_envs}), got {shown}"
            )
    return channel


def _is_indicator(values: torch.Tensor) -> bool:
    """True when every value is 0 or 1.

    The std of a 0/1 column is ``sqrt(p(1-p))`` -- fully determined by the mean, so printing
    it beside the mean can only restate it, while inviting the reading that a 0.89 success
    rate with a 0.31 "std" says something about spread between episodes. A success either
    happened or it did not; it is not a measurement with error. So :meth:`summary` emits a
    std only where it carries information.
    """
    return bool(((values == 0.0) | (values == 1.0)).all())


# ----------------------------------------------------------------------------- accounting
@dataclass
class Episode:
    """One valid episode: an env's first of its round."""

    round: int
    env: int
    ret: float
    length: int
    #: the flags the env published at the step that closed this episode
    terminated: bool
    truncated: bool
    #: mean over the episode's valid steps, per per-step metric name
    metrics: Dict[str, float] = field(default_factory=dict)
    #: the value the env published at the done step, per episode metric name
    episode_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        """Hit a terminal condition, as opposed to running out of time.

        ``terminated and not truncated``, because Isaac Lab tasks (Factory and Forge among
        them) raise **both** flags at the time limit: an episode that ends with ``truncated``
        set ran out of budget, whatever ``terminated`` says alongside it.
        """
        return self.terminated and not self.truncated

    @property
    def timeout(self) -> bool:
        """Ran out of step budget instead of hitting a terminal condition."""
        return not self.terminal


class EvalAccounting:
    """Per-round, per-env episode accounting under the first-episode-per-round rule.

    Fed one step at a time, on device; it syncs once per round, in :meth:`end_round`. The
    mask it returns from :meth:`step` is the same mask the state writer and the recorder must
    use, so the data, the video and the counts always agree on which steps were real.
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        device: Optional[Any] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if max_episode_length < 1:
            raise ValueError(f"max_episode_length must be >= 1, got {max_episode_length}")
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.episodes: List[Episode] = []
        self._round = -1
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._returns = torch.zeros(self.num_envs, device=self.device)
        self._lengths = torch.zeros(self.num_envs, device=self.device)
        self._terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._metric_sums: Dict[str, torch.Tensor] = {}
        self._episode_values: Dict[str, torch.Tensor] = {}
        self._steps_taken = 0
        self._open = False

    # ------------------------------------------------------------------ the round
    def start_round(self, active_envs: int) -> None:
        """Begin a round in which the first ``active_envs`` envs contribute an episode."""
        if not 1 <= active_envs <= self.num_envs:
            raise ValueError(
                f"active_envs must be in [1, {self.num_envs}], got {active_envs}"
            )
        self._round += 1
        self._active.zero_()
        self._active[:active_envs] = True
        self._finished.zero_()
        self._returns.zero_()
        self._lengths.zero_()
        self._terminated.zero_()
        self._truncated.zero_()
        self._metric_sums = {}
        self._episode_values = {}
        self._steps_taken = 0
        self._open = True

    @property
    def valid(self) -> torch.Tensor:
        """``(num_envs,)`` bool: envs whose first episode of the round is still running."""
        return self._active & ~self._finished

    def step(
        self,
        *,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        metrics: Optional[Dict[str, torch.Tensor]] = None,
        episode_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Fold one env step in; returns the ``(num_envs,)`` valid mask for that step."""
        if not self._open:
            raise RuntimeError("call start_round() before step()")
        self._steps_taken += 1
        if self._steps_taken > self.max_episode_length:
            raise RuntimeError(
                f"round {self._round} stepped {self._steps_taken} times, past the "
                f"max_episode_length budget of {self.max_episode_length}"
            )
        valid = self.valid
        mask = valid.to(self._returns.dtype)

        self._returns += rewards.reshape(-1) * mask
        self._lengths += mask

        for name, value in (metrics or {}).items():
            flat = value.reshape(-1).to(self._returns.dtype)
            if name not in self._metric_sums:
                self._metric_sums[name] = torch.zeros_like(self._returns)
            self._metric_sums[name] += flat * mask

        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1))
        closing = valid & done
        # the episode's own outcome, recorded before the env auto-resets under us
        self._terminated |= closing & terminated.reshape(-1)
        self._truncated |= closing & truncated.reshape(-1)
        for name, value in (episode_metrics or {}).items():
            flat = value.reshape(-1).to(self._returns.dtype)
            if name not in self._episode_values:
                self._episode_values[name] = torch.full_like(self._returns, float("nan"))
            self._episode_values[name] = torch.where(
                closing, flat, self._episode_values[name]
            )
        self._finished |= closing
        return valid

    def end_round(self) -> None:
        """Close the round: envs still running used the whole budget. One host sync."""
        if not self._open:
            raise RuntimeError("call start_round() before end_round()")
        active = self._active.tolist()
        finished = self._finished.tolist()
        returns = self._returns.tolist()
        lengths = self._lengths.tolist()
        terminated = self._terminated.tolist()
        truncated = self._truncated.tolist()
        metric_sums = {name: value.tolist() for name, value in self._metric_sums.items()}
        episode_values = {name: value.tolist() for name, value in self._episode_values.items()}

        for env in range(self.num_envs):
            if not active[env]:
                continue
            length = int(lengths[env])
            if not finished[env] and length != self.max_episode_length:
                raise RuntimeError(
                    f"round {self._round} env {env} ran {length} valid steps but never "
                    f"signalled done; an env that never finishes must have been stepped the "
                    f"whole max_episode_length budget ({self.max_episode_length}) to count as "
                    f"a clean timeout, and this round stopped after {self._steps_taken} steps"
                )
            self.episodes.append(
                Episode(
                    round=self._round,
                    env=env,
                    ret=returns[env],
                    length=length,
                    terminated=bool(terminated[env]),
                    truncated=bool(truncated[env]),
                    metrics={
                        name: sums[env] / length for name, sums in metric_sums.items()
                    }
                    if length
                    else {},
                    episode_metrics={
                        name: values[env]
                        for name, values in episode_values.items()
                        if not math.isnan(values[env])
                    },
                )
            )
        self._open = False

    # ------------------------------------------------------------------ the result
    def summary(self) -> Dict[str, Any]:
        """Aggregate every valid episode. A ``success`` episode metric is the success rate."""
        episodes = self.episodes
        if not episodes:
            return {"episodes": 0, "rounds": self._round + 1}

        returns = torch.tensor([e.ret for e in episodes], dtype=torch.float64)
        lengths = torch.tensor([float(e.length) for e in episodes], dtype=torch.float64)
        out: Dict[str, Any] = {
            "episodes": len(episodes),
            "rounds": self._round + 1,
            "terminal": sum(1 for e in episodes if e.terminal),
            "timeout": sum(1 for e in episodes if e.timeout),
            "return/mean": float(returns.mean()),
            "return/std": float(returns.std(unbiased=False)),
            "length/mean": float(lengths.mean()),
            "length/std": float(lengths.std(unbiased=False)),
        }
        for attribute in ("metrics", "episode_metrics"):
            names = sorted({name for e in episodes for name in getattr(e, attribute)})
            for name in names:
                values = [
                    getattr(e, attribute)[name]
                    for e in episodes
                    if name in getattr(e, attribute)
                ]
                tensor = torch.tensor(values, dtype=torch.float64)
                out[f"{name}/mean"] = float(tensor.mean())
                if not _is_indicator(tensor):
                    out[f"{name}/std"] = float(tensor.std(unbiased=False))
                if len(values) != len(episodes):
                    # an episode closed by the step budget never published an episode metric
                    out[f"{name}/episodes"] = len(values)
        return out

    def episode_table(self) -> List[Dict[str, Any]]:
        """Every valid episode as a flat dict, for ``summary.yaml`` and offline analysis."""
        return [
            {
                "round": e.round,
                "env": e.env,
                "return": e.ret,
                "length": e.length,
                "outcome": "terminal" if e.terminal else "timeout",
                "terminated": int(e.terminated),
                "truncated": int(e.truncated),
                **{f"metric/{k}": v for k, v in e.metrics.items()},
                **{f"episode_metric/{k}": v for k, v in e.episode_metrics.items()},
            }
            for e in self.episodes
        ]


# --------------------------------------------------------------------------- state writer
class EvalStateWriter:
    """The per-step full-state capture, written as one parquet table.

    Ported from the step-trace recorder in RoboNuke/generalized_hybrid_vic_action_space: one
    row per (round, env, step), with vector signals expanded into ``name_0 .. name_k`` columns
    and a NaN back-fill for a column a round never produced, so the table stays rectangular
    and any aggregate can be re-derived offline. Two differences: rows carry a ``round``, and
    trailing dimensions are expanded in full rather than mean-collapsed to one number — a
    parquet column per element loses nothing.

    Only valid steps become rows: an env's first episode of the round and nothing after it.
    Tensors land on the host as they arrive — eval is not a throughput path, and this keeps
    the capture off the GPU's memory budget.
    """

    #: flags that read better as small ints than as floats
    INT_COLUMNS = ("terminated", "truncated")

    def __init__(self, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = int(num_envs)
        #: one column dict per finished round, each already masked to its valid rows
        self._rounds: List[Dict[str, "np.ndarray"]] = []
        self._round: Optional[int] = None
        self._written: List[int] = []
        self._steps: List[Dict[str, torch.Tensor]] = []
        self._valid: List[torch.Tensor] = []

    def start_round(self, round_index: int) -> None:
        if self._round is not None:
            raise RuntimeError(f"round {self._round} is still open; call end_round() first")
        if round_index in self._written:
            raise ValueError(f"round {round_index} was already written")
        self._round = int(round_index)
        self._steps = []
        self._valid = []

    def capture(self, valid: torch.Tensor, **values: Any) -> None:
        """Record one step. ``valid`` is the accountant's mask for that same step."""
        if self._round is None:
            raise RuntimeError("call start_round() before capture()")
        row: Dict[str, torch.Tensor] = {}
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, dict):  # a channel: flatten its keys into the row
                for key, item in value.items():
                    row[f"{name}/{key}"] = _host(item, self.num_envs, f"{name}/{key}")
                continue
            row[name] = _host(value, self.num_envs, name)
        self._steps.append(row)
        self._valid.append(valid.reshape(-1).detach().to("cpu", torch.bool))

    def end_round(self) -> None:
        """Turn the round's steps into rows, keeping only each env's valid steps."""
        import numpy as np

        if self._round is None:
            raise RuntimeError("call start_round() before end_round()")
        columns: Dict[str, "np.ndarray"] = {}
        if self._steps:
            names = sorted({name for row in self._steps for name in row})
            missing = [name for name in names if any(name not in row for row in self._steps)]
            if missing:
                raise ValueError(
                    f"round {self._round}: {missing} were captured on some steps but not "
                    "others; a channel must be present every step or never"
                )
            valid = torch.stack(self._valid)  # (steps, num_envs)
            # row order follows the recorder this ports from: step-major, env fastest
            step_index, env_index = valid.nonzero(as_tuple=True)
            columns["round"] = np.full(step_index.numel(), self._round, dtype=np.int32)
            columns["env"] = env_index.numpy().astype(np.int32)
            columns["step"] = step_index.numpy().astype(np.int32)
            for name in names:
                stacked = torch.stack([row[name] for row in self._steps])
                flat = stacked.reshape(stacked.shape[0], self.num_envs, -1)
                kept = flat[step_index, env_index]  # (rows, width)
                array = kept.to(torch.float32).numpy()
                if name in self.INT_COLUMNS:
                    array = np.nan_to_num(array).astype(np.uint8)
                if array.shape[1] == 1:
                    columns[name] = array[:, 0]
                else:
                    for index in range(array.shape[1]):
                        columns[f"{name}_{index}"] = array[:, index]
        self._rounds.append(columns)
        self._written.append(self._round)
        self._round = None
        self._steps = []
        self._valid = []

    # ------------------------------------------------------------------ the output
    def table(self) -> Dict[str, "np.ndarray"]:
        """Every round's rows as one rectangular set of columns, NaN where a round had none."""
        import numpy as np

        if self._round is not None:
            raise RuntimeError(f"round {self._round} is still open; call end_round() first")
        index = ["round", "env", "step"]
        names = sorted({name for columns in self._rounds for name in columns} - set(index))
        lengths = [len(columns.get("round", ())) for columns in self._rounds]
        out: Dict[str, "np.ndarray"] = {}
        for key in index:
            out[key] = np.concatenate(
                [columns[key] for columns, size in zip(self._rounds, lengths) if size]
                or [np.zeros(0, dtype=np.int32)]
            )
        for name in names:
            pieces = []
            for columns, size in zip(self._rounds, lengths):
                if not size:
                    continue
                if name in columns:
                    pieces.append(columns[name])
                else:  # this round never produced the column: rectangular NaN back-fill
                    pieces.append(np.full(size, np.nan, dtype=np.float32))
            out[name] = np.concatenate(pieces or [np.zeros(0, dtype=np.float32)])
        return out

    @property
    def rows(self) -> int:
        """Rows the table holds: one per valid (round, env, step)."""
        return sum(len(columns.get("round", ())) for columns in self._rounds)

    def write_parquet(self, path: Any) -> "Path":
        """Write the table to ``path``. Raises if the parquet engine is missing."""
        from pathlib import Path

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas ships with the env
            raise RuntimeError(
                "writing the eval trace needs pandas (and pyarrow) installed"
            ) from exc

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.table())
        try:
            frame.to_parquet(out, index=False)
        except Exception as exc:
            raise RuntimeError(
                f"failed to write the eval trace to {out} ({exc!r}); is pyarrow installed?"
            ) from exc
        return out


def _host(value: Any, num_envs: int, name: str) -> torch.Tensor:
    """One per-env tensor on the host, shape-checked."""
    if not torch.is_tensor(value):
        raise TypeError(f"eval state '{name}' must be a tensor, got {type(value).__name__}")
    if value.shape[:1] != (num_envs,):
        raise ValueError(
            f"eval state '{name}' must have first dimension num_envs ({num_envs}), got "
            f"{tuple(value.shape)}"
        )
    return value.detach().to("cpu")


# --------------------------------------------------------------------------------- the run
def force_env_reset(env: Any) -> tuple:
    """Genuinely reset **every** env and return the fresh ``(observations, states)``.

    Ported from the eval collector in RoboNuke/generalized_hybrid_vic_action_space: skrl's
    ``IsaacLabWrapper.reset()`` is guarded by a ``_reset_once`` flag, so only the FIRST call
    resets the sim and every later one silently returns the cached observation. Its trainers
    never need a second reset because Isaac Lab auto-resets in-step, but a round-based eval
    does: without this, round 2 would start wherever each env's own auto-reset clock happened
    to leave it, and the round's "fresh random initial conditions" would be a fiction.

    Afterwards ``episode_length_buf`` must be zero, which a genuine reset guarantees. A
    non-zero clock means the reset was swallowed again, so this raises rather than quietly
    collecting mid-rollout fragments.
    """
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if hasattr(node, "_reset_once"):
            node._reset_once = True
        node = getattr(node, "_env", None) or getattr(node, "env", None)

    observations, _ = env.reset()
    buffer = getattr(getattr(env, "unwrapped", env), "episode_length_buf", None)
    if buffer is not None and int(buffer.max()) != 0:
        raise RuntimeError(
            f"env.reset() did not reset the envs: episode_length_buf.max()={int(buffer.max())}, "
            "expected 0 right after a full reset. skrl's _reset_once guard was not cleared, so "
            "every round after the first would start mid-rollout. Check the wrapper stack for a "
            "reset() that swallows the call."
        )
    return observations, env_states(env)


def env_states(env: Any) -> Optional[torch.Tensor]:
    """The critic state for an asymmetric env, or None when the env is symmetric."""
    getter = getattr(env, "state", None)
    if getter is None:
        return None
    return getter()


def run_eval(
    *,
    env: Any,
    policy: Any,
    num_rollouts: int,
    max_episode_length: int,
    seed: int,
    save_state: bool = True,
    recorder: Any = None,
    device: Optional[Any] = None,
) -> "EvalResult":
    """Collect ``num_rollouts`` valid episodes in rounds, under the first-episode rule.

    One round = one global reset (fresh random initial conditions, seeded
    ``seed + round``) plus exactly ``max_episode_length`` steps. ``policy`` is called as
    ``policy(observations, states) -> actions``; nothing here updates a normalizer or a
    weight. ``recorder``, when given, follows the same ``start_round`` / ``capture`` /
    ``end_round`` protocol as the state writer and sees the same valid mask.
    """
    num_envs = int(env.num_envs)
    accounting = EvalAccounting(num_envs, max_episode_length, device)
    writer = EvalStateWriter(num_envs) if save_state else None

    for index, active in enumerate(eval_rounds(num_rollouts, num_envs)):
        torch.manual_seed(int(seed) + index)
        observations, states = force_env_reset(env)
        accounting.start_round(active)
        if writer is not None:
            writer.start_round(index)
        if recorder is not None:
            recorder.start_round(index)

        for _ in range(max_episode_length):
            if not bool(accounting.valid.any()):
                # every active env has closed its first episode; the rest of the budget would
                # only step episodes that are masked out anyway. Isaac Lab tasks time out at
                # max_episode_length - 1, so this is the normal way a round ends.
                break
            actions = policy(observations, states)
            next_observations, rewards, terminated, truncated, infos = env.step(actions)
            metrics = per_env_channel(infos, "metrics_to_log", num_envs)
            episode_metrics = per_env_channel(infos, "episode_metrics_to_log", num_envs)
            valid = accounting.step(
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                metrics=metrics,
                episode_metrics=episode_metrics,
            )
            if writer is not None:
                writer.capture(
                    valid,
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    metrics=metrics,
                    episode_metrics=episode_metrics,
                    eval_state=per_env_channel(infos, "eval_state", num_envs),
                )
            if recorder is not None:
                recorder.capture(
                    valid,
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                    metrics=metrics,
                )
            observations = next_observations
            states = env_states(env)

        accounting.end_round()
        if writer is not None:
            writer.end_round()
        if recorder is not None:
            recorder.end_round()

    return EvalResult(accounting=accounting, state=writer)


@dataclass
class EvalResult:
    """What one eval produced: the accounting and, when enabled, the per-step state."""

    accounting: EvalAccounting
    state: Optional[EvalStateWriter] = None

    @property
    def summary(self) -> Dict[str, Any]:
        return self.accounting.summary()

    @property
    def episodes(self) -> List[Dict[str, Any]]:
        return self.accounting.episode_table()


# ------------------------------------------------------------------- loading a trained run
def resolved_config_path(run_dir: Any) -> "Path":
    """The ``resolved_config.yaml`` for a run dir.

    A training run writes one config for the whole group (every agent shares it) at the group
    directory, while checkpoints live in each agent's own run directory beneath it. So an
    agent's run dir is the natural thing to point ``--from_run`` at, and the config is found
    either there or one level up.
    """
    from pathlib import Path

    from .config import RESOLVED_NAME

    directory = Path(run_dir).expanduser()
    for candidate in (directory / RESOLVED_NAME, directory.parent / RESOLVED_NAME):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {RESOLVED_NAME} for run {directory}: looked in the run dir and its group dir "
        f"({directory / RESOLVED_NAME}, {directory.parent / RESOLVED_NAME})"
    )


def find_checkpoint(run_dir: Any, which: str = "best") -> "Path":
    """Resolve ``best``, a step number, a file name, or a path to one checkpoint file."""
    from pathlib import Path

    from .learners.base import CHECKPOINT_BEST, checkpoint_name

    direct = Path(str(which)).expanduser()
    if direct.is_file():
        return direct
    directory = Path(run_dir).expanduser() / "checkpoints"
    path = directory / checkpoint_file(which)
    if path.is_file():
        return path
    available = sorted(p.name for p in directory.glob("ckpt_*")) if directory.is_dir() else []
    raise FileNotFoundError(
        f"checkpoint {which!r} not found: looked for {path}"
        + (f"; {directory} holds {available}" if available else f"; {directory} holds nothing")
    )


def checkpoint_file(which: str = "best") -> str:
    """The checkpoint file name a ``--checkpoint`` value means.

    ``best`` is the best-return checkpoint, a bare number is that step's checkpoint, and
    anything else is taken as the file name itself, so an exact name always works.
    """
    from .learners.base import CHECKPOINT_BEST, checkpoint_name

    text = str(which)
    if text == "best":
        return CHECKPOINT_BEST
    if text.isdigit():
        return checkpoint_name(int(text))
    return text


# ------------------------------------------------------------------------------ from wandb
def fetch_wandb_run(
    spec: str, which: str = "best", cache_root: Any = None, *, api: Any = None
) -> "Path":
    """Download one wandb run's config and checkpoint into a local run directory.

    ``spec`` is ``entity/project/run`` where ``run`` is the run id or its exact display name.
    Only ordinary run **files** are used — never the Artifacts API — so this is the mirror of
    what the training logger uploads. The result is a directory shaped like a local run
    (``resolved_config.yaml`` beside ``checkpoints/<name>``), which the local code path then
    takes over.
    """
    from pathlib import Path

    from .config import RESOLVED_NAME

    parts = [piece for piece in str(spec).split("/") if piece]
    if len(parts) != 3:
        raise ValueError(
            f"--run must be 'entity/project/run' (run id or exact run name), got {spec!r}"
        )
    entity, project, wanted = parts

    if api is None:
        import wandb  # lazy: the package imports without wandb installed

        api = wandb.Api()
    run = _wandb_run(api, entity, project, wanted)

    name = checkpoint_file(which)
    cache = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache" / "robonuke_rl_core"
    directory = cache / entity / project / str(run.id)
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)

    _download(run, RESOLVED_NAME, directory, spec)
    _download(run, name, directory / "checkpoints", spec)
    return directory


def _wandb_run(api: Any, entity: str, project: str, wanted: str) -> Any:
    """The run with this id, or failing that the one run with this exact display name."""
    try:
        return api.run(f"{entity}/{project}/{wanted}")
    except Exception as by_id:
        matches = list(api.runs(f"{entity}/{project}", {"display_name": wanted}))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise LookupError(
                f"no wandb run {wanted!r} in {entity}/{project}: not a run id "
                f"({by_id!r}) and no run has that name"
            ) from by_id
        raise LookupError(
            f"{len(matches)} runs in {entity}/{project} are named {wanted!r} "
            f"({[m.id for m in matches]}); pass the run id instead"
        ) from by_id


def _download(run: Any, name: str, directory: Any, spec: str) -> "Path":
    """One run file into ``directory``, replacing what is there."""
    from pathlib import Path

    try:
        handle = run.file(name)
    except Exception as exc:
        raise FileNotFoundError(f"wandb run {spec} has no file {name!r}: {exc!r}") from exc
    if getattr(handle, "size", 1) == 0:  # wandb hands back an empty stub for a missing file
        raise FileNotFoundError(
            f"wandb run {spec} has no file {name!r}; the run's Files tab must hold it "
            "(training uploads the config and each checkpoint as plain run files)"
        )
    handle.download(root=str(directory), replace=True)
    path = Path(directory) / name
    if not path.is_file():
        raise FileNotFoundError(f"wandb reported {name!r} downloaded but {path} is not there")
    return path


def upload_eval_results(
    spec: str, out_dir: Any, root: Any, *, api: Any = None
) -> List[str]:
    """Attach an eval's output files to the ORIGINAL training run, as plain run files.

    Stored under the path ``out_dir`` has relative to ``root`` — which is
    ``eval/<eval-config-stem>_<timestamp>/...``, so repeated evals never overwrite each
    other. Nothing is logged as a wandb metric: the per-step trace is the data, and any
    aggregate can be re-derived from it offline.

    Returns the names the files were stored under.
    """
    from pathlib import Path

    parts = [piece for piece in str(spec).split("/") if piece]
    if len(parts) != 3:
        raise ValueError(f"expected 'entity/project/run', got {spec!r}")
    entity, project, wanted = parts
    if api is None:
        import wandb  # lazy

        api = wandb.Api()
    run = _wandb_run(api, entity, project, wanted)

    out_dir, root = Path(out_dir), Path(root)
    stored: List[str] = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        name = str(path.relative_to(root))
        run.upload_file(str(path), root=str(root))
        stored.append(name)
    if not stored:
        raise FileNotFoundError(f"nothing to upload: {out_dir} holds no files")
    return stored


def build_eval_policy(
    cfg: Any, env: Any, checkpoint: Any, *, deterministic: bool = True
) -> tuple:
    """A frozen single-agent policy from a checkpoint: ``(policy, learner, metadata)``.

    Eval always runs **one** agent, whatever the training run's ``experiment.num_agents`` was:
    the checkpoint holds one agent's slot and every env belongs to it. Normalizers come from
    the checkpoint and never train again — the policy call passes no ``train=True`` anywhere —
    and no optimizer state is loaded.
    """
    from .learners.cfg import LEARNERS
    from .learners.ppo import PPO
    from .learners.sac import SAC
    from .models.factory import build_models

    classes = {"sac": SAC, "ppo": PPO}
    learner_name = cfg.trainer.learner
    if learner_name not in classes:
        raise ValueError(f"cannot evaluate learner {learner_name!r}; known: {sorted(LEARNERS)}")

    state_space = env.state_space
    controller_cfg = getattr(cfg, "controller", None)
    models = build_models(
        learner_name,
        cfg.model,
        env.observation_space,
        state_space,
        env.action_space,
        1,  # one agent
        env.device,
        controller_cfg,
    )
    extra = (
        {"model_cfg": cfg.model, "controller_cfg": controller_cfg}
        if learner_name == "sac"
        else {}
    )
    learner = classes[learner_name](
        models=models,
        memory=None,  # eval stores nothing in a replay buffer
        observation_space=env.observation_space,
        state_space=state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=cfg[learner_name],
        trainer_cfg=cfg.trainer,
        num_agents=1,
        num_envs=env.num_envs,
        **extra,
    )
    metadata = learner.load_agent(checkpoint, slot=0, with_optimizer=False)
    learner.enable_training_mode(False)  # eval mode: no dropout, no stat updates

    def policy(observations: torch.Tensor, states: Optional[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            inputs = {"observations": learner.normalize_observations(observations)}
            actions, outputs = learner.policy.act(inputs, role="policy")
            return outputs["mean_actions"] if deterministic else actions

    return policy, learner, metadata

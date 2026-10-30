"""SAC over ``num_agents`` independent agents.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/sac.py``, minus AMP,
the distributed branches, the auxiliary-loss manager, the contact/rotation supervision and
the per-axis action metrics. Changed for per-agent independence: gradients are clipped per
agent and the observation/state normalizers are per agent (``BlockRunningNorm``).

Beyond classic SAC, all off by default:

* **A Q ensemble** (``num_critics`` > 2): the bootstrap target is the min over a random
  ``critic_subset`` of the target critics, drawn per agent (REDQ, Chen et al. 2021); the actor
  reduces the critics' Q by ``actor_q_reduction``. The critics (twin included) are ONE module,
  ``critics``, stacked on a member axis inside each agent and evaluated in one vmapped call;
  ``target_critics`` likewise.
* **A delayed actor** (``policy_delay``): the actor and temperature step once every so many
  gradient steps, so ``gradient_steps`` raises the critic's update-to-data ratio alone.
* **A sparser update cadence** (``update_every``): a round of ``gradient_steps`` every so many
  env steps, for an update-to-data ratio below one gradient step per env step.
* **A critic-only start** (``policy_learning_starts``).
* **Demo transitions in the critic's batches** (``demo_transitions`` with the
  ``demo_sampling_*`` window): half demo, half online, RLPD's symmetric sampling — for the
  critic only; the actor's batch is always online.
* **CUDA graphs** (``cuda_graph``): each kind of gradient step is captured once and replayed
  as one launch (:class:`~robonuke_rl_core.learners.cuda_graph.StepGraphs`).
* **n-step TD targets** (``n_step``, :mod:`~robonuke_rl_core.learners.n_step`): the memory
  stores n-step rows (summed rewards, the bootstrap state ``m`` steps on, ``discounts``
  ``gamma ** m``) and the demo rows are folded the same way when loaded.
* **FlashSAC's pieces** (arXiv:2604.04539, ported from github.com/Holiday-Robot/FlashSAC
  ``agents/flashSAC``; never its mixed precision): a categorical critic
  (``critic_distribution``) with adaptive reward scaling (``adaptive_reward_scaling``,
  :class:`~robonuke_rl_core.models.normalizer.BlockReturnScaler`), weight normalization
  (``weight_normalization``), the unified entropy target (``target_entropy_sigma``),
  cross-batch value prediction (``cross_batch_value``) and noise repetition
  (``noise_repeat``). FlashSAC's own network (BatchNorm, RMSNorm) is not part of it.

Every tensor op either keeps the rows separate (the block networks reduce within a block) or
reduces with a constant factor, so agent ``i``'s update never depends on agent ``j``'s data.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F

from ..optim import BlockAdamW, clip_grad_norm_per_agent, lr_at
from ..models.factory import build_models
from ..models.normalizer import BlockReturnScaler, BlockRunningNorm
from ..losses.losses import LossContext
from .base import LearnerBase
from .cfg import sac_critic_outputs
from .cuda_graph import StepGraphs
from .n_step import NStepWindow, n_step_rows

__all__ = ["SAC", "register_critic_action", "critic_action_transform", "critic_action_space", "categorical_projection"]

#: ``fn(cfg, env) -> transform | None``, registered by a project in its training script's
#: ``setup()``. A transform ``(critic_inputs, actions) -> actions`` gives the critics another action
#: input than the agent's own (a residual agent's executed action, say); ``critic_inputs`` is the
#: batch's RAW critic input, before any normalizer: the policy observations when the learner is
#: symmetric, the critic ``states`` (``next_states`` for the target) when a state space exists.
#: At most one builder may answer. Two optional attributes on the transform:
#:
#: * ``critic_action_width``: the critics' action is that wide instead of the agent's (they are
#:   built with that action width: :func:`critic_action_space`);
#: * ``side_inputs``: ``{name: width}`` per-transition tensors the transform reads and the critics
#:   never see as input (a residual's base action, say). The env publishes ``infos[name]`` (this
#:   step's, ``(num_envs, width)``) and ``infos["next_" + name]`` (the next observation's) every
#:   step; the memory stores both and the transform is called ``(critic_inputs, actions, side)``,
#:   ``side`` the ``{name: tensor}`` of the same rows (``next_`` ones for the target). Not with
#:   demo transitions, which carry none.
CRITIC_ACTION_BUILDERS: List[Any] = []


def register_critic_action(fn: Any) -> None:
    """Add a :data:`CRITIC_ACTION_BUILDERS` entry; ``scripts/train.py`` asks each once its env is built."""
    CRITIC_ACTION_BUILDERS.append(fn)


def critic_action_space(transform: Any, action_space: Any) -> Any:
    """The space the SAC critics are built with for their action input: the agent's
    ``action_space``, or a ``critic_action_width``-wide Box when the transform declares one."""
    width = getattr(transform, "critic_action_width", None)
    if width is None:
        return action_space
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(int(width),), dtype=np.float32)


def critic_action_transform(cfg: Any, env: Any) -> Any:
    """The one transform a registered builder returns for this run, or None (the critics take the
    agent's action). Two builders answering is a configuration error, not a choice to guess."""
    found = [t for t in (fn(cfg, env) for fn in CRITIC_ACTION_BUILDERS) if t is not None]
    if len(found) > 1:
        raise ValueError(f"{len(found)} registered critic-action builders answered for this run; at most one may")
    return found[0] if found else None


def categorical_projection(probs: torch.Tensor, atoms: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """C51's projection, FlashSAC's ``_compute_categorical_td_target``: the mass ``probs``
    ``(rows, bins)`` sits at ``atoms`` ``(rows, bins)`` (the target's shifted atoms); clamp them
    to the support and split each atom's mass between the two support atoms around it, in
    proportion to the distance. Returns ``(rows, bins)`` probabilities on the support."""
    bins = probs.shape[-1]
    width = (max_value - min_value) / (bins - 1)
    position = (atoms.clamp(min_value, max_value) - min_value) / width
    lower = torch.floor(position).long()
    upper = torch.clamp(lower + 1, 0, bins - 1)
    fraction = position - lower.to(position.dtype)
    projected = torch.zeros_like(probs)
    projected.scatter_add_(1, lower, probs * (1.0 - fraction))
    projected.scatter_add_(1, upper, probs * fraction)
    return projected


class SAC(LearnerBase):
    def __init__(self, *, model_cfg=None, controller_cfg=None, critic_action_space=None, **kwargs) -> None:
        """``model_cfg`` and ``controller_cfg`` are only needed when
        ``sac.periodic_reset_enabled`` rebuilds models — a rebuild has to reproduce the same
        actor, including the derived MATCH pairs."""
        super().__init__(**kwargs)
        self._model_cfg = model_cfg
        self._controller_cfg = controller_cfg
        #: what the critics were built with as their action input (:func:`critic_action_space`); None:
        #: the agent's action space. A periodic reset rebuilds them the same way
        self._critic_action_space = critic_action_space
        #: ``(raw critic inputs, actions) -> critic actions`` (:func:`critic_action_transform`): the
        #: observations, or the states when asymmetric; or None: the critics take the agent's
        #: action as it is
        self.critic_action_transform: Optional[Any] = None

        self._num_critics = int(self.cfg.num_critics)
        required = ("policy", "critics", "target_critics")
        missing = [k for k in required if self.models.get(k) is None]
        if missing:
            raise ValueError(f"SAC requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self._bind_critics()
        for key in ("critics", "target_critics"):
            members = getattr(self.models[key], "members", None)
            if members != self._num_critics:
                raise ValueError(
                    f"SAC's '{key}' stacks {members} critics but sac.num_critics is "
                    f"{self._num_critics}; build the models with the same num_critics"
                )
        expected = sac_critic_outputs(self.cfg)
        for key in ("critics", "target_critics"):
            outputs = getattr(self.models[key], "outputs", 1)
            if outputs != expected:
                raise ValueError(
                    f"SAC's '{key}' emit {outputs} columns but sac.critic_distribution "
                    f"{self.cfg.critic_distribution!r} needs {expected}; build the models with "
                    "critic_outputs=sac_critic_outputs(sac)"
                )
        self._categorical = self.cfg.critic_distribution == "categorical"
        #: the categorical critics' support, (bins,)
        self._atoms = (
            torch.linspace(
                float(self.cfg.critic_min_value), float(self.cfg.critic_max_value),
                int(self.cfg.critic_num_bins), device=self.device,
            )
            if self._categorical
            else None
        )
        cuda = torch.device(self.device).type == "cuda"
        self._graphs = StepGraphs() if (self.cfg.cuda_graph and cuda) else None

        self._asymmetric = self.state_space is not None
        self._n_periodic_resets = 0

        # per-agent entropy coefficient: (N, 1), and Adam's state is elementwise, so one
        # optimizer over it stays independent across agents
        self._entropy_coefficient = torch.full(
            (self.num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            self._target_entropy = self._resolve_target_entropy()
            self.log_entropy_coefficient = torch.log(
                self._entropy_coefficient.clone()
            ).requires_grad_(True)
            # no weight decay on log_alpha: pulling it toward 0 has no meaning. Its leading
            # dim is the agent, so one BlockAdamW is N independent temperature optimizers.
            self.entropy_optimizer = BlockAdamW(
                [self.log_entropy_coefficient],
                self.num_agents,
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

        self._build_optimizers()
        self._normalize_weights(self.policy, self.critics)
        self._sync_targets()

        #: adaptive_reward_scaling: the per-agent return statistics the critics' rewards are
        #: divided by (updated in record_transition from every stored step); None when off
        self.return_scaler: Optional[BlockReturnScaler] = (
            BlockReturnScaler(
                self.num_agents,
                gamma=float(self.cfg.discount_factor),
                g_max=float(self.cfg.adaptive_reward_g_max),
                device=self.device,
            )
            if self.cfg.adaptive_reward_scaling
            else None
        )
        #: noise_repeat: the held standard-normal draw per env, and per agent the steps it has
        #: been held and the length drawn for it (count 0: draw at the next step)
        self._held_noise: Optional[torch.Tensor] = None
        self._noise_count = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
        self._noise_length = torch.ones(self.num_agents, dtype=torch.long, device=self.device)
        if self.cfg.noise_repeat:
            # FlashSAC's _build_truncated_zeta_cdf: P(k) ~ k^-s on 1..max
            lengths = torch.arange(1, int(self.cfg.noise_repeat_max) + 1, dtype=torch.float32, device=self.device)
            pmf = lengths ** (-float(self.cfg.noise_repeat_zeta_s))
            self._zeta_cdf = torch.cumsum(pmf / pmf.sum(), dim=0)

        if self.cfg.normalize_observations:
            self.observation_normalizer = self.make_normalizer(self.observation_space)
            self.state_normalizer = (
                self.make_normalizer(self.state_space) if self._asymmetric else None
            )
        else:
            self.observation_normalizer = None
            self.state_normalizer = None

        # loaded with the replay memory (``_create_memory_tensors``): eval builds this learner
        # with no memory and never needs the demos
        self._demo: Optional[Dict[str, torch.Tensor]] = None
        #: n_step > 1: the online stream's n-step folding (a new one with every memory)
        self._n_step_window: Optional[NStepWindow] = None

    # ------------------------------------------------------------------ setup helpers
    def _bind_critics(self) -> None:
        self.critics = self.models["critics"]
        self.target_critics = self.models["target_critics"]

    @staticmethod
    @contextlib.contextmanager
    def _batch_mode(model, training: bool):
        """Run ``model`` in ``training`` mode for one call, then restore its mode: FlashSAC's per-call
        choice for a network with BatchNorm (``uses_batch_stats``, ``models/flashsac.py``) -- the actor
        and the critics in training mode on their own loss batches (the target critics too), the actor
        in eval mode for the bootstrap's next actions, the critics in eval mode for the actor's Q. A
        network without batch statistics is left as it is."""
        if not getattr(model, "uses_batch_stats", False):
            yield
            return
        previous = model.training
        model.train(training)
        try:
            yield
        finally:
            model.train(previous)

    def _normalize_weights(self, *models) -> None:
        """``weight_normalization``: project each model's weights (FlashSAC's
        ``normalize_parameters``); nothing when off."""
        if not self.cfg.weight_normalization:
            return
        for model in models:
            net = getattr(model, "net", None)
            if net is None or not hasattr(net, "normalize_weights"):
                raise TypeError(
                    f"sac.weight_normalization needs models holding a VmapEnsemble (.net); "
                    f"{type(model).__name__} does not"
                )
            net.normalize_weights()

    def _sync_targets(self) -> None:
        self.target_critics.freeze_parameters(True)
        self.target_critics.update_parameters(self.critics, polyak=1)

    def _polyak(self) -> None:
        """``target <- target + polyak * (critic - target)`` over every tensor in one launch."""
        with torch.no_grad():
            torch._foreach_lerp_(
                list(self.target_critics.parameters()),
                list(self.critics.parameters()),
                float(self.cfg.polyak),
            )

    def _policy_learning_starts(self) -> int:
        if self.cfg.policy_learning_starts is None:
            return int(self.cfg.learning_starts)
        return int(self.cfg.policy_learning_starts)

    def _demo_names(self) -> List[str]:
        names = ["observations", "actions", "rewards", "next_observations", "terminated", "truncated"]
        if self._asymmetric:
            names += ["states", "next_states"]
        # the critic action transform's side inputs: a demo file must carry them as well (and their next_ values)
        for name in sorted(getattr(self, "_side_inputs", {}) or {}):
            names += [name, f"next_{name}"]
        return names

    def _n_step(self) -> int:
        return int(self.cfg.n_step)

    def _own_potential_terms(self) -> Dict[str, float]:
        """``potential_terms``' entry for this SAC (its route's, or "*"), ``{channel: weight}``; empty if unshaped."""
        route = self.cfg.route_label or "*"
        return {str(t.channel): float(t.weight) for t in (self.cfg.potential_terms or []) if t.route == route}

    def _load_demo_transitions(self) -> Dict[str, torch.Tensor]:
        """The demo file's tensors on the device, checked against the agent's spaces."""
        path = self.cfg.demo_transitions
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"sac.demo_transitions: no file {path}") from None
        if not isinstance(data, dict) or "transitions" not in data:
            raise ValueError(
                f"sac.demo_transitions ({path}) must be a dict with a 'transitions' dict of "
                "tensors (and optional 'meta')"
            )
        transitions = data["transitions"]
        meta = data.get("meta") or {}
        own = self._own_potential_terms()
        if own or meta.get("potential_terms"):
            # the demo rewards must carry the online rewards' shaping
            made = {str(c): float(w) for c, w in (meta.get("potential_terms") or {}).items()}
            if made != own or float(meta.get("discount_factor", float("nan"))) != float(self.cfg.discount_factor):
                raise ValueError(
                    f"sac.demo_transitions ({path}) was shaped with potential_terms {made} and discount_factor "
                    f"{meta.get('discount_factor')}, this learner with {own} and {self.cfg.discount_factor}: remake it"
                )
        names = self._demo_names()
        missing = [n for n in names if n not in transitions]
        if missing:
            raise KeyError(f"sac.demo_transitions ({path}) is missing {missing}; needs {names}")
        widths = {
            "observations": self._space_width(self.observation_space),
            "next_observations": self._space_width(self.observation_space),
            "actions": self._space_width(self.action_space),
            "rewards": 1,
            "terminated": 1,
            "truncated": 1,
        }
        if self._asymmetric:
            widths["states"] = widths["next_states"] = self._space_width(self.state_space)
        for name, width in (getattr(self, "_side_inputs", {}) or {}).items():
            widths[name] = widths[f"next_{name}"] = int(width)
        rows = None
        out = {}
        for name in names:
            tensor = transitions[name]
            if tensor.dim() != 2 or tensor.shape[1] != widths[name]:
                raise ValueError(
                    f"sac.demo_transitions ({path}): '{name}' must be (rows, {widths[name]}), got "
                    f"{tuple(tensor.shape)} — was the file made for this env and agent?"
                )
            if rows is None:
                rows = tensor.shape[0]
            elif tensor.shape[0] != rows:
                raise ValueError(
                    f"sac.demo_transitions ({path}): '{name}' has {tensor.shape[0]} rows, "
                    f"'{names[0]}' has {rows}"
                )
            dtype = torch.bool if name in ("terminated", "truncated") else torch.float32
            out[name] = tensor.to(device=self.device, dtype=dtype)
        if not rows:
            raise ValueError(f"sac.demo_transitions ({path}) holds no transitions")
        # each row's demo episode (0, 1, ...; a row that ends one closes it) and each episode's outcome (1 when it
        # holds a positive reward; shaped rewards are no success flag, so a shaped file carries its rows' own
        # "success"): demo_storage memory gives them episode ids
        ends = (out["terminated"] | out["truncated"]).reshape(-1)
        self._demo_episode = torch.cumsum(ends.to(torch.long), 0) - ends.to(torch.long)
        episodes = int(self._demo_episode.max()) + 1
        if own:
            if "success" not in transitions or tuple(transitions["success"].shape) != (rows, 1):
                raise KeyError(f"sac.demo_transitions ({path}) is shaped (potential_terms) but carries no (rows, 1) 'success'")
            success = transitions["success"].to(device=self.device, dtype=torch.float32).reshape(-1) > 0.5
        else:
            success = out["rewards"].reshape(-1) > 0
        self._demo_episode_success = torch.zeros(episodes, device=self.device).index_reduce_(
            0, self._demo_episode, success.to(torch.float32), "amax"
        )
        if self._n_step() > 1:
            out = n_step_rows(out, self._n_step(), float(self.cfg.discount_factor))
        # actions may lie beyond the squashed policy's [-1, 1]: only the critic sees demo rows,
        # and it takes the action as an input, so the value of a correction the actor cannot
        # reach is still a true value. They must be finite.
        for name in names:
            if out[name].is_floating_point() and not torch.isfinite(out[name]).all():
                raise ValueError(f"sac.demo_transitions ({path}): '{name}' holds non-finite values")
        print(f"[sac] {rows} demo transitions from {path}", flush=True)
        return out

    def _write_demos_to_memory(self, demo: Dict[str, torch.Tensor]) -> None:
        """``demo_storage: memory``: the demo rows into every agent's replay memory, ahead of the online rows, with
        episode ids of their own when the PARTS terms track outcomes."""
        rows = dict(demo)
        if getattr(self, "_track_outcomes", False):
            first = self._next_episode
            ids = first + self._demo_episode
            episodes = self._demo_episode_success.numel()
            need = first + episodes
            if need > self._outcomes.numel():
                grown = torch.full((max(need, 2 * self._outcomes.numel()),), -1.0, device=self.device)
                grown[: self._outcomes.numel()] = self._outcomes
                self._outcomes = grown
            self._outcomes[first:need] = self._demo_episode_success
            self._next_episode = need
            rows["episode_ids"] = ids.to(torch.float32).unsqueeze(-1)
        if self.cfg.demo_storage == "memory_critic":
            rows["is_demo"] = torch.ones(next(iter(demo.values())).shape[0], 1, device=self.device)
            if getattr(self, "_track_outcomes", False):
                # outcome -1 ("still running"): in neither PARTS term (the actor skips demo rows anyway)
                self._outcomes[first:need] = -1.0
        missing = [name for name in self._tensors_names if name not in rows]
        if missing:
            raise KeyError(f"sac.demo_storage: memory: the demo rows lack the memory's {missing}")
        if self.cfg.demo_storage == "memory_critic" and int(self.memory.size) != 0:
            raise RuntimeError("sac.demo_storage: memory_critic writes the demos first, into an empty memory")
        self.memory.add_block(**{name: rows[name] for name in self._tensors_names})
        if self.cfg.demo_storage == "memory_critic":
            # the online rows of each agent are [n_demo, fill) until the memory wraps: the actor's sampler bound
            self._demo_rows = list(self.memory._agent_size)
            self._online_low = torch.tensor(self._demo_rows, dtype=torch.float32, device=self.device)
        print(f"[sac] {next(iter(demo.values())).shape[0]} demo transitions written into the replay memory", flush=True)

    @staticmethod
    def _space_width(space) -> int:
        if isinstance(space, int):
            return int(space)
        return int(np.prod(space.shape))

    def _demo_active(self, timestep: int) -> bool:
        return self._demo is not None and (
            self.cfg.demo_sampling_start <= timestep < self.cfg.demo_sampling_end
        )

    def _sample(self, rows: int, graph: bool = False) -> Dict[str, torch.Tensor]:
        sample = self.memory.sample_graph_safe if graph else self.memory.sample
        return dict(zip(self._tensors_names, sample(names=self._tensors_names, batch_size=rows)[0]))

    def _sample_online(self, rows: int, graph: bool = False) -> Dict[str, torch.Tensor]:
        """``demo_storage: memory_critic``: ``rows`` rows per agent from its online rows only (``is_demo`` 0). Until the
        memory wraps those are each agent's rows ``[n_demo, fill)``, sampled with device work only (capturable);
        after a wrap, by the flag (eager)."""
        names = self._tensors_names
        if not self._memory_wrapped():
            sample = self.memory.sample_range_graph_safe if graph else self._sample_range_eager
            return dict(zip(names, sample(names=names, batch_size=rows, low=self._online_low)[0]))
        return dict(zip(names, self.memory.sample_where(names=names, batch_size=rows, flag="is_demo", value=0.0)[0]))

    def _sample_range_eager(self, *, names, batch_size: int, low: torch.Tensor):
        self.memory.sync_size()
        return self.memory.sample_range_graph_safe(names=names, batch_size=batch_size, low=low)

    def _memory_wrapped(self) -> bool:
        """Whether some agent's memory has written past its capacity (old rows, demos first, overwritten)."""
        return max(self.memory.written()) > self.memory.capacity

    def _online_rows_available(self) -> bool:
        """``demo_storage: memory_critic``: every agent holds at least one online row (host side, outside any graph)."""
        if self._memory_wrapped():
            return min(self.memory.count_where("is_demo", 0.0)) > 0
        fills = self.memory._agent_size or [int(self.memory.size)] * self.num_agents
        return min(f - d for f, d in zip(fills, self._demo_rows)) > 0

    def _online_rows_ready(self) -> bool:
        """``learning_starts_rows``: every agent's memory holds that many online rows (cached once true)."""
        need = int(self.cfg.learning_starts_rows)
        if not need or getattr(self, "_rows_ready", False):
            return True
        if self.memory is None or self.memory.size == 0:
            return False
        if "is_demo" in self._tensors_names:
            counts = self.memory.count_where("is_demo", 0.0)
        else:
            counts = list(self.memory._agent_size or [int(self.memory.size)] * self.num_agents)
        self._rows_ready = min(counts) >= need
        return self._rows_ready

    def _mixed_batch(self, rows: int, graph: bool = False) -> Dict[str, torch.Tensor]:
        """The critic's batch inside the demo window: per agent, ``min(rows // 2, stored)``
        online rows, the rest demo rows. Draw shapes depend only on the step count, never on
        the data, and each agent draws its own demo rows, so agents stay independent."""
        n_online = min(rows // 2, int(self.memory.size))
        n_demo = rows - n_online
        names = self._demo_names() + (["discounts"] if self._n_step() > 1 else [])
        demo_rows = next(iter(self._demo.values())).shape[0]
        index = torch.randint(0, demo_rows, (self.num_agents, n_demo), device=self.device)
        online = self._sample(n_online, graph) if n_online else None
        batch = {}
        for name in names:
            demo = self._demo[name][index]  # (num_agents, n_demo, width)
            if online is not None:
                part = online[name].view(self.num_agents, n_online, -1)
                demo = torch.cat((part.to(demo.dtype), demo), dim=1)
            batch[name] = demo.reshape(self.num_agents * rows, -1)
        if getattr(self, "_track_outcomes", False):
            # demo rows carry no attempt of their own: a placeholder id, and a flag that keeps them out of PARTS'
            # clone / anchor terms (the Q term still reads them, e.g. with demo_batch_shared)
            ids = torch.zeros((self.num_agents, n_demo, 1), device=self.device)
            skip = torch.ones((self.num_agents, n_demo, 1), device=self.device)
            if online is not None:
                ids = torch.cat((online["episode_ids"].view(self.num_agents, n_online, 1).to(ids.dtype), ids), dim=1)
                skip = torch.cat((torch.zeros((self.num_agents, n_online, 1), device=self.device), skip), dim=1)
            batch["episode_ids"] = ids.reshape(self.num_agents * rows, 1)
            batch["_parts_skip"] = skip.reshape(self.num_agents * rows, 1)
        return batch

    def _resolve_target_entropy(self) -> float:
        if self.cfg.target_entropy is not None:
            return float(self.cfg.target_entropy)
        if self.cfg.target_entropy_sigma is not None:
            if not isinstance(self.action_space, gymnasium.spaces.Box):
                raise ValueError(
                    f"sac.target_entropy_sigma is a Gaussian's entropy per action dim; action space "
                    f"{self.action_space} is not a Box"
                )
            # FlashSAC: 0.5 |A| log(2 pi e sigma^2)
            sigma = float(self.cfg.target_entropy_sigma)
            return float(0.5 * np.prod(self.action_space.shape) * np.log(2.0 * np.pi * np.e * sigma**2))
        if isinstance(self.action_space, gymnasium.spaces.Box):
            return float(-np.prod(self.action_space.shape))
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            return float(-self.action_space.n)
        raise ValueError(
            f"cannot derive a target entropy for action space {self.action_space}; set "
            "sac.target_entropy"
        )

    def _build_optimizers(self) -> None:
        self.policy_optimizer = BlockAdamW(
            self.policy.parameters(),
            self.num_agents,
            lr=self.cfg.actor_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.critic_optimizer = BlockAdamW(
            self.critics.parameters(),
            self.num_agents,
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self._updates = 0
        if getattr(self, "_graphs", None) is not None:
            self._graphs.clear()  # captured against the replaced optimizers

    def _update_rounds(self, timesteps: int) -> int:
        """How many update rounds a run of ``timesteps`` env steps holds: one at every env step
        ``t >= learning_starts`` with ``(t - learning_starts) % update_every == 0``."""
        steps = max(0, int(timesteps) - int(self.cfg.learning_starts))
        return -(-steps // int(self.cfg.update_every))  # ceil

    def _set_learning_rates(self, timesteps: int) -> None:
        """The LR for this gradient step, from the update count alone (never from the data)."""
        cfg = self.cfg
        if cfg.lr_schedule == "constant" and not (cfg.actor_lr_warmup or cfg.critic_lr_warmup):
            return
        total = max(1, self._update_rounds(timesteps) * int(cfg.gradient_steps))
        self.policy_optimizer.set_lr(
            lr_at(self._updates, total, cfg.actor_lr, cfg.lr_end, cfg.lr_schedule, warmup=cfg.actor_lr_warmup)
        )
        self.critic_optimizer.set_lr(
            lr_at(self._updates, total, cfg.critic_lr, cfg.lr_end, cfg.lr_schedule, warmup=cfg.critic_lr_warmup)
        )

    def _create_memory_tensors(self) -> None:
        if self.memory is None:
            raise ValueError("SAC needs a replay memory")
        self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        # truncated is stored too, because the bootstrap must tell a terminal state from the
        # clock running out: Isaac Lab's Factory and Forge raise BOTH flags at the time limit
        self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self._tensors_names = [
            "observations", "actions", "rewards", "next_observations", "terminated", "truncated",
        ]
        if self._n_step() > 1:
            # gamma ** m of each n-step row; the demo rows get theirs when they are loaded
            self.memory.create_tensor(name="discounts", size=1, dtype=torch.float32)
            self._tensors_names.append("discounts")
            self._n_step_window = NStepWindow(self._n_step(), float(self.cfg.discount_factor))
        if self._asymmetric:
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self._tensors_names += ["states", "next_states"]
        # whatever the configured aux losses need per transition (e.g. contact flags)
        self._tensors_names += self.create_aux_memory_tensors()
        #: PARTS' actor terms: every stored transition carries its episode's id; the outcomes table holds each
        #: finished episode's outcome (1 success, 0 failure; -1 while it runs)
        self._track_outcomes = bool(
            self.cfg.success_clone_weight or self.cfg.failure_anchor_weight or self.cfg.retrain_every
        )
        #: potential_terms: this SAC's channels and weights (its route's entry, or "*"); Phi of the observation the
        #: next transition starts from, and whether it belongs to that transition's episode
        self._potential_terms = self._own_potential_terms()
        self._potential_prev = None
        self._potential_valid = None
        if self._track_outcomes:
            self.memory.create_tensor(name="episode_ids", size=1, dtype=torch.float32)
            self._tensors_names.append("episode_ids")
            self._env_episode = torch.arange(self.num_envs, device=self.device)
            self._next_episode = self.num_envs
            # preallocated so a captured CUDA graph keeps reading the same tensor (growth re-captures)
            self._outcomes = torch.full((max(1 << 16, 4 * self.num_envs),), -1.0, device=self.device)
        #: the critic action transform's side_inputs: name -> width, stored as name and next_name
        self._side_inputs = {
            str(name): int(width)
            for name, width in (getattr(self.critic_action_transform, "side_inputs", None) or {}).items()
        }
        for name, width in sorted(self._side_inputs.items()):
            for stored in (name, f"next_{name}"):
                self.memory.create_tensor(name=stored, size=width, dtype=torch.float32)
                self._tensors_names.append(stored)
        if self.cfg.demo_storage == "memory_critic":
            # 1 on a demo row: the critics train on it, the actor's loss skips it
            self.memory.create_tensor(name="is_demo", size=1, dtype=torch.float32)
            self._tensors_names.append("is_demo")
        if self.cfg.demo_transitions:
            self._demo = self._load_demo_transitions()
            if self.cfg.demo_storage in ("memory", "memory_critic"):
                self._write_demos_to_memory(self._demo)
                self._demo = None  # ordinary memory rows from here on
        if self._demo is not None and self.aux_memory_keys():
            raise ValueError(
                "sac.demo_transitions cannot be mixed into batches while aux losses need per-"
                f"transition keys {sorted(self.aux_memory_keys())}: the demos do not carry them"
            )

    # ------------------------------------------------------------------ normalization
    def normalize_observations(self, observations: torch.Tensor, train: bool = False) -> torch.Tensor:
        if self.observation_normalizer is None:
            return observations
        return self.observation_normalizer(observations, train=train)

    def normalize_states(self, states: torch.Tensor, train: bool = False) -> torch.Tensor:
        if self.state_normalizer is None:
            return states
        return self.state_normalizer(states, train=train)

    # ------------------------------------------------------------------ interaction
    def _warming_up(self) -> bool:
        """``warmup_episodes``: still within the first episodes, run on zero actions with no updates."""
        return int(self.cfg.warmup_episodes) > getattr(self, "_finished_episodes", 0)

    def act(self, observations, states, *, timestep: int, timesteps: int):
        if self.training and self._warming_up():
            width = self._space_width(self.action_space)
            return torch.zeros((observations.shape[0], width), device=self.device), {}
        if self.training and timestep < self.cfg.random_timesteps:
            return self._explore_progressively(self.random_actions(observations.shape[0]), timestep), {}
        inputs = {"observations": self.normalize_observations(observations)}
        if self.training and self.cfg.noise_repeat:
            inputs["noise"] = self._repeated_noise(observations.shape[0])
        # no_grad: these actions only drive the env; the update re-runs the policy on replay
        # batches. A live graph here would be spliced into the env's action buffers and grow.
        with torch.no_grad(), self._batch_mode(self.policy, False):  # a BatchNorm actor acts on its running stats
            actions, outputs = self.policy.act(inputs, role="policy")
        self.emit_selection(outputs, timestep)
        self._emit_std_by_label(outputs, timestep)
        return self._explore_progressively(actions, timestep), outputs

    def _repeated_noise(self, rows: int) -> torch.Tensor:
        """``noise_repeat``: FlashSAC's ``_sample_flashsac_actions`` per agent. Every step a
        fresh standard-normal draw and a fresh length k ~ truncated Zeta are made; an agent
        whose held draw has been used k times (or never drawn) takes them, the others keep
        theirs. The draws' shapes depend on nothing but the env count, and an agent's counter
        on nothing but its own draws. Episode ends do not cut a hold short (nor in FlashSAC)."""
        width = int(self.policy.num_continuous)
        if self._held_noise is None or self._held_noise.shape != (rows, width):
            self._held_noise = torch.zeros(rows, width, device=self.device)
            self._noise_count.zero_()
        redraw = (self._noise_count == 0) | (self._noise_count >= self._noise_length)  # (N,)
        fresh = torch.randn(rows, width, device=self.device)
        uniform = torch.rand(self.num_agents, 1, device=self.device)
        length = torch.argmax((uniform < self._zeta_cdf).to(torch.int32), dim=1) + 1
        per_row = redraw.repeat_interleave(rows // self.num_agents).unsqueeze(-1)
        self._held_noise = torch.where(per_row, fresh, self._held_noise)
        self._noise_length = torch.where(redraw, length, self._noise_length)
        self._noise_count = torch.where(redraw, torch.zeros_like(self._noise_count), self._noise_count) + 1
        return self._held_noise

    def _std_labels_from(self, infos: Any) -> Optional[torch.Tensor]:
        """``std_by_label``'s per-env labels in this step's ``infos`` (long, ``(num_envs,)``), or None."""
        path = self.cfg.std_by_label
        if not path or not isinstance(infos, dict):
            return None
        value = infos.get(path)
        if value is None and "/" in path:
            key, name = path.split("/", 1)
            channel = infos.get(key)
            value = channel.get(name) if isinstance(channel, dict) else None
        if value is None:
            return None
        return torch.as_tensor(value, device=self.device).reshape(-1).round().long()

    def _emit_std_by_label(self, outputs: Dict[str, Any], timestep: int) -> None:
        """``policy/std_<name>``: each env's acting std (mean over its action dims), grouped by the label its env
        published for this observation (the previous step's ``infos``; :meth:`record_transition` keeps it), one
        value per env in the group, per agent. Nothing before the first label or for an empty group."""
        labels = getattr(self, "_std_labels", None)
        log_std = outputs.get("log_std") if isinstance(outputs, dict) else None
        if not self.on_log or not self.training or labels is None or log_std is None:
            return
        std = log_std.exp().mean(dim=-1)
        if labels.shape[0] != std.shape[0]:
            return
        per_agent = std.shape[0] // self.num_agents
        for index, name in enumerate(self.cfg.std_label_names):
            hit = labels == index
            for agent in range(self.num_agents):
                rows = slice(agent * per_agent, (agent + 1) * per_agent)
                values = std[rows][hit[rows]]
                if values.numel():
                    self.emit(agent, {f"policy/std_{name}": values}, timestep)

    def _emit_metrics_by_label(self, infos: Any, timestep: int) -> None:
        """``metrics_by_label``: this step's per-env ``metrics_to_log`` values, per ``std_by_label`` group (the label
        of the observation acted on), over the envs that ran the policy (all of them without progressive
        exploration), as ``<name>_<group>``, one value per env, per agent."""
        labels = getattr(self, "_std_labels", None)
        channel = infos.get("metrics_to_log") if isinstance(infos, dict) else None
        if not self.on_log or not self.training or labels is None or not isinstance(channel, dict):
            return
        active = getattr(self, "_explore_keep", None)
        if int(self.cfg.progressive_exploration_steps) <= 0 or active is None or active.shape != labels.shape:
            active = torch.ones_like(labels, dtype=torch.bool)
        per_agent = labels.shape[0] // self.num_agents
        for name in self.cfg.metrics_by_label:
            value = channel.get(name)
            if value is None:
                continue
            value = torch.as_tensor(value, device=self.device).reshape(-1)
            if value.shape != labels.shape:
                continue
            for index, group in enumerate(self.cfg.std_label_names):
                hit = (labels == index) & active
                for agent in range(self.num_agents):
                    rows = slice(agent * per_agent, (agent + 1) * per_agent)
                    values = value[rows][hit[rows]]
                    if values.numel():
                        self.emit(agent, {f"{name}_{group}": values}, timestep)

    def _explore_share(self, timestep: int) -> float:
        """``min(max(timestep - start, 0) / H, 1)``: the chance an env runs the policy this step, 0 until
        ``progressive_exploration_start``."""
        since = max(timestep - int(self.cfg.progressive_exploration_start), 0)
        return min(since / int(self.cfg.progressive_exploration_steps), 1.0)

    def _explore_progressively(self, actions: torch.Tensor, timestep: int) -> torch.Tensor:
        """``progressive_exploration_steps`` H (Policy Decorator): while training, each env runs
        this step on its actions with probability ``min(timestep / H, 1)`` and on zeros
        otherwise, one uniform per env drawn EVERY env step (Policy Decorator's
        ``mask = rand(num_envs) < min(step / H, 1)`` inside its step loop), so an env's draw
        depends on nothing but the step count. What this returns is what the env executes and
        the memory stores. Off (H = 0) or outside training it draws nothing.
        ``exploration/policy_share``: the agent's envs on their policy this step."""
        if not self.training or int(self.cfg.progressive_exploration_steps) <= 0:
            return actions
        keep = torch.rand(actions.shape[0], 1, device=actions.device) < self._explore_share(timestep)
        #: the envs that ran the policy this step (metrics_by_label counts only those)
        self._explore_keep = keep.reshape(-1)
        self.emit_per_agent(
            {"exploration/policy_share": keep.float().view(self.num_agents, -1).mean(dim=1)}, timestep
        )
        return torch.where(keep, actions, torch.zeros_like(actions))

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep: int,
        timesteps: int,
    ) -> None:
        self.observe_step(
            infos=infos,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            step=timestep,
        )
        if self.training:
            # the env's own episode ends (warmup_episodes counts them)
            self._finished_episodes = getattr(self, "_finished_episodes", 0) + int((terminated | truncated).sum())
        shaping = self._potential_shaping(infos, terminated | truncated) if self._potential_terms else None
        #: the label of the observation this transition starts from (store_labels)
        start_labels = getattr(self, "_std_labels", None)
        if self.cfg.std_by_label:
            self._emit_metrics_by_label(infos, timestep)
            # the label of the observation the next act() sees
            self._std_labels = self._std_labels_from(infos)

        if not self.training:
            return
        local_outcome = None
        if self.cfg.route_label:
            # this residual's own attempts: rewarded and ended by its label's local outcome
            rewards, terminated, truncated, local_outcome = self._route_signals(start_labels, terminated, truncated)
        if shaping is not None:
            rewards = rewards + shaping.to(rewards.dtype).reshape(rewards.shape)
        extra: Dict[str, torch.Tensor] = {}
        if self._asymmetric:
            if states is None or next_states is None:
                raise RuntimeError(
                    "asymmetric SAC needs states and next_states from the trainer "
                    f"(got states={states is not None}, next_states={next_states is not None})"
                )
            extra["states"] = states
            extra["next_states"] = next_states
        extra.update(self.aux_memory_values(infos))
        extra.update(self._side_input_values(infos, observations.shape[0]))
        if self.cfg.route_label:
            # route_clock: this step is one of this residual's own when an env starts it in the route's label
            index = self.cfg.std_label_names.index(self.cfg.route_label)
            self._route_step_now = start_labels is not None and bool((start_labels == index).any())
            self._route_steps = getattr(self, "_route_steps", 0) + int(self._route_step_now)
        if self.cfg.demo_storage == "memory_critic":
            extra["is_demo"] = torch.zeros(observations.shape[0], 1, device=self.device)
        if self.cfg.hold_still_key:
            n = observations.shape[0]
            extra["_still"] = self._infos_flags(infos, self.cfg.hold_still_key, n)
            # a last-step field of an n-step window (the "next_" prefix): set on every row whose window ends on a stall
            extra["next__stalled"] = self._infos_flags(infos, self.cfg.hold_stall_key, n) * (terminated | truncated).reshape(n, 1).to(torch.float32)
        if self.cfg.executed_action_mask:
            actions = actions * self._executed_mask(infos, actions)
        if getattr(self, "_track_outcomes", False):
            extra["episode_ids"] = self._env_episode.to(torch.float32).unsqueeze(-1)
            self._record_outcomes(infos, terminated, truncated, outcome=local_outcome)
        stored_labels = [self.cfg.route_label] if self.cfg.route_label else list(self.cfg.store_labels)
        if stored_labels:
            # stored only where the starting observation's label is listed (none before the first label is known)
            allowed = [self.cfg.std_label_names.index(label) for label in stored_labels]
            if start_labels is None:
                keep = torch.zeros(observations.shape[0], dtype=torch.bool, device=self.device)
            else:
                keep = torch.isin(start_labels, torch.tensor(allowed, device=start_labels.device))
            extra["_keep"] = keep.to(torch.float32).unsqueeze(-1)
        stored_rewards = self.shape_rewards(rewards, timestep, timesteps)
        if self.return_scaler is not None:
            self.return_scaler.update(stored_rewards, terminated | truncated)
        row = dict(
            observations=observations,
            actions=actions,
            rewards=stored_rewards,
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
            **extra,
        )
        if self._n_step_window is not None:
            # the n-step row of the step n - 1 back, once the window holds n steps
            row = self._n_step_window.push(row)
            if row is None:
                return
        if self.cfg.hold_still_key:
            self._write_holding(row)
            return
        if "_keep" in row:  # store_labels: only the transitions starting in a listed phase
            keep = row.pop("_keep").reshape(-1) > 0.5
            self.memory.add_rows(keep, **row)
            return
        self.memory.add_samples(**row)

    def forced_env_reset(self) -> None:
        super().forced_env_reset()
        self._potential_prev = None  # potential_terms: every env starts over; the kept Phi is the cut episodes'

    def _potential_shaping(self, infos: Any, done: torch.Tensor) -> torch.Tensor:
        """``potential_terms``: this step's ``(num_envs, 1)`` shaping ``gamma Phi(s') - Phi(s)`` (0 where ``Phi(s)``
        is unknown or the ended episode's), ``Phi(s')`` from this step's ``infos``, kept as the next step's ``Phi(s)``."""
        rows = done.reshape(-1).shape[0]
        phi = torch.zeros(rows, 1, device=self.device)
        for path, weight in self._potential_terms.items():
            value = infos.get(path) if isinstance(infos, dict) else None
            if value is None and isinstance(infos, dict) and "/" in path:
                key, name = path.split("/", 1)
                channel = infos.get(key)
                value = channel.get(name) if isinstance(channel, dict) else None
            if value is None:
                raise RuntimeError(f"sac.potential_terms: the env's infos carry no '{path}' this step")
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(-1, 1)
            if value.shape[0] != rows:
                raise ValueError(f"infos '{path}' has {value.shape[0]} rows, the step {rows}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"sac.potential_terms: infos '{path}' is not finite this step: {value.reshape(-1).tolist()}")
            phi = phi - float(weight) * value
        if self._potential_prev is None:
            shaping = torch.zeros_like(phi)
        else:
            shaping = (float(self.cfg.discount_factor) * phi - self._potential_prev) * self._potential_valid
        self._potential_prev = phi
        # an env whose episode ended starts the next step from its new episode's first observation
        self._potential_valid = (~done.reshape(-1, 1).to(torch.bool)).to(torch.float32).to(self.device)
        return shaping

    def _infos_flags(self, infos: Any, path: str, rows: int) -> torch.Tensor:
        """``(rows, 1)`` float flags from the ``infos`` channel ``path`` (``"<key>/<name>"`` reads ``infos[key][name]``)."""
        value = infos.get(path) if isinstance(infos, dict) else None
        if value is None and isinstance(infos, dict) and "/" in path:
            key, name = path.split("/", 1)
            channel = infos.get(key)
            value = channel.get(name) if isinstance(channel, dict) else None
        if value is None:
            raise RuntimeError(f"sac.hold_still_key / hold_stall_key: the env's infos carry no '{path}' this step")
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(-1, 1)
        if value.shape[0] != rows:
            raise ValueError(f"infos '{path}' has {value.shape[0]} rows, the step {rows}")
        return (value > 0.5).to(torch.float32)

    def _write_holding(self, row: Dict[str, torch.Tensor]) -> None:
        """``hold_still_key``: write each env's row now when its step made progress (after the rows held for that env,
        in order); hold it while the task makes no progress; when the episode ends for lack of progress, write only
        the first held row, as a terminal failure, and drop the rest (and the window's later rows of that end)."""
        n = self.num_envs
        still = row.pop("_still").reshape(-1) > 0.5
        stalled = row.pop("next__stalled").reshape(-1) > 0.5
        keep = row.pop("_keep").reshape(-1) > 0.5 if "_keep" in row else torch.ones(n, dtype=torch.bool, device=self.device)
        if not hasattr(self, "_held"):
            self._held = [[] for _ in range(n)]
            self._stall_resolved = [False] * n
        for env in range(n):
            one = {name: value[env : env + 1] for name, value in row.items()}
            if bool(stalled[env]):
                if self._stall_resolved[env]:
                    continue  # a later row of the same stall: dropped
                self._stall_resolved[env] = True
                first, first_keep = self._held[env][0] if self._held[env] else (one, bool(keep[env]))
                first = dict(first)
                first["terminated"] = torch.ones_like(first["terminated"])
                first["truncated"] = torch.zeros_like(first["truncated"])
                self._write_env_row(env, first, first_keep)
                self._held[env] = []
                continue
            self._stall_resolved[env] = False
            if bool(still[env]):
                self._held[env].append((one, bool(keep[env])))
                continue
            for held, held_keep in self._held[env]:
                self._write_env_row(env, held, held_keep)
            self._held[env] = []
            self._write_env_row(env, one, bool(keep[env]))

    def _write_env_row(self, env: int, row: Dict[str, torch.Tensor], keep: bool) -> None:
        """One env's row (every value ``(1, width)``) into the memory, at that env's agent's pointer."""
        if not keep:
            return
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env] = True
        self.memory.add_rows(mask, **{name: value.expand(self.num_envs, *value.shape[1:]) for name, value in row.items()})

    def _executed_mask(self, infos: Any, actions: torch.Tensor) -> torch.Tensor:
        """``executed_action_mask``: ``(num_envs, 1)`` 1 where the env executed the agent's action this step."""
        key = self.cfg.executed_action_mask
        value = infos.get(key) if isinstance(infos, dict) else None
        if value is None:
            raise RuntimeError(f"sac.executed_action_mask: the env's infos carry no '{key}' this step")
        mask = torch.as_tensor(value, device=actions.device, dtype=actions.dtype).reshape(-1, 1)
        if mask.shape[0] != actions.shape[0]:
            raise ValueError(f"infos['{key}'] has {mask.shape[0]} rows, the actions {actions.shape[0]}")
        return mask

    def _route_signals(self, start_labels, terminated: torch.Tensor, truncated: torch.Tensor):
        """``route_label``: ``(rewards, terminated, truncated, outcome)`` of this residual's own attempts, per env:
        success (reward 1) when the transition starts in the label and the next observation's label is a later
        ``std_label_names`` entry; the attempt ends there, or when the env's episode ends (success 0). Truncation is
        never separate: an attempt's end is terminal."""
        names = list(self.cfg.std_label_names)
        index = names.index(self.cfg.route_label)
        done = (terminated | truncated).reshape(-1, 1).to(torch.bool)
        if start_labels is None or self._std_labels is None:
            success = torch.zeros_like(done)
        else:
            start = start_labels.reshape(-1, 1)
            after = self._std_labels.reshape(-1, 1)
            success = (start == index) & (after > index) & (after < len(names))
        end = success | done
        return success.to(torch.float32), end, torch.zeros_like(end), success.reshape(-1).to(torch.float32)

    def _record_outcomes(self, infos: Any, terminated: torch.Tensor, truncated: torch.Tensor, outcome=None) -> None:
        """Record the outcome of every episode that ended this step and give those envs new episode ids
        (``outcome``: the per-env outcomes to record, e.g. a route's local ones; None reads ``outcome_metric``)."""
        done = (terminated.reshape(-1) | truncated.reshape(-1)).to(self.device)
        if not bool(done.any()):
            return
        if outcome is None:
            channel = infos.get("episode_metrics_to_log") if isinstance(infos, dict) else None
            value = channel.get(self.cfg.outcome_metric) if isinstance(channel, dict) else None
            if value is None:
                raise RuntimeError(
                    f"sac.success_clone_weight / failure_anchor_weight: an episode ended but infos carry no "
                    f"episode_metrics_to_log['{self.cfg.outcome_metric}']"
                )
            outcome = (torch.as_tensor(value, device=self.device).reshape(-1) > 0.5).to(torch.float32)
        outcome = outcome.to(self.device)
        ids = self._env_episode[done]
        need = int(ids.max()) + 1 + int(done.sum())
        if need > self._outcomes.numel():
            grown = torch.full((max(need, 2 * self._outcomes.numel()),), -1.0, device=self.device)
            grown[: self._outcomes.numel()] = self._outcomes
            self._outcomes = grown
            if self._graphs is not None:
                self._graphs.clear()  # the captured steps read the old table
        self._outcomes[ids] = outcome[done]
        fresh = torch.arange(self._next_episode, self._next_episode + int(done.sum()), device=self.device)
        self._env_episode[done] = fresh
        self._next_episode += int(done.sum())

    def _side_input_values(self, infos: Any, rows: int) -> Dict[str, torch.Tensor]:
        """This step's ``side_inputs`` (and their ``next_`` values) from ``infos``, shape-checked."""
        out: Dict[str, torch.Tensor] = {}
        for name, width in getattr(self, "_side_inputs", {}).items():
            for stored in (name, f"next_{name}"):
                value = infos.get(stored) if isinstance(infos, dict) else None
                if value is None:
                    raise RuntimeError(
                        f"the critic action transform reads the side input '{name}', but the env's infos carry "
                        f"no '{stored}' this step"
                    )
                if tuple(value.shape) != (rows, width):
                    raise ValueError(f"infos['{stored}'] must be ({rows}, {width}), got {tuple(value.shape)}")
                out[stored] = value.to(device=self.device, dtype=torch.float32)
        return out

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        if self._warming_up() and not self.cfg.warmup_trains_critics:
            return  # warmup_episodes: no update until the base's episodes have finished
        if not self._online_rows_ready():
            return  # learning_starts_rows: not yet this much of its own data
        if self.cfg.route_clock:
            # this residual's own steps only: nothing while its phase is not running
            if not getattr(self, "_route_step_now", False):
                return
            clock = self._route_steps - 1
            since_start = clock - int(self.cfg.learning_starts)
            if since_start >= 0 and since_start % int(self.cfg.update_every) == 0:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
            self._maybe_periodic_reset(timestep)
            self._maybe_retrain(clock, timesteps, logged_at=timestep)
            return
        # learning_starts is the first update of anything (the critics); the actor joins at
        # policy_learning_starts. From there one round of gradient_steps every update_every
        # env steps (policy_delay keeps counting gradient steps)
        since_start = timestep - int(self.cfg.learning_starts)
        if since_start >= 0 and since_start % int(self.cfg.update_every) == 0:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)
        self._maybe_periodic_reset(timestep)
        self._maybe_retrain(timestep, timesteps)

    # ------------------------------------------------------------------ losses
    def _target_min(self, target_qs: torch.Tensor, rows: int) -> torch.Tensor:
        """Min over the target critics ``(critics, agents * rows, 1)``: all of them, or a random
        ``critic_subset`` per agent."""
        subset = int(self.cfg.critic_subset)
        if subset == self._num_critics and not self.cfg.critic_subset_replacement:
            return target_qs.min(dim=0).values  # draws nothing
        per_agent = target_qs.view(self._num_critics, self.num_agents, rows, -1).permute(1, 0, 2, 3)
        if self.cfg.critic_subset_replacement:
            chosen = torch.randint(0, self._num_critics, (self.num_agents, subset), device=self.device)
        else:
            chosen = torch.argsort(torch.rand(self.num_agents, self._num_critics, device=self.device), dim=1)[:, :subset]
        index = chosen[:, :, None, None].expand(-1, -1, rows, per_agent.shape[-1])
        return per_agent.gather(1, index).min(dim=1).values.reshape(self.num_agents * rows, -1)

    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, rows):
        """Min-over-targets bootstrapped target (with the entropy bonus unless
        ``backup_entropy`` is off) + mean MSE over the critics; with a categorical critic the
        projected target distribution of the lowest-Q target critic + mean cross-entropy.
        Also returns the policy's log-prob of its next actions, for ``entropy_observations:
        next``, and (categorical) each agent's cross-entropy, else None."""
        with torch.no_grad(), self._batch_mode(self.policy, False):
            next_actions, outputs = self.policy.act(next_inputs, role="policy")
            next_critic_actions = self._critic_actions(sampled, "next_observations", next_actions)
        critic_actions = self._critic_actions(sampled, "observations", sampled["actions"])
        if self.cfg.cross_batch_value:
            # FlashSAC: [s, a] ++ [s', a'] in one call for both; the target reads the next half,
            # the critics their current half
            both = {
                **critic_inputs,
                "observations": self._join_halves(critic_inputs["observations"], critic_next_inputs["observations"], rows),
                "taken_actions": self._join_halves(critic_actions, next_critic_actions, rows),
            }
            with torch.no_grad(), self._batch_mode(self.target_critics, True):
                target_out = self._critic_half(self.target_critics.act(both, role="target_critics")[0], rows, 1)
            with self._batch_mode(self.critics, True):
                values = self._critic_half(self.critics.act(both, role="critics")[0], rows, 0)
        else:
            with torch.no_grad(), self._batch_mode(self.target_critics, True):
                target_out, _ = self.target_critics.act(
                    {**critic_next_inputs, "taken_actions": next_critic_actions}, role="target_critics"
                )
            with self._batch_mode(self.critics, True):
                values, _ = self.critics.act({**critic_inputs, "taken_actions": critic_actions}, role="critics")

        with torch.no_grad():
            rewards = sampled["rewards"]
            if self.return_scaler is not None:
                rewards = rewards / self.expand_per_agent(self.return_scaler.denominator, rows)
            # Bootstrap unless the next state is genuinely terminal. A time limit is not the
            # end of the world: the episode had a future, it just stopped being observed, so
            # dropping gamma*V(next) there would teach the critic that running out the clock
            # is worth nothing. Isaac Lab's Factory and Forge raise terminated AND truncated
            # at the limit (``_get_dones`` returns one ``time_out`` tensor twice), so reading
            # ``terminated`` alone would do exactly that to every episode.
            terminal = sampled["terminated"] & sampled["truncated"].logical_not()
            if not self.cfg.bootstrap_time_limits:
                # every episode end is terminal: Isaac Lab's s' after a reset is the next episode's first observation
                terminal = sampled["terminated"] | sampled["truncated"]
            # an n-step row bootstraps gamma ** m later (its own discounts), a 1-step row gamma
            discount = sampled["discounts"] if self._n_step() > 1 else self.cfg.discount_factor
            bonus = (
                self.expand_per_agent(self._entropy_coefficient, rows) * outputs["log_prob"]
                if self.cfg.backup_entropy
                else 0.0
            )
            if self._categorical:
                # FlashSAC's _select_min_q_log_probs + _compute_categorical_td_target
                target_log_probs = F.log_softmax(target_out, dim=-1)  # (critics, R, bins)
                target_qs = (target_log_probs.exp() * self._atoms).sum(dim=-1)
                lowest = target_qs.argmin(dim=0)
                chosen = target_log_probs.gather(
                    0, lowest[None, :, None].expand(1, -1, target_log_probs.shape[-1])
                )[0]
                shifted = rewards + discount * terminal.logical_not() * (self._atoms - bonus)
                target_probs = categorical_projection(
                    chosen.exp(), shifted, float(self.cfg.critic_min_value), float(self.cfg.critic_max_value)
                )
                target_values = (target_probs * self._atoms).sum(dim=-1, keepdim=True)
            else:
                target_q = self._target_min(target_out, rows) - bonus
                target_values = rewards + discount * terminal.logical_not() * target_q

        if self._categorical:
            log_probs = F.log_softmax(values, dim=-1)
            cross_entropy = -(target_probs.unsqueeze(0) * log_probs).sum(dim=-1)  # (critics, R)
            critic_loss = cross_entropy.mean()
            per_agent = cross_entropy.detach().mean(dim=0).view(self.num_agents, rows).mean(dim=1)
            values = (log_probs.exp() * self._atoms).sum(dim=-1, keepdim=True)
            return critic_loss, values[0], values[1], target_values, outputs["log_prob"], per_agent
        # the mean over critics of each critic's MSE (they share the target)
        critic_loss = (values - target_values.unsqueeze(0)).pow(2).mean(dim=(1, 2)).mean()
        return critic_loss, values[0], values[1], target_values, outputs["log_prob"], None

    def _q_values(self, out: torch.Tensor) -> torch.Tensor:
        """The critics' output as Q values ``(critics, R, 1)``: as is, or a categorical
        critic's mean over its atoms."""
        if not self._categorical:
            return out
        return (torch.softmax(out, dim=-1) * self._atoms).sum(dim=-1, keepdim=True)

    def _join_halves(self, current: torch.Tensor, following: torch.Tensor, rows: int) -> torch.Tensor:
        """``cross_batch_value``: two flat ``(N * rows, d)`` batches as one ``(N * 2 rows, d)``,
        each agent's current rows then its next rows, so each agent's block stays its own."""
        n = self.num_agents
        return torch.cat((current.view(n, rows, -1), following.view(n, rows, -1)), dim=1).reshape(n * 2 * rows, -1)

    def _critic_half(self, out: torch.Tensor, rows: int, half: int) -> torch.Tensor:
        """``(critics, N * 2 rows, C) -> (critics, N * rows, C)``: half 0 (current) or 1 (next)."""
        members, _, columns = out.shape
        return out.view(members, self.num_agents, 2, rows, columns)[:, :, half].reshape(members, -1, columns)

    def _actor_half(self, out: torch.Tensor, rows: int) -> torch.Tensor:
        """``(N * 2 rows, ...) -> (N * rows, ...)``: the current half of a joined actor batch."""
        return out.view(self.num_agents, 2, rows, *out.shape[1:])[:, 0].reshape(self.num_agents * rows, *out.shape[1:])

    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, rows, next_inputs=None):
        """``alpha * log_pi - Q``, with Q the min (or mean, ``actor_q_reduction``) over the
        critics. ``cross_batch_value``: the actor runs on ``s ++ s'`` and the loss reads the
        current half (FlashSAC's update_actor)."""
        if self.cfg.reference_dropout > 0:
            inputs = {**inputs, "observations": self._drop_reference(inputs["observations"])}
            if next_inputs is not None:
                next_inputs = {**next_inputs, "observations": self._drop_reference(next_inputs["observations"])}
        if self.cfg.cross_batch_value:
            both = {
                **inputs,
                "observations": self._join_halves(inputs["observations"], next_inputs["observations"], rows),
            }
            with self._batch_mode(self.policy, True):
                actions, outputs = self.policy.act(both, role="policy")
            actions = self._actor_half(actions, rows)
            outputs = {
                key: self._actor_half(value, rows)
                if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 2 * self.num_agents * rows
                else value
                for key, value in outputs.items()
            }
        else:
            with self._batch_mode(self.policy, True):
                actions, outputs = self.policy.act(inputs, role="policy")
        log_prob = outputs["log_prob"]
        with self._batch_mode(self.critics, False):
            q_out, _ = self.critics.act(
                {**critic_inputs, "taken_actions": self._critic_actions(sampled, "observations", actions)}, role="critics"
            )
        q_pi = self._q_values(q_out)
        q = q_pi.mean(dim=0) if self.cfg.actor_q_reduction == "mean" else q_pi.min(dim=0).values
        entropy = self.expand_per_agent(self._entropy_coefficient, rows)
        q_term = q
        # demo_storage: memory_critic: the actor's rows are the online ones (1), never the demos (0)
        online = (1.0 - sampled["is_demo"]).reshape(-1, 1) if sampled is not None and "is_demo" in sampled else None
        if self.cfg.actor_q_scale is not None:
            # TD3+BC's lambda = scale / mean|Q|, per agent, no gradient through it
            if online is None:
                mean_abs = q.detach().abs().view(self.num_agents, rows).mean(dim=1).clamp(min=1e-6)
            else:
                weights = online.view(self.num_agents, rows)
                mean_abs = ((q.detach().abs().view(self.num_agents, rows) * weights).sum(dim=1)
                            / weights.sum(dim=1).clamp(min=1.0)).clamp(min=1e-6)
            q_term = q * self.expand_per_agent((float(self.cfg.actor_q_scale) / mean_abs).unsqueeze(-1), rows)
        if online is None:
            policy_loss = (entropy * log_prob - q_term).mean()
        else:
            per_row = (entropy * log_prob - q_term).view(self.num_agents, rows)
            weights = online.view(self.num_agents, rows)
            policy_loss = ((per_row * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)).mean()
        if self.cfg.action_l2_weight > 0:
            # action_l2_weight: the actor's mean action pulled toward zero (per-dim mean of its square)
            per_row = outputs["mean_actions"].pow(2).mean(dim=-1).view(self.num_agents, rows)
            if online is None:
                l2 = per_row.mean(dim=1)
            else:
                weights = online.view(self.num_agents, rows)
                l2 = (per_row * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            policy_loss = policy_loss + float(self.cfg.action_l2_weight) * l2.mean()
            outputs = {**outputs, "parts_metrics": {**outputs.get("parts_metrics", {}), "actor/action_l2": l2.detach()}}
        if getattr(self, "_track_outcomes", False):
            extra, metrics = self._parts_terms(sampled, outputs["mean_actions"], rows)
            policy_loss = policy_loss + extra
            # the Q term's size in the actor loss, beside the clone and anchor terms (each agent's batch mean)
            metrics = {**outputs.get("parts_metrics", {}), **metrics, "actor/q_term": q_term.detach().view(self.num_agents, rows).mean(dim=1)}
            if online is not None:
                metrics["actor/demo_rows"] = 1.0 - online.view(self.num_agents, rows).mean(dim=1)
            outputs = {**outputs, "parts_metrics": metrics}
        if self.cfg.demo_actor_clone and self._demo is not None:
            clone, value = self._demo_clone_term(rows)
            policy_loss = policy_loss + clone
            outputs = {**outputs, "parts_metrics": {**outputs.get("parts_metrics", {}), "actor/demo_clone_loss": value}}
        return policy_loss, actions, log_prob, q, outputs

    def _drop_reference(self, observations: torch.Tensor) -> torch.Tensor:
        """``reference_dropout``: the last ``reference_columns`` columns zeroed in that fraction of the rows."""
        columns = int(self.cfg.reference_columns)
        dropped = torch.rand(observations.shape[0], 1, device=observations.device) < float(self.cfg.reference_dropout)
        mask = torch.ones_like(observations)
        mask[:, -columns:] = (~dropped).to(observations.dtype)
        return observations * mask

    def _demo_clone_term(self, rows: int):
        """``demo_actor_clone``: PARTS' clone term on the demos, which are successes: ``success_clone_weight *
        mean ||a_mean(s_demo) - clip(a_demo, -1, 1)||^2`` over ``rows`` demo rows drawn per agent (each agent its
        own). Returns ``(loss, per-agent value)``."""
        demo_rows = next(iter(self._demo.values())).shape[0]
        index = torch.randint(0, demo_rows, (self.num_agents, rows), device=self.device).reshape(-1)
        observations = self.normalize_observations(self._demo["observations"][index])
        with self._batch_mode(self.policy, True):
            _, outputs = self.policy.act({"observations": observations}, role="policy")
        target = self._demo["actions"][index].clamp(-1.0, 1.0)
        per_row = (outputs["mean_actions"] - target).pow(2).sum(dim=-1).view(self.num_agents, rows)
        value = per_row.mean(dim=1)
        return float(self.cfg.success_clone_weight) * value.mean(), value.detach()

    def _parts_terms(self, sampled, mean_actions: torch.Tensor, rows: int):
        """PARTS' eq. 4 terms, per agent over its batch rows of finished episodes: ``success_clone_weight *
        mean_{B+} ||a_mean - a_stored||^2 + failure_anchor_weight * mean_{B-} ||a_mean||^2``, averaged over the
        agents (as the rest of the actor loss). Returns ``(loss, metrics)``, the metrics per agent ``(N,)``."""
        outcome = self._outcomes[sampled["episode_ids"].reshape(-1).long()].view(self.num_agents, rows)
        success = (outcome == 1.0).to(mean_actions.dtype)
        failure = (outcome == 0.0).to(mean_actions.dtype)
        for flag in ("is_demo", "_parts_skip"):  # demo rows: in neither term ...
            if flag in sampled:
                demo_row = sampled[flag].view(self.num_agents, rows).to(mean_actions.dtype)
                success, failure = success * (1.0 - demo_row), failure * (1.0 - demo_row)
                if flag == "_parts_skip" and self.cfg.clone_demo_rows:
                    success = success + demo_row  # ... or, clone_demo_rows, successes to clone
        stored = sampled["actions"]
        if not self.cfg.unsquashed_actor:
            # a squashed actor reaches [-1, 1]; only demo rows (demo_storage: memory) lie beyond it
            stored = stored.clamp(-1.0, 1.0)
        reduce = torch.mean if self.cfg.parts_term_reduction == "mean" else torch.sum
        clone = reduce((mean_actions - stored).pow(2), dim=-1).view(self.num_agents, rows)
        anchor = reduce(mean_actions.pow(2), dim=-1).view(self.num_agents, rows)
        clone_mean = (clone * success).sum(dim=1) / success.sum(dim=1).clamp(min=1.0)
        anchor_mean = (anchor * failure).sum(dim=1) / failure.sum(dim=1).clamp(min=1.0)
        loss = (float(self.cfg.success_clone_weight) * clone_mean + float(self.cfg.failure_anchor_weight) * anchor_mean).mean()
        metrics = {
            "actor/clone_loss": clone_mean.detach(),
            "actor/anchor_loss": anchor_mean.detach(),
            "actor/success_rows": success.mean(dim=1).detach(),
            "actor/failure_rows": failure.mean(dim=1).detach(),
        }
        return loss, metrics

    #: the critic's own input for each actor-side key, when the learner is asymmetric
    _CRITIC_INPUT_KEYS = {"observations": "states", "next_observations": "next_states"}

    def _critic_actions(self, sampled, key: str, actions: torch.Tensor) -> torch.Tensor:
        """What the critics take as the action: the agent's own, or :attr:`critic_action_transform`
        of it with the batch's raw critic input (device ops only, so a CUDA graph captures it).
        ``key`` names the actor-side tensor (``observations`` / ``next_observations``); an
        asymmetric learner hands the transform the critic's ``states`` / ``next_states`` of the
        same rows instead, online and demo rows alike, so it can read what is in the critic state
        and not in the actor's observation. ``sampled`` is read only when there is a transform."""
        if self.critic_action_transform is None:
            return actions
        side_names = getattr(self, "_side_inputs", {})
        declared = set(getattr(self.critic_action_transform, "side_inputs", None) or {})
        if declared != set(side_names):
            raise RuntimeError(
                f"the critic action transform reads side inputs {sorted(declared)} but the memory stores "
                f"{sorted(side_names)}: set learner.critic_action_transform before init()"
            )
        prefix = "next_" if key == "next_observations" else ""
        if self._asymmetric:
            key = self._CRITIC_INPUT_KEYS[key]
        if side_names:
            side = {name: sampled[f"{prefix}{name}"] for name in side_names}
            out = self.critic_action_transform(sampled[key], actions, side)
        else:
            out = self.critic_action_transform(sampled[key], actions)
        width = int(getattr(self.critic_action_transform, "critic_action_width", None) or actions.shape[-1])
        if out.shape != (*actions.shape[:-1], width):
            raise ValueError(
                f"the critic action transform returned {tuple(out.shape)} for actions {tuple(actions.shape)}; "
                f"expected width {width}"
            )
        return out

    # ------------------------------------------------------------------ update
    def update(self, *, timestep: int, timesteps: int) -> None:
        with self.update_timer():  # publishes stats/update_time_ms with the rest
            self._update(timestep=timestep, timesteps=timesteps)

    def _update(self, *, timestep: int, timesteps: int) -> None:
        rows = self.cfg.batch_size  # per agent; the memory returns N * batch_size rows
        if (self.cfg.store_labels or self.cfg.route_label) and int(self.memory.size) == 0:
            return  # no transition of a stored phase yet for some agent: nothing to learn from
        # warmup_trains_critics: the actor stays frozen until the warm-up episodes have ended
        policy_on = timestep >= self._policy_learning_starts() and not self._warming_up()
        if policy_on and self.cfg.demo_storage == "memory_critic" and not self._online_rows_available():
            policy_on = False  # only demos so far: the critics train, the actor waits for its own data
        demo_on = self._demo_active(timestep)
        # a graph replays fixed shapes: a mixed batch's online share is fixed only once the
        # memory holds half a batch; aux losses are captured only when every term is graph-safe
        # (losses.terms[*].graph_safe), the others run eagerly. Per-agent fills (store_labels,
        # route_label, demos in the memory) sample with device-side bounds (sync_size); a
        # memory_critic memory that has wrapped samples its online rows by flag, eagerly
        graph = (
            self._graphs is not None
            and self.aux_loss_graph_safe()
            and (not demo_on or int(self.memory.size) >= rows // 2)
            and not (self.cfg.demo_storage == "memory_critic" and self._memory_wrapped())
        )
        if graph:
            self.memory.sync_size()

        last_critic, last_actor = {}, {}
        for gradient_step in range(self.cfg.gradient_steps):
            self._set_learning_rates(timesteps)
            # the actor (and temperature) step on every policy_delay-th gradient step
            actor_step = policy_on and (self._updates + 1) % self.cfg.policy_delay == 0
            mode = (demo_on, policy_on, actor_step)
            if graph:
                out = self._graphs.run(mode, lambda: self._gradient_step(mode, rows, timestep, graph=True))
            else:
                out = self._gradient_step(mode, rows, timestep, graph=False)
            self._updates += 1
            last_critic = out["critic"]
            if actor_step:
                last_actor = out["actor"]

        if self.on_log:
            # once per update, from its last gradient step (the actor's from its last step in
            # this update), not after every gradient step: emitting costs kernel launches and
            # a memory query, and consecutive gradient steps' values are alike
            self._emit_update_metrics(
                timestep=timestep,
                rows=rows,
                demo_rows=(rows - min(rows // 2, int(self.memory.size))) if demo_on else 0,
                **last_critic,
                **last_actor,
            )

    def _gradient_step(self, mode, rows: int, timestep: int, graph: bool) -> Dict[str, Dict[str, torch.Tensor]]:
        """One critic step, the actor and temperature step when ``mode`` says so, the targets'
        Polyak step. Device work only (no host sync, every state written in place), so it can
        be captured as a CUDA graph. Returns the tensors the metrics are built from."""
        demo_on, policy_on, actor_step = mode
        if self.cfg.demo_storage == "memory_critic":
            # the critics sample the whole memory, demos included; the actor its own full batch of online rows
            sampled = self._sample(rows, graph)
            online = self._sample_online(rows, graph) if actor_step else sampled
        else:
            # outside the demo window one online batch serves both losses, as in classic SAC;
            # inside it the critic's batch is mixed and the actor's stays online -- or, with
            # demo_batch_shared, the actor takes the same mixed batch (RLPD)
            if demo_on and self.cfg.demo_batch_shared:
                sampled = self._mixed_batch(rows, graph)
                online = sampled
            else:
                online = self._sample(rows, graph) if (policy_on or not demo_on) else None
                sampled = self._mixed_batch(rows, graph) if demo_on else online
        inputs, next_inputs, critic_inputs, critic_next_inputs = self._batch_inputs(sampled)

        critic_loss, critic_1_values, critic_2_values, target_values, next_log_prob, critic_ce = self._compute_critic_loss(
            sampled=sampled,
            inputs=inputs,
            next_inputs=next_inputs,
            critic_inputs=critic_inputs,
            critic_next_inputs=critic_next_inputs,
            rows=rows,
        )
        ctx = LossContext(
            learner=self,
            step=timestep,
            target="critic",
            sampled=sampled,
            critic_inputs=critic_inputs,
            critic_1_values=critic_1_values,
            critic_2_values=critic_2_values,
            target_values=target_values,
        )
        aux = self.compute_aux_loss(ctx)
        if aux is not None:
            critic_loss = critic_loss + aux

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norms = clip_grad_norm_per_agent(
            self.critics, self.num_agents, self.cfg.grad_norm_clip
        )
        self.critic_optimizer.step()
        self._normalize_weights(self.critics)

        actor = {}
        if actor_step:
            if online is not sampled:
                # a second batch this step: it must not count twice in the normalizer
                inputs, next_inputs, critic_inputs, _ = self._batch_inputs(online, train=False)
            actor = self._actor_step(
                timestep=timestep, sampled=online, inputs=inputs, critic_inputs=critic_inputs, rows=rows,
                next_log_prob=next_log_prob, next_inputs=next_inputs,
            )

        self._polyak()
        critic = dict(
            critic_1_values=critic_1_values.detach(),
            critic_2_values=critic_2_values.detach(),
            target_values=target_values,
            critic_grad_norms=critic_grad_norms,
            critic_loss=critic_ce,
            # the aux terms' raw values: returned like the rest, so a replayed graph's are fresh
            aux_metrics={k: v.detach() for k, v in ctx.aux_metrics.items()},
        )
        return {"critic": critic, "actor": actor}

    def _batch_inputs(self, sampled, train: bool = True):
        """The (normalized) actor and critic inputs of a sampled batch; ``train`` updates the
        normalizers' running stats with it."""
        inputs = {"observations": self.normalize_observations(sampled["observations"], train=train)}
        next_inputs = {
            "observations": self.normalize_observations(sampled["next_observations"], train=train)
        }
        if self._asymmetric:
            critic_inputs = {"observations": self.normalize_states(sampled["states"], train=train)}
            critic_next_inputs = {
                "observations": self.normalize_states(sampled["next_states"], train=train)
            }
        else:
            critic_inputs, critic_next_inputs = inputs, next_inputs
        return inputs, next_inputs, critic_inputs, critic_next_inputs

    def _actor_step(
        self, *, timestep: int, sampled, inputs, critic_inputs, rows: int, next_log_prob=None, next_inputs=None
    ) -> dict:
        """The actor's step, then the temperature's on the entropy ``entropy_observations``
        names: this batch's ``log_prob`` or the critic step's ``next_log_prob``."""
        policy_loss, actions, log_prob, q_pi, outputs = self._compute_actor_loss(
            sampled=sampled, inputs=inputs, critic_inputs=critic_inputs, rows=rows, next_inputs=next_inputs
        )
        ctx = LossContext(
            learner=self,
            step=timestep,
            target="policy",
            sampled=sampled,
            inputs=inputs,
            actions=actions,
            log_prob=log_prob,
            policy_outputs=outputs,
        )
        aux = self.compute_aux_loss(ctx)
        if aux is not None:
            policy_loss = policy_loss + aux

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norms = clip_grad_norm_per_agent(
            self.policy, self.num_agents, self.cfg.grad_norm_clip
        )
        self.policy_optimizer.step()
        self._normalize_weights(self.policy)

        if self.cfg.learn_entropy:
            entropy_log_prob = next_log_prob if self.cfg.entropy_observations == "next" else log_prob
            if entropy_log_prob is None:
                raise RuntimeError("entropy_observations: next needs the critic step's next_log_prob")
            entropy_loss = self._update_entropy(entropy_log_prob, rows)
        else:
            entropy_loss = None
        with torch.no_grad():
            per_agent_loss = self.per_agent_mean(
                self.expand_per_agent(self._entropy_coefficient, rows) * log_prob - q_pi, rows
            )
        return {
            "policy_grad_norms": policy_grad_norms,
            "log_prob": log_prob.detach(),
            "policy_loss": per_agent_loss,
            "log_std": outputs["log_std"].detach() if "log_std" in outputs else None,
            "entropy_loss": entropy_loss,
            "actor_aux_metrics": {
                **{k: v.detach() for k, v in ctx.aux_metrics.items()},
                **(outputs.get("parts_metrics") or {}),
            },
        }

    def _update_entropy(self, log_prob: torch.Tensor, rows: int) -> torch.Tensor:
        """Per-agent temperature step: each agent's loss term holds only its own rows. Returns
        the temperature's loss per agent."""
        log_prob_per_agent = log_prob.view(self.num_agents, rows, 1).mean(dim=1)  # (N, 1)
        if self.cfg.entropy_loss_form == "alpha":
            # the multiplicative form: alpha * (H_hat - H_target); the adaptation rate is ~alpha
            entropy_loss = torch.exp(self.log_entropy_coefficient) * (
                -log_prob_per_agent - self._target_entropy
            ).detach()
        else:
            # skrl's form: -log_alpha * (log_pi + H_target)
            entropy_loss = -(
                self.log_entropy_coefficient * (log_prob_per_agent + self._target_entropy).detach()
            )
        self.entropy_optimizer.zero_grad()
        entropy_loss.sum().backward()  # a sum of per-agent terms: no cross-agent gradient
        self.entropy_optimizer.step()
        with torch.no_grad():
            # in place: a captured graph keeps reading this tensor
            self._entropy_coefficient.copy_(torch.exp(self.log_entropy_coefficient))
        return entropy_loss.detach().reshape(-1)

    def _emit_update_metrics(
        self,
        *,
        timestep: int,
        rows: int,
        critic_1_values,
        critic_2_values,
        target_values,
        critic_grad_norms,
        critic_loss=None,
        demo_rows: int = 0,
        policy_grad_norms=None,
        log_prob=None,
        policy_loss=None,
        log_std=None,
        entropy_loss=None,
        aux_metrics=None,
        actor_aux_metrics=None,
    ) -> None:
        """One point per update: the critic's last gradient step, the actor's last step (with
        graph-safe aux losses, their terms' raw values too, from the same steps)."""
        with torch.no_grad():
            metrics = {
                # categorical: the cross-entropy it trains on; scalar: the halved MSE sum
                "loss/critic": critic_loss if critic_loss is not None else 0.5
                * (
                    F.mse_loss(
                        critic_1_values.view(self.num_agents, rows, -1),
                        target_values.view(self.num_agents, rows, -1),
                        reduction="none",
                    ).mean(dim=(1, 2))
                    + F.mse_loss(
                        critic_2_values.view(self.num_agents, rows, -1),
                        target_values.view(self.num_agents, rows, -1),
                        reduction="none",
                    ).mean(dim=(1, 2))
                ),
                "q/q1_mean": self.per_agent_mean(critic_1_values, rows),
                "q/q2_mean": self.per_agent_mean(critic_2_values, rows),
                "q/target_mean": self.per_agent_mean(target_values, rows),
                "grad_norm/critic": critic_grad_norms,
                "entropy/coefficient": self._entropy_coefficient.reshape(-1),
                "lr/actor": self.policy_optimizer.lr.to(torch.float32),
                "lr/critic": self.critic_optimizer.lr.to(torch.float32),
            }
            if self.cfg.route_label:
                # how much practice this residual has had: the agent steps that started in its phase (an agent step is
                # residual.steps_per_chunk env steps), its replay rows (demos included) and its gradient updates
                metrics["route/steps"] = torch.full((self.num_agents,), float(getattr(self, "_route_steps", 0)), device=self.device)
                sizes = self.memory._agent_size or [int(self.memory.size)] * self.num_agents
                metrics["route/memory_rows"] = torch.tensor([float(v) for v in sizes], device=self.device)
                metrics["route/updates"] = torch.full((self.num_agents,), float(self._updates), device=self.device)
            if self.return_scaler is not None:
                # what the critics' rewards are divided by (adaptive_reward_scaling)
                metrics["reward/scale_denominator"] = self.return_scaler.denominator.reshape(-1).clone()
            if self._demo is not None:
                # demo rows per critic batch, as a fraction: 0 outside the window
                metrics["sac/demo_fraction"] = torch.full(
                    (self.num_agents,), demo_rows / rows, device=self.device
                )
            if policy_grad_norms is not None:
                metrics["grad_norm/policy"] = policy_grad_norms
            if log_prob is not None:
                metrics["policy/log_prob"] = self.per_agent_mean(log_prob, rows)
            if policy_loss is not None:
                # the actor's own objective: alpha * log_pi - min(Q1, Q2), per agent
                metrics["loss/policy"] = policy_loss
            if entropy_loss is not None:
                # the temperature's loss, i.e. how hard alpha is being pushed
                metrics["loss/entropy"] = entropy_loss
            if log_std is not None:
                metrics["policy/std"] = log_std.exp().view(self.num_agents, rows, -1).mean(dim=(1, 2))
            if self.aux_loss_graph_safe():  # otherwise the hook emitted them every gradient step
                metrics.update(aux_metrics or {})
                metrics.update(actor_aux_metrics or {})
            metrics.update(self.stats_metrics())
            self.emit_per_agent(metrics, timestep)

    # ------------------------------------------------------------------ SimBa periodic reset
    def _maybe_retrain(self, timestep: int, timesteps: int, logged_at: Optional[int] = None) -> None:
        """``retrain_every``: PARTS' success-reweighted retraining (see the field's doc). ``timestep`` is the clock it
        counts (``route_clock``: this residual's own steps), ``logged_at`` the env step its metric goes out at."""
        every = int(self.cfg.retrain_every)
        if not every or timestep <= 0 or timestep % every or int(self.memory.size) == 0:
            return
        if logged_at is not None:
            print(f"[sac] {self.cfg.route_label}: retraining at its own step {timestep} (env step {logged_at})", flush=True)
            timestep = logged_at
        size = max(self.memory._agent_size) if self.memory._agent_size is not None else int(self.memory.size)
        ids = self.memory.get_tensor_by_name("episode_ids")[:, :size, 0].long()
        outcome = self._outcomes[ids.clamp(min=0)]
        # each failed episode is kept whole or dropped whole, drawn once per episode id
        draw = torch.rand(self._outcomes.numel(), device=self.device) < float(self.cfg.retrain_failure_fraction)
        keep = (outcome == 1.0) | (outcome == -1.0) | ((outcome == 0.0) & draw[ids.clamp(min=0)])
        before = [int(n) for n in (self.memory._agent_size or [self.memory.size] * self.num_agents)]
        kept = self.memory.keep_rows(keep)
        print(f"[sac] retrain at step {timestep}: memory {before} -> {kept} rows (every success, "
              f"{self.cfg.retrain_failure_fraction:.0%} of the failed episodes); fresh networks, "
              f"{self.cfg.retrain_gradient_steps} offline updates", flush=True)
        self._periodic_reset()  # fresh networks (weight-normalized as at build), optimizers and temperature
        if self.memory.size == 0:
            return
        rows = self.cfg.batch_size
        self.enable_models_training_mode(True)
        for _ in range(int(self.cfg.retrain_gradient_steps)):
            self._set_learning_rates(timesteps)
            actor_step = (self._updates + 1) % self.cfg.policy_delay == 0
            demo_on = self._demo_active(timestep)
            self._gradient_step((demo_on, True, actor_step), rows, timestep, graph=False)
            self._updates += 1
        self.enable_models_training_mode(False)
        self.emit_per_agent(
            {"retrain/kept_rows": torch.tensor(kept, dtype=torch.float32, device=self.device)}, timestep
        )

    def _maybe_periodic_reset(self, timestep: int) -> None:
        cfg = self.cfg
        if not cfg.periodic_reset_enabled or timestep <= 0:
            return
        if cfg.periodic_reset_max > 0 and self._n_periodic_resets >= cfg.periodic_reset_max:
            return
        if timestep % cfg.periodic_reset_frequency == 0:
            self._periodic_reset()
            self._n_periodic_resets += 1

    def _periodic_reset(self) -> None:
        """SimBa's hard reset: fresh networks, optimizers and temperature; keep the buffer.

        Mirrors SimBa (arXiv:2410.09754 §7.3): reinitialize the whole network and optimizer,
        keeping the replay buffer and the normalizer statistics.
        """
        if self._model_cfg is None:
            raise RuntimeError(
                "sac.periodic_reset_enabled needs model_cfg passed to the learner so models can "
                "be rebuilt"
            )
        fresh = build_models(
            "sac",
            self._model_cfg,
            self.observation_space,
            self.state_space,
            self.action_space,
            self.num_agents,
            self.device,
            self._controller_cfg,
            num_critics=self._num_critics,
            critic_outputs=sac_critic_outputs(self.cfg),
            critic_action_space=self._critic_action_space,
        )
        for key in ("policy", "critics", "target_critics"):
            self.models[key] = fresh[key]
        self.policy = self.models["policy"]
        self._bind_critics()
        self._normalize_weights(self.policy, self.critics)
        self._sync_targets()

        self._build_optimizers()
        self._entropy_coefficient = torch.full(
            (self.num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            self.log_entropy_coefficient = torch.log(
                self._entropy_coefficient.clone()
            ).requires_grad_(True)
            self.entropy_optimizer = BlockAdamW(
                [self.log_entropy_coefficient],
                self.num_agents,
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_model_keys(self) -> List[str]:
        # the target critics are saved too: resuming must keep their EMA lag, and loading one
        # slot must not touch any other agent's targets
        return ["policy", "critics", "target_critics"]

    def _upgrade_checkpoint_models(self, models: Dict[str, Any], path) -> Dict[str, Any]:
        """Checkpoints written before the critics were stacked hold ``critic_1 .. critic_N``
        and ``target_critic_1 .. N`` as separate plain state dicts: stack them member-wise."""
        if "critics" in models:
            for key in ("critics", "target_critics"):
                stacked = next(iter(models.get(key, {}).values()), None)
                if stacked is not None and stacked.shape[0] != self._num_critics:
                    raise ValueError(
                        f"{path} stacks {stacked.shape[0]} critics in '{key}', but "
                        f"sac.num_critics is {self._num_critics}"
                    )
            return models
        if "critic_1" not in models:
            return models
        upgraded = {k: v for k, v in models.items() if not k.startswith(("critic_", "target_critic_"))}
        for prefix, key in (("critic_", "critics"), ("target_critic_", "target_critics")):
            found = sorted(
                (int(k[len(prefix):]), v) for k, v in models.items()
                if k.startswith(prefix) and k[len(prefix):].isdigit()
            )
            if [i for i, _ in found] != list(range(1, self._num_critics + 1)):
                raise KeyError(
                    f"{path} holds {[f'{prefix}{i}' for i, _ in found]}, but sac.num_critics is "
                    f"{self._num_critics}"
                )
            upgraded[key] = {name: torch.stack([v[name] for _, v in found]) for name in found[0][1]}
        return upgraded

    def _checkpoint_optimizer_keys(self) -> List[str]:
        return ["policy_optimizer", "critic_optimizer"]

    def _checkpoint_normalizers(self) -> Dict[str, BlockRunningNorm]:
        normalizers = {}
        if self.observation_normalizer is not None:
            normalizers["observations"] = self.observation_normalizer
        if self.state_normalizer is not None:
            normalizers["states"] = self.state_normalizer
        if self.reward_normalizer is not None:
            normalizers["rewards"] = self.reward_normalizer
        if self.return_scaler is not None:
            normalizers["returns"] = self.return_scaler
        return normalizers

    def _checkpoint_extras(self, agent: int) -> Dict[str, Any]:
        return {
            "entropy_coefficient": self._entropy_coefficient[agent].detach().clone().cpu(),
            "log_entropy_coefficient": (
                self.log_entropy_coefficient.detach()[agent].clone().cpu()
                if self.cfg.learn_entropy
                else None
            ),
        }

    def _load_extras(self, agent: int, extras: Dict[str, Any], path) -> None:
        missing = {"entropy_coefficient", "log_entropy_coefficient"} - set(extras)
        if missing:
            raise KeyError(f"{path} is missing learner extras: {sorted(missing)}")
        saved_log = extras["log_entropy_coefficient"]
        if self.cfg.learn_entropy and saved_log is None:
            raise ValueError(
                f"learn_entropy is on but {path} was saved with it off (no log_entropy_coefficient)"
            )
        if not self.cfg.learn_entropy and saved_log is not None:
            raise ValueError(
                f"learn_entropy is off but {path} was saved with it on (log_entropy_coefficient set)"
            )
        with torch.no_grad():
            self._entropy_coefficient[agent].copy_(
                extras["entropy_coefficient"].to(self.device)
            )
            if self.cfg.learn_entropy:
                self.log_entropy_coefficient.data[agent].copy_(saved_log.to(self.device))

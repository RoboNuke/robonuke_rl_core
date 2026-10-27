"""FlashSAC's network (``model.architecture: flashsac``), beside SimBa and the MLP.

Ported from Holiday-Robot/FlashSAC (arXiv:2604.04539) ``flash_rl/agents/flashSAC/layer.py`` and
``network.py``, in full precision (never its mixed precision). Configured in ``models/cfg.py``
(``FlashSACActorCfg`` / ``FlashSACCriticCfg``):

* an **embedder**: BatchNorm on the raw input, then a bias-free Linear to the hidden width;
* **inverted residual blocks**: ``x + ReLU(BN(W2 ReLU(BN(W1 x))))``, ``W1`` expanding the width
  ``expansion`` times (4), BatchNorm before each nonlinearity, bias-free Linears;
* a **post RMSNorm**, then the head: the actor's mean and log std (Linear weights with free biases,
  ``log_std = min + 0.5 (max - min) (1 + tanh(raw))`` on [-10, 2]), tanh-squashed; the Q critic's
  ``num_bins`` logits of a categorical value (``sac.critic_distribution: categorical``).

Linears start orthogonal with gain 1 (every row a unit vector); the learner's
``sac.weight_normalization`` keeps them there, and the normalization layers' affine at norm
sqrt(width).

**BatchNorm under vmap.** :class:`~robonuke_rl_core.models.ensemble.VmapEnsemble` forbids in-place
buffer writes inside ``forward``, so a :class:`FlashBatchNorm` normalizes with the batch's own
statistics in training mode and *returns* them; the plain nets return them after their outputs and
:func:`run` folds them into the stacked running buffers after the call, FlashSAC's update
(``running = lerp(running, batch, momentum)``, the variance unbiased). Each agent (and each member
of a Q ensemble) normalizes over its own rows only. In eval mode the running statistics are used and
nothing is written. Which calls run in which mode is the learner's (FlashSAC's): the actor trains on
its loss batch and acts in eval mode, the critics train on their loss batch (the targets too) and are
evaluated in eval mode for the actor's Q (``SAC._batch_mode``).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch.distributions import Normal

from .ensemble import VmapEnsemble
from .simba import EnsembleModel, safe_atanh, squash_log_prob_correction

__all__ = ["FlashBatchNorm", "FlashSACActorNet", "FlashSACQCriticNet", "FlashSACActor", "FlashSACQCritic", "run"]


class FlashBatchNorm(nn.Module):
    """FlashSAC's ``UnitBatchNorm``: BatchNorm over the batch dim with momentum 0.01, eps 1e-5, its
    affine (scale, shift) kept at norm sqrt(width) by weight normalization (``normalize_affine``).
    ``forward`` returns ``(y, (batch_mean, batch_var_unbiased))`` in training mode and ``(y, None)``
    in eval mode; it never writes its buffers (see the module docstring)."""

    #: VmapEnsemble.normalize_weights treats this affine as a normalization layer's
    normalize_affine = True

    def __init__(self, width: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.register_buffer("running_mean", torch.zeros(width))
        self.register_buffer("running_var", torch.ones(width))
        self.momentum = float(momentum)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, correction=0)
            y = (x - mean) * torch.rsqrt(var + self.eps)
            rows = x.shape[0]
            stats = (mean, var * (rows / max(rows - 1, 1)))
        else:
            y = (x - self.running_mean) * torch.rsqrt(self.running_var + self.eps)
            stats = None
        return y * self.weight + self.bias, stats


def _linear(width_in: int, width_out: int, bias: bool = False) -> nn.Linear:
    """FlashSAC's ``UnitLinear``: orthogonal, gain 1 (unit rows), no bias unless a head's."""
    layer = nn.Linear(width_in, width_out, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=1.0)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class _Trunk(nn.Module):
    """Embedder -> ``num_blocks`` inverted residual blocks -> post RMSNorm. ``forward`` returns
    ``(features, stats)``, ``stats`` the batch statistics of every BatchNorm in :attr:`bn_names`
    order (empty in eval mode)."""

    def __init__(self, in_dim: int, hidden: int, num_blocks: int, expansion: int = 4) -> None:
        super().__init__()
        self.embed_norm = FlashBatchNorm(in_dim)
        self.embed = _linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "w1": _linear(hidden, hidden * expansion),
                        "norm1": FlashBatchNorm(hidden * expansion),
                        "w2": _linear(hidden * expansion, hidden),
                        "norm2": FlashBatchNorm(hidden),
                    }
                )
                for _ in range(num_blocks)
            ]
        )
        self.post_norm = nn.RMSNorm(hidden, eps=1e-6)

    def bn_names(self, prefix: str) -> List[str]:
        names = [f"{prefix}embed_norm"]
        for i in range(len(self.blocks)):
            names += [f"{prefix}blocks.{i}.norm1", f"{prefix}blocks.{i}.norm2"]
        return names

    def forward(self, x: torch.Tensor):
        stats = []
        x, s = self.embed_norm(x)
        stats.append(s)
        x = self.embed(x)
        for block in self.blocks:
            residual = x
            h, s1 = block["norm1"](block["w1"](x))
            h = F.relu(h)
            h, s2 = block["norm2"](block["w2"](h))
            x = F.relu(h) + residual
            stats += [s1, s2]
        return self.post_norm(x), stats


def _flat_stats(stats) -> tuple:
    """``[(mean, var) | None]`` -> a flat tuple of tensors (empty in eval mode)."""
    return tuple(t for pair in stats if pair is not None for t in pair)


class FlashSACActorNet(nn.Module):
    """FlashSAC's actor: trunk -> mean and raw log std heads. Returns ``(mean, log_std, *stats)``."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, num_blocks: int, expansion: int,
                 min_log_std: float, max_log_std: float, mean_init_scale: float = 1.0, init_std=None) -> None:
        super().__init__()
        self.trunk = _Trunk(obs_dim, hidden, num_blocks, expansion)
        self.mean = _linear(hidden, action_dim, bias=True)
        if mean_init_scale != 1.0:
            # a residual's near-zero start (Policy Decorator's 0.01 output init): the mean head starts at this
            # orthogonal gain and stays out of weight normalization, which would set its rows back to unit length
            nn.init.orthogonal_(self.mean.weight, gain=float(mean_init_scale))
            self.mean.skip_weight_norm = True
        self.log_std = _linear(hidden, action_dim, bias=True)
        if init_std is not None:
            # every state starts at this std: the (state-dependent) log std head's weights start at gain 0.01, out of
            # weight normalization, and its bias at the raw value the tanh bound maps to log(init_std)
            unit = 2.0 * (math.log(float(init_std)) - float(min_log_std)) / (float(max_log_std) - float(min_log_std)) - 1.0
            with torch.no_grad():
                nn.init.orthogonal_(self.log_std.weight, gain=0.01)
                self.log_std.bias.fill_(math.atanh(unit))
            self.log_std.skip_weight_norm = True
        self.min_log_std, self.max_log_std = float(min_log_std), float(max_log_std)

    def bn_names(self) -> List[str]:
        return self.trunk.bn_names("trunk.")

    def forward(self, observations: torch.Tensor):
        h, stats = self.trunk(observations)
        raw = self.log_std(h)
        log_std = self.min_log_std + (self.max_log_std - self.min_log_std) * 0.5 * (1.0 + torch.tanh(raw))
        return (self.mean(h), log_std) + _flat_stats(stats)


class FlashSACQCriticNet(nn.Module):
    """FlashSAC's critic: trunk on ``[s, a]`` -> ``outputs`` logits (a categorical value's bins; 1 for a
    scalar Q). Returns ``(out, *stats)``."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int, num_blocks: int, expansion: int, outputs: int) -> None:
        super().__init__()
        self.trunk = _Trunk(obs_dim + action_dim, hidden, num_blocks, expansion)
        self.out = _linear(hidden, outputs, bias=True)

    def bn_names(self) -> List[str]:
        return self.trunk.bn_names("trunk.")

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        h, stats = self.trunk(torch.cat((observations, actions), dim=-1))
        return (self.out(h),) + _flat_stats(stats)


def run(net: VmapEnsemble, n_out: int, training: bool, *inputs: torch.Tensor):
    """Call a FlashSAC ensemble on flat ``(num_agents * rows, ...)`` inputs in ``training`` mode and
    return its ``n_out`` outputs flat (``(members, num_agents * rows, ...)`` with members). In training
    mode the BatchNorms' batch statistics that come back after them are folded into the stacked
    running buffers (``lerp(running, batch, momentum)``), each agent's and member's from its own rows."""
    net._meta.train(training)
    rows = net._rows(inputs[0])
    out = net.forward(*[net.split(t, rows) for t in inputs])
    main, stats = out[:n_out], out[n_out:]
    if training:
        _, buffers = net._stacked()
        names = net._meta.bn_names()
        if len(stats) != 2 * len(names):
            raise RuntimeError(f"expected batch statistics of {len(names)} BatchNorms, got {len(stats)} tensors")
        with torch.no_grad():
            for i, name in enumerate(names):
                momentum = net._meta.get_submodule(name).momentum
                buffers[f"{name}.running_mean"].lerp_(stats[2 * i].detach(), momentum)
                buffers[f"{name}.running_var"].lerp_(stats[2 * i + 1].detach(), momentum)
    flatten = net.flatten if net.members == 1 else net._flatten_members
    return tuple(flatten(t) for t in main)


class FlashSACActor(EnsembleModel, GaussianMixin, Model):
    """FlashSAC's tanh-squashed Gaussian policy over ``num_agents`` agents (``act()`` as the other
    actors'). Its BatchNorms use batch statistics while the model is in training mode
    (``enable_training_mode``), the running ones otherwise."""

    #: the learner sets this model's mode call by call (SAC._batch_mode)
    uses_batch_stats = True

    def __init__(self, observation_space, action_space, device, num_agents: int = 1, hidden_size: int = 128,
                 num_blocks: int = 2, expansion: int = 4, min_log_std: float = -10.0, max_log_std: float = 2.0,
                 mean_init_scale: float = 1.0, init_std=None) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=False, reduction="sum")
        self.num_agents = num_agents
        self.num_continuous = self.num_actions
        self._distribution = None

        def build() -> nn.Module:
            return FlashSACActorNet(self.num_observations, self.num_actions, int(hidden_size), int(num_blocks),
                                    int(expansion), float(min_log_std), float(max_log_std), float(mean_init_scale), init_std)

        self.net = VmapEnsemble(build, num_agents, device=device)

    def compute(self, inputs, role):
        mean, log_std = run(self.net, 2, self.training, inputs["observations"])
        return mean, {"log_std": log_std}

    def act(self, inputs, *, role: str = ""):
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        distribution = Normal(mean, log_std.exp())
        self._distribution = distribution
        taken, noise = inputs.get("taken_actions"), inputs.get("noise")
        if taken is not None:
            u = safe_atanh(taken)
        elif noise is not None:  # sac.noise_repeat's held draw
            u = mean + log_std.exp() * noise
        else:
            u = distribution.rsample()
        outputs["log_prob"] = distribution.log_prob(u).sum(dim=-1, keepdim=True) - squash_log_prob_correction(u).unsqueeze(-1)
        outputs["mean_actions"] = torch.tanh(mean)
        return torch.tanh(u), outputs

    def get_entropy(self, *, role: str = ""):
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)


class FlashSACQCritic(EnsembleModel, DeterministicMixin, Model):
    """FlashSAC's critic over ``num_agents`` agents, ``members`` per agent (``compute`` returns
    ``(members, num_agents * rows, outputs)``)."""

    uses_batch_stats = True

    def __init__(self, observation_space, action_space, device, num_agents: int = 1, hidden_size: int = 256,
                 num_blocks: int = 2, expansion: int = 4, members: int = 1, outputs: int = 1) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_agents = num_agents
        self.members = members
        self.outputs = outputs

        def build() -> nn.Module:
            return FlashSACQCriticNet(self.num_observations, self.num_actions, int(hidden_size), int(num_blocks),
                                      int(expansion), int(outputs))

        self.net = VmapEnsemble(build, num_agents, device=device, members=members)

    def compute(self, inputs, role):
        (out,) = run(self.net, 1, self.training, inputs["observations"], inputs["taken_actions"])
        return out, {}

"""A plain MLP architecture (``model.architecture: mlp``), beside SimBa.

Linear -> activation stacks for the actor, the Q critics and the value critic, built the way
the residual-RL papers build theirs and configured in ``models/cfg.py``
(``MlpActorCfg`` / ``MlpCriticCfg``):

* **ResiP** (Ankile et al., 2024, https://arxiv.org/abs/2407.15403): 512 x 2 SiLU actor and
  critic, kaiming hidden layers, an orthogonal output layer with gain 0.01 on the actor's
  mean, a plain (unsquashed) Gaussian whose state-independent log std starts at -1.0 and is
  never learned (``init: kaiming``, ``output_init_scale: 0.01``, ``squash: false``,
  ``init_log_std: -1.0``, ``learn_std: false``).
* **Policy Decorator** (Yuan et al., 2024, https://arxiv.org/abs/2412.13630): 3 x 256 ReLU
  SAC actor and critics, a state-dependent log std bounded softly,
  ``min + 0.5 (max - min) (tanh(x) + 1)`` on [-20, 2] (``use_state_dependent_std: true``,
  ``log_std_bound: tanh``).

Same two layers as SimBa: ``MlpActorNet`` / ``MlpQCriticNet`` / ``MlpValueCriticNet`` are
plain single-agent ``nn.Module``s returning raw tensors (no sampling, no in-place buffer
writes, so :class:`~robonuke_rl_core.models.ensemble.VmapEnsemble` can vmap them), and
``MlpActor`` / ``MlpQCritic`` / ``MlpValueCritic`` are the skrl models the learners call,
with the same ``act()`` contract SimBa's wrappers keep (``log_prob``, ``mean_actions``,
``log_std``; ``get_entropy``). Continuous actions only: a controller with a selection block
(Bernoulli dims, ``match``) needs SimBa, and the actor says so rather than guessing.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch.distributions import Normal

from .ensemble import VmapEnsemble
from .simba import EnsembleModel, safe_atanh, squash_log_prob_correction

__all__ = [
    "INITS",
    "LOG_STD_BOUNDS",
    "activation_module",
    "check_mlp_actor_fields",
    "check_mlp_critic_fields",
    "MlpActorNet",
    "MlpQCriticNet",
    "MlpValueCriticNet",
    "MlpActor",
    "MlpQCritic",
    "MlpValueCritic",
]

#: hidden-layer initializations: ``orthogonal`` (gain sqrt(2), the CleanRL / Policy Decorator
#: convention) or ``kaiming`` (``kaiming_normal_`` fan-in, ResiP's ``layer_init``); biases are
#: zero, and the output layer is orthogonal with the configured gain either way
INITS = ("orthogonal", "kaiming")
#: how the log std is kept in ``[min_log_std, max_log_std]``: a hard ``clip``, the soft
#: ``tanh`` map ``min + 0.5 (max - min) (tanh(x) + 1)``, or ``none``
LOG_STD_BOUNDS = ("clip", "tanh", "none")


# ----------------------------------------------------------------- building blocks
def activation_module(name: str) -> nn.Module:
    """The ``torch.nn`` activation class ``name`` (``ReLU``, ``SiLU``, ``Tanh``, ...), instantiated."""
    cls = getattr(nn, str(name), None)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ValueError(
            f"activation {name!r} is not a torch.nn module class; use the class name, e.g. "
            "ReLU, SiLU, Tanh, ELU, GELU"
        )
    return cls()


def check_mlp_critic_fields(prefix: str, hidden_size: int, num_layers: int, activation: str, init: str, output_init_scale: float) -> None:
    """The rules every MLP net shares; ``prefix`` names the config group in the error."""
    if int(hidden_size) < 1:
        raise ValueError(f"{prefix}.hidden_size must be >= 1, got {hidden_size}")
    if int(num_layers) < 1:
        raise ValueError(f"{prefix}.num_layers must be >= 1 hidden layer, got {num_layers}")
    try:
        activation_module(activation)
    except ValueError as exc:
        raise ValueError(f"{prefix}.activation: {exc}") from None
    if init not in INITS:
        raise ValueError(f"{prefix}.init must be one of {INITS}, got {init!r}")
    if float(output_init_scale) <= 0.0:
        raise ValueError(f"{prefix}.output_init_scale must be > 0, got {output_init_scale}")


def check_mlp_actor_fields(
    prefix: str,
    *,
    hidden_size: int,
    num_layers: int,
    activation: str,
    init: str,
    output_init_scale: float,
    init_log_std: float,
    learn_std: bool,
    use_state_dependent_std: bool,
    log_std_bound: str,
    min_log_std: float,
    max_log_std: float,
    squash: bool,
) -> None:
    """The actor's rules, shared by the config's ``validate`` and the model's constructor."""
    check_mlp_critic_fields(prefix, hidden_size, num_layers, activation, init, output_init_scale)
    if log_std_bound not in LOG_STD_BOUNDS:
        raise ValueError(f"{prefix}.log_std_bound must be one of {LOG_STD_BOUNDS}, got {log_std_bound!r}")
    if not learn_std and use_state_dependent_std:
        raise ValueError(
            f"{prefix}.learn_std: false freezes a state-independent log std, so "
            f"{prefix}.use_state_dependent_std must be false"
        )
    if log_std_bound != "none":
        if float(min_log_std) >= float(max_log_std):
            raise ValueError(
                f"{prefix}.min_log_std ({min_log_std}) must be below max_log_std ({max_log_std})"
            )
        inside = float(min_log_std) < float(init_log_std) < float(max_log_std)
        on_edge = float(init_log_std) in (float(min_log_std), float(max_log_std))
        if not inside and not (on_edge and log_std_bound == "clip"):
            raise ValueError(
                f"{prefix}.init_log_std ({init_log_std}) must lie inside "
                f"[min_log_std, max_log_std] = [{min_log_std}, {max_log_std}] for log_std_bound "
                f"{log_std_bound!r}" + (" (strictly: tanh never reaches the bounds)" if log_std_bound == "tanh" else "")
            )


def _init_hidden(linear: nn.Linear, init: str) -> None:
    if init == "orthogonal":
        nn.init.orthogonal_(linear.weight, math.sqrt(2.0))
    elif init == "kaiming":
        nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
    else:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    nn.init.zeros_(linear.bias)


def _init_output(linear: nn.Linear, scale: float) -> None:
    """ResiP's and CleanRL's output layer: orthogonal with gain ``scale``, zero bias."""
    nn.init.orthogonal_(linear.weight, float(scale))
    nn.init.zeros_(linear.bias)


def _trunk(in_dim: int, hidden_size: int, num_layers: int, activation: str, init: str) -> nn.Sequential:
    layers = []
    width = in_dim
    for _ in range(num_layers):
        linear = nn.Linear(width, hidden_size)
        _init_hidden(linear, init)
        layers += [linear, activation_module(activation)]
        width = hidden_size
    return nn.Sequential(*layers)


def _bound_log_std(raw: torch.Tensor, bound: str, lo: float, hi: float) -> torch.Tensor:
    if bound == "clip":
        return raw.clamp(lo, hi)
    if bound == "tanh":
        return lo + 0.5 * (hi - lo) * (torch.tanh(raw) + 1.0)
    return raw


def _raw_for_log_std(log_std: float, bound: str, lo: float, hi: float) -> float:
    """The pre-bound value whose bounded log std is ``log_std`` (so the net starts at the
    configured std whatever the bound)."""
    if bound == "tanh":
        unit = 2.0 * (log_std - lo) / (hi - lo) - 1.0
        return math.atanh(unit)
    return float(log_std)


# ----------------------------------------------------------------- plain single-agent nets
class MlpActorNet(nn.Module):
    """Trunk -> mean head; the log std is a second head (state dependent), a parameter, or a
    frozen buffer. Returns ``(mean, log_std)`` with the bound applied; it never samples."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
        init: str,
        output_init_scale: float,
        init_log_std: float,
        learn_std: bool,
        use_state_dependent_std: bool,
        log_std_bound: str,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        self.trunk = _trunk(obs_dim, hidden_size, num_layers, activation, init)
        self.mean = nn.Linear(hidden_size, action_dim)
        _init_output(self.mean, output_init_scale)
        self.use_state_dependent_std = bool(use_state_dependent_std)
        self.log_std_bound = str(log_std_bound)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        raw = _raw_for_log_std(float(init_log_std), self.log_std_bound, self.min_log_std, self.max_log_std)
        if self.use_state_dependent_std:
            # the std head starts at the configured sigma for every state
            self.log_std_head = nn.Linear(hidden_size, action_dim)
            _init_output(self.log_std_head, output_init_scale)
            with torch.no_grad():
                self.log_std_head.bias.fill_(raw)
            self.log_std = None
        else:
            value = torch.full((action_dim,), raw)
            self.log_std_head = None
            if learn_std:
                self.log_std = nn.Parameter(value)
            else:
                # a buffer: stacked and checkpointed like a parameter, invisible to the optimizer
                self.register_buffer("log_std", value)

    def forward(self, observations: torch.Tensor):
        hidden = self.trunk(observations)
        mean = self.mean(hidden)
        if self.use_state_dependent_std:
            raw = self.log_std_head(hidden)
        else:
            raw = self.log_std.expand(observations.shape[0], -1)
        return mean, _bound_log_std(raw, self.log_std_bound, self.min_log_std, self.max_log_std)


class MlpQCriticNet(nn.Module):
    """Q(observation, action) -> scalar, or ``outputs`` logits of a categorical Q."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, num_layers: int, activation: str, init: str, output_init_scale: float, outputs: int = 1) -> None:
        super().__init__()
        self.trunk = _trunk(obs_dim + action_dim, hidden_size, num_layers, activation, init)
        self.out = nn.Linear(hidden_size, outputs)
        _init_output(self.out, output_init_scale)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(torch.cat([observations, actions], dim=-1)))


class MlpValueCriticNet(nn.Module):
    """V(observation) -> scalar."""

    def __init__(self, obs_dim: int, hidden_size: int, num_layers: int, activation: str, init: str, output_init_scale: float) -> None:
        super().__init__()
        self.trunk = _trunk(obs_dim, hidden_size, num_layers, activation, init)
        self.out = nn.Linear(hidden_size, 1)
        _init_output(self.out, output_init_scale)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(observations))


# ----------------------------------------------------------------- the skrl-facing models
class MlpActor(EnsembleModel, GaussianMixin, Model):
    """Gaussian policy over ``num_agents`` agents: tanh-squashed (``squash``, what SAC needs)
    or plain (ResiP's residual). ``log_prob`` sums over the action dims; ``get_entropy`` is
    the pre-squash Gaussian's, per dim, as SimBa's. Continuous actions only."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
        init: str = "orthogonal",
        output_init_scale: float = 0.01,
        init_log_std: float = 0.0,
        learn_std: bool = True,
        use_state_dependent_std: bool = False,
        log_std_bound: str = "clip",
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        squash: bool = True,
        **layout,
    ) -> None:
        if layout:
            raise ValueError(
                f"architecture mlp has no selection head, so it cannot take {sorted(layout)}: a "
                "controller with a selection block (bernoulli_action_dims, match) needs "
                "architecture simba"
            )
        check_mlp_actor_fields(
            "model.actor",
            hidden_size=hidden_size, num_layers=num_layers, activation=activation, init=init,
            output_init_scale=output_init_scale, init_log_std=init_log_std, learn_std=learn_std,
            use_state_dependent_std=use_state_dependent_std, log_std_bound=log_std_bound,
            min_log_std=min_log_std, max_log_std=max_log_std, squash=squash,
        )
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        # the bound is the net's own (log_std_bound); skrl's clamp stays off
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=False, reduction="sum")
        self.num_agents = num_agents
        self.squash = bool(squash)
        self.learn_std = bool(learn_std)
        self.use_state_dependent_std = bool(use_state_dependent_std)
        #: every action dim is a Gaussian one (SimBa's actor has Bernoulli dims beside them)
        self.num_continuous = self.num_actions
        self._distribution: Optional[Normal] = None

        def build() -> nn.Module:
            return MlpActorNet(
                obs_dim=self.num_observations,
                action_dim=self.num_actions,
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                activation=str(activation),
                init=str(init),
                output_init_scale=float(output_init_scale),
                init_log_std=float(init_log_std),
                learn_std=bool(learn_std),
                use_state_dependent_std=bool(use_state_dependent_std),
                log_std_bound=str(log_std_bound),
                min_log_std=float(min_log_std),
                max_log_std=float(max_log_std),
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    def compute(self, inputs, role):
        mean, log_std = self.net.forward_flat(inputs["observations"])
        return mean, {"log_std": log_std}

    def act(self, inputs, *, role: str = ""):
        mean, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        distribution = Normal(mean, log_std.exp())
        self._distribution = distribution  # get_entropy reads this
        taken = inputs.get("taken_actions", None)
        noise = inputs.get("noise", None)
        # a given standard-normal draw (sac.noise_repeat) takes the place of rsample's own
        drawn = None
        if taken is None:
            drawn = distribution.rsample() if noise is None else mean + log_std.exp() * noise
        if self.squash:
            u = drawn if taken is None else safe_atanh(taken)
            log_prob = (
                distribution.log_prob(u).sum(dim=-1, keepdim=True)
                - squash_log_prob_correction(u).unsqueeze(-1)
            )
            actions, mean_actions = torch.tanh(u), torch.tanh(mean)
        else:
            actions = drawn if taken is None else taken
            log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
            mean_actions = mean
        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        """Entropy of the last ``act()``'s Gaussian, per dim ``(rows, num_actions)``."""
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)


class MlpQCritic(EnsembleModel, DeterministicMixin, Model):
    """Q(observation, action) over ``num_agents`` agents, ``members`` critics per agent stacked
    in one :class:`VmapEnsemble` (``compute`` returns ``(members, num_agents * rows, outputs)``:
    the Q value, or a categorical Q's logits)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
        init: str = "orthogonal",
        output_init_scale: float = 1.0,
        members: int = 1,
        outputs: int = 1,
    ) -> None:
        check_mlp_critic_fields("model.critic", hidden_size, num_layers, activation, init, output_init_scale)
        if outputs < 1:
            raise ValueError(f"a Q critic emits >= 1 column, got outputs={outputs}")
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_agents = num_agents
        self.members = members
        self.outputs = outputs

        def build() -> nn.Module:
            return MlpQCriticNet(
                self.num_observations, self.num_actions, int(hidden_size), int(num_layers),
                str(activation), str(init), float(output_init_scale), int(outputs),
            )

        self.net = VmapEnsemble(build, num_agents, device=device, members=members)

    def compute(self, inputs, role):
        return self.net.forward_flat(inputs["observations"], inputs["taken_actions"]), {}


class MlpValueCritic(EnsembleModel, DeterministicMixin, Model):
    """V(observation) over ``num_agents`` agents."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
        init: str = "orthogonal",
        output_init_scale: float = 1.0,
    ) -> None:
        check_mlp_critic_fields("model.critic", hidden_size, num_layers, activation, init, output_init_scale)
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_agents = num_agents

        def build() -> nn.Module:
            return MlpValueCriticNet(
                self.num_observations, int(hidden_size), int(num_layers), str(activation), str(init),
                float(output_init_scale),
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    def compute(self, inputs, role):
        return self.net.forward_flat(inputs["observations"]), {}

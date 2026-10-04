"""SimBa networks as plain single-agent modules, plus the ensemble wrappers skrl sees.

Two layers:

* ``SimbaTrunk`` / ``SimbaActorNet`` / ``SimbaQCriticNet`` / ``SimbaValueCriticNet`` are
  ordinary ``nn.Module``s for one agent. They return raw tensors only — no sampling, no
  in-place buffer writes — so :class:`~robonuke_rl_core.models.ensemble.VmapEnsemble` can
  batch ``num_agents`` copies of them with vmap.
* ``EnsembleActor`` / ``EnsembleQCritic`` / ``EnsembleValueCritic`` are the skrl models the
  learners call. They hold an ensemble, take and return flat ``(num_agents * rows, ...)``
  tensors, and build the distributions **outside** vmap.

Behavior is the port of the block SimBa models from
RoboNuke/generalized_hybrid_vic_action_space ``models/block_simba.py``: the same compressed
output layout (continuous means, then Bernoulli logits, with force-zero dims carrying no
parameters), the same initialization options, and the same ``act()`` contract.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from torch.distributions import Normal

from .ensemble import VmapEnsemble

__all__ = [
    "EnsembleModel",
    "squash_log_prob_correction",
    "safe_atanh",
    "SimbaTrunk",
    "SimbaActorNet",
    "SimbaQCriticNet",
    "SimbaValueCriticNet",
    "EnsembleActor",
    "EnsembleQCritic",
    "EnsembleValueCritic",
]


# ----------------------------------------------------------------- squashed-Gaussian utils
def squash_log_prob_correction(u: torch.Tensor) -> torch.Tensor:
    """``log(1 - tanh(u)^2)`` summed over the last dim, in a stable form."""
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def safe_atanh(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(torch.clamp(a, -1.0 + eps, 1.0 - eps))


# ----------------------------------------------------------------- plain single-agent nets
class SimbaTrunk(nn.Module):
    """Input projection -> ``num_blocks`` residual blocks -> LayerNorm -> output projection."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln": nn.LayerNorm(hidden_dim),
                        "fc1": nn.Linear(hidden_dim, 4 * hidden_dim),
                        "fc2": nn.Linear(4 * hidden_dim, hidden_dim),
                    }
                )
                for _ in range(num_blocks)
            ]
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        for module in (self.fc_in, self.fc_out, *[b[k] for b in self.blocks for k in ("fc1", "fc2")]):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        for block in self.blocks:
            x = x + block["fc2"](F.relu(block["fc1"](block["ln"](x))))
        return self.fc_out(self.ln_out(x))


class SimbaActorNet(nn.Module):
    """Policy trunk. Returns ``(mean_and_logits, log_std)``; it never samples.

    The trunk emits ``policy_out_dim`` columns laid out ``[continuous means | Bernoulli
    logits]``, plus ``num_continuous`` extra columns when the std is state dependent.
    """

    def __init__(
        self,
        obs_dim: int,
        policy_out_dim: int,
        num_continuous: int,
        hidden_dim: int,
        num_blocks: int,
        use_state_dependent_std: bool,
        log_std_init: torch.Tensor,
        last_layer_scale: float,
        scale_rows: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.policy_out_dim = policy_out_dim
        self.num_continuous = num_continuous
        self.use_state_dependent_std = use_state_dependent_std
        std_out_dim = num_continuous if use_state_dependent_std else 0
        self.trunk = SimbaTrunk(obs_dim, hidden_dim, policy_out_dim + std_out_dim, num_blocks)

        with torch.no_grad():
            if use_state_dependent_std:
                # the std rows start at the configured sigma and learn slowly
                self.trunk.fc_out.bias[policy_out_dim:] = log_std_init
                self.trunk.fc_out.weight[policy_out_dim:] *= 0.1
            if last_layer_scale != 1.0:
                rows = self.trunk.fc_out.weight[:policy_out_dim]
                rows *= (
                    last_layer_scale if scale_rows is None else scale_rows.view(-1, 1)
                )
        self.log_std = (
            None if use_state_dependent_std else nn.Parameter(log_std_init.clone())
        )

    def forward(self, observations: torch.Tensor):
        out = self.trunk(observations)
        mean = out[..., : self.policy_out_dim]
        if self.use_state_dependent_std:
            log_std = out[..., self.policy_out_dim :]
        else:
            log_std = self.log_std.expand(observations.shape[0], self.num_continuous)
        return mean, log_std


class SimbaQCriticNet(nn.Module):
    """Q(observation, action) -> scalar."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int, num_blocks: int, output_init_mean: float
    ) -> None:
        super().__init__()
        self.trunk = SimbaTrunk(obs_dim + action_dim, hidden_dim, 1, num_blocks)
        nn.init.constant_(self.trunk.fc_out.bias, output_init_mean)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([observations, actions], dim=-1))


class SimbaValueCriticNet(nn.Module):
    """V(observation) -> scalar."""

    def __init__(
        self, obs_dim: int, hidden_dim: int, num_blocks: int, output_init_mean: float
    ) -> None:
        super().__init__()
        self.trunk = SimbaTrunk(obs_dim, hidden_dim, 1, num_blocks)
        nn.init.constant_(self.trunk.fc_out.bias, output_init_mean)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.trunk(observations)


# ----------------------------------------------------------------- the skrl-facing models
class EnsembleModel:
    """Per-agent checkpoint state, delegated to the ensemble this model holds."""

    def agent_state_dict(self, agent: int):
        """One agent's weights, keyed exactly like the plain module's state dict."""
        return self.net.agent_state_dict(agent)

    def load_agent_state_dict(self, agent: int, state) -> None:
        """Write a plain module's state dict into slot ``agent``."""
        self.net.load_agent_state_dict(agent, state)


class EnsembleActor(EnsembleModel, GaussianMixin, Model):
    """Squashed-Gaussian policy (plus optional Bernoulli dims) over ``num_agents`` agents.

    Continuous dims use a tanh-squashed Gaussian. Dims in ``bernoulli_action_dims`` are drawn
    from a Bernoulli and mapped to {-1, +1} (Isaac Lab's BinaryJointAction convention), with a
    straight-through estimator so the critic's gradient still reaches the policy. Dims in
    ``force_zero_action_dims`` carry no parameters and are emitted as 0.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        act_init_std: float = 0.60653066,
        actor_n: int = 2,
        actor_latent: int = 512,
        last_layer_scale: float = 1.0,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        use_state_dependent_std: bool = False,
        bernoulli_action_dims: Optional[List[int]] = None,
        force_zero_action_dims: Optional[List[int]] = None,
        scale_down_action_dims: Optional[List[int]] = None,
        second_act_init_std: Optional[float] = None,
        second_act_init_std_dims: Optional[List[int]] = None,
    ) -> None:
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )
        self.num_agents = num_agents
        self.use_state_dependent_std = use_state_dependent_std

        bernoulli = sorted(set(bernoulli_action_dims or []))
        force_zero = sorted(set(force_zero_action_dims or []))
        for name, dims in (("bernoulli_action_dims", bernoulli), ("force_zero_action_dims", force_zero)):
            for dim in dims:
                if not 0 <= dim < self.num_actions:
                    raise ValueError(f"{name} index {dim} out of range [0, {self.num_actions})")
        overlap = sorted(set(bernoulli) & set(force_zero))
        if overlap:
            raise ValueError(
                f"bernoulli_action_dims and force_zero_action_dims must be disjoint; overlap={overlap}"
            )
        continuous = [d for d in range(self.num_actions) if d not in bernoulli and d not in force_zero]

        self.bernoulli_dims, self.force_zero_dims, self.continuous_dims = bernoulli, force_zero, continuous
        self.num_bernoulli = len(bernoulli)
        self.num_continuous = len(continuous)
        self._policy_out_dim = self.num_continuous + self.num_bernoulli

        # where to read each head from the trunk output, and where to scatter it in the action
        self._cont_out_idx = torch.arange(self.num_continuous, dtype=torch.long, device=device)
        self._bern_out_idx = torch.arange(
            self.num_continuous, self._policy_out_dim, dtype=torch.long, device=device
        )
        self._cont_action_idx = torch.as_tensor(continuous, dtype=torch.long, device=device)
        self._bern_action_idx = torch.as_tensor(bernoulli, dtype=torch.long, device=device)

        log_std_init = self._log_std_init(act_init_std, second_act_init_std, second_act_init_std_dims, device)
        scale_rows = self._scale_rows(last_layer_scale, scale_down_action_dims, device)

        def build() -> nn.Module:
            return SimbaActorNet(
                obs_dim=self.num_observations,
                policy_out_dim=self._policy_out_dim,
                num_continuous=self.num_continuous,
                hidden_dim=actor_latent,
                num_blocks=actor_n,
                use_state_dependent_std=use_state_dependent_std,
                log_std_init=log_std_init,
                last_layer_scale=last_layer_scale,
                scale_rows=scale_rows,
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    # ---- initialization helpers ----
    def _log_std_init(self, act_init_std, second_act_init_std, second_dims, device) -> torch.Tensor:
        """Per-continuous-dim initial log std, with the optional second value."""
        std = torch.full((self.num_continuous,), float(act_init_std), device=device)
        if second_dims:
            if second_act_init_std is None:
                raise ValueError("second_act_init_std_dims was given but second_act_init_std is None")
            position = {dim: i for i, dim in enumerate(self.continuous_dims)}
            for dim in sorted(set(second_dims)):
                if not 0 <= dim < self.num_actions:
                    raise ValueError(
                        f"second_act_init_std_dims index {dim} out of range [0, {self.num_actions})"
                    )
                if dim in position:  # a Bernoulli or force-zero dim has no sigma
                    std[position[dim]] = float(second_act_init_std)
        return torch.log(std)

    def _scale_rows(self, last_layer_scale, scale_down_dims, device) -> Optional[torch.Tensor]:
        """Per-output-row multiplier for the mean head, or None for "scale every row"."""
        if scale_down_dims is None:
            return None
        continuous_position = {dim: i for i, dim in enumerate(self.continuous_dims)}
        bernoulli_position = {
            dim: self.num_continuous + i for i, dim in enumerate(self.bernoulli_dims)
        }
        rows = torch.ones(self._policy_out_dim, device=device)
        for dim in sorted(set(scale_down_dims)):
            if not 0 <= dim < self.num_actions:
                raise ValueError(
                    f"scale_down_action_dims index {dim} out of range [0, {self.num_actions})"
                )
            row = continuous_position.get(dim, bernoulli_position.get(dim))
            if row is not None:  # force-zero dims carry no weights
                rows[row] = last_layer_scale
        return rows

    # ---- the skrl contract ----
    def compute(self, inputs, role):
        observations = inputs["observations"]
        raw_out, log_std = self.net.forward_flat(observations)
        return raw_out, {"log_std": log_std}

    def act(self, inputs, *, role: str = ""):
        raw_out, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        taken_actions = inputs.get("taken_actions", None)
        rows = raw_out.shape[0]
        actions = raw_out.new_zeros((rows, self.num_actions))
        mean_actions = raw_out.new_zeros((rows, self.num_actions))
        log_prob_parts = []

        if self.num_continuous > 0:
            cont_mean = raw_out.index_select(-1, self._cont_out_idx)
            distribution = Normal(cont_mean, log_std.exp())
            self._g_distribution = distribution  # GaussianMixin.get_entropy reads this
            if taken_actions is None:
                u = distribution.rsample()
            else:
                u = safe_atanh(taken_actions.index_select(-1, self._cont_action_idx))
            log_prob_parts.append(
                distribution.log_prob(u).sum(dim=-1, keepdim=True)
                - squash_log_prob_correction(u).unsqueeze(-1)
            )
            actions.index_copy_(-1, self._cont_action_idx, torch.tanh(u))
            mean_actions.index_copy_(-1, self._cont_action_idx, torch.tanh(cont_mean))

        if self.num_bernoulli > 0:
            probability = torch.sigmoid(raw_out.index_select(-1, self._bern_out_idx))
            if taken_actions is None:
                with torch.no_grad():
                    sample = (torch.rand_like(probability) < probability).float()
            else:
                taken = taken_actions.index_select(-1, self._bern_action_idx)
                sample = ((taken + 1.0) / 2.0).round().clamp(0.0, 1.0)
            # straight through: forward is the sample, backward is the probability
            through = (sample - probability).detach() + probability
            log_prob_parts.append(
                torch.distributions.Bernoulli(probs=probability)
                .log_prob(sample)
                .sum(dim=-1, keepdim=True)
            )
            actions.index_copy_(-1, self._bern_action_idx, 2.0 * through - 1.0)
            mean_actions.index_copy_(
                -1, self._bern_action_idx, 2.0 * (probability > 0.5).float() - 1.0
            )

        outputs["log_prob"] = (
            log_prob_parts[0] if len(log_prob_parts) == 1 else sum(log_prob_parts)
        )
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        """Continuous-Gaussian entropy. The squashed/Bernoulli mixture has no closed form."""
        if getattr(self, "_g_distribution", None) is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)


class EnsembleQCritic(EnsembleModel, DeterministicMixin, Model):
    """Q(observation, action) over ``num_agents`` agents."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_output_init_mean: float = 0.0,
        critic_n: int = 2,
        critic_latent: int = 512,
        clip_actions: bool = False,
    ) -> None:
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.num_agents = num_agents

        def build() -> nn.Module:
            return SimbaQCriticNet(
                self.num_observations, self.num_actions, critic_latent, critic_n, critic_output_init_mean
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    def compute(self, inputs, role):
        return self.net.forward_flat(inputs["observations"], inputs["taken_actions"]), {}


class EnsembleValueCritic(EnsembleModel, DeterministicMixin, Model):
    """V(observation) over ``num_agents`` agents."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_output_init_mean: float = 0.0,
        critic_n: int = 2,
        critic_latent: int = 512,
        clip_actions: bool = False,
    ) -> None:
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.num_agents = num_agents

        def build() -> nn.Module:
            return SimbaValueCriticNet(
                self.num_observations, critic_latent, critic_n, critic_output_init_mean
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    def compute(self, inputs, role):
        return self.net.forward_flat(inputs["observations"]), {}

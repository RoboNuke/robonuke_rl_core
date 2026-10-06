"""SimBa networks as plain single-agent modules, plus the ensemble wrappers skrl sees.

SimBa (Lee et al., 2025, https://arxiv.org/abs/2410.09754): residual MLP blocks with
LayerNorm, an architecture whose simplicity bias lets RL networks scale in parameters.
Its config lives in ``models/cfg.py`` (``SimbaActorCfg`` / ``SimbaCriticCfg``).

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

#: how the selection and the continuous dims form one joint distribution
SELECTION_DISTRIBUTIONS = ("product", "match")

#: **A selection bit of 1 means the axis is FORCE-controlled**, 0 that it is
#: position-controlled. The bit is the diagonal of ``S`` in
#: ``tau = J^T [ (I - S) (K e - D v) + S K_f (f_d - f) ]`` (``envs/forge/control.py``), and
#: this one convention holds everywhere: the Bernoulli bit, the action vector, the selection
#: matrix, ``outputs["selection_prob"]`` and the ``selection/*`` metrics all read "1 is
#: force". So the match gating below picks the **force** component where the bit is set, and
#: ``selection_init_bias`` is negative to start a run position-dominant.
SELECTION_BIT_IS_FORCE = True

__all__ = [
    "EnsembleModel",
    "squash_log_prob_correction",
    "safe_atanh",
    "SimbaTrunk",
    "SimbaActorNet",
    "SimbaQCriticNet",
    "SimbaValueCriticNet",
    "EnsembleActor",
    "SELECTION_DISTRIBUTIONS",
    "SELECTION_BIT_IS_FORCE",
    "EnsembleQCritic",
    "EnsembleValueCritic",
]


# ----------------------------------------------------------------- squashed-Gaussian utils
def squash_log_prob_correction(u: torch.Tensor) -> torch.Tensor:
    """``log(1 - tanh(u)^2)`` summed over the last dim, in a stable form."""
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def squash_correction_per_dim(u: torch.Tensor) -> torch.Tensor:
    """``log(1 - tanh(u)^2)`` per dim — :func:`squash_log_prob_correction` before its sum."""
    return 2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)


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
        selection_init_bias: float = 0.0,
        num_bernoulli: int = 0,
    ) -> None:
        super().__init__()
        self.policy_out_dim = policy_out_dim
        self.num_continuous = num_continuous
        self.use_state_dependent_std = use_state_dependent_std
        std_out_dim = num_continuous if use_state_dependent_std else 0
        self.trunk = SimbaTrunk(obs_dim, hidden_dim, policy_out_dim + std_out_dim, num_blocks)

        # The init tensors come from the ensemble wrapper, which may have built them on the
        # target device, while this plain module is built on the CPU and moved afterwards
        # (VmapEnsemble: build N copies, then .to(device)). So put them where the weights are
        # — otherwise an option that actually touches them (a last_layer_scale != 1, a state
        # dependent std) only fails on CUDA, and only when that option is set.
        weight_device = self.trunk.fc_out.weight.device
        log_std_init = log_std_init.to(weight_device)
        if scale_rows is not None:
            scale_rows = scale_rows.to(weight_device)

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
            if selection_init_bias and num_bernoulli:
                # after the output scaling, so the bias dominates the initial logit. The bit
                # is 1 for force (SELECTION_BIT_IS_FORCE), so a NEGATIVE value starts the
                # policy position-dominant (sigmoid(-2.2) ~ 0.1 force) instead of pushing on
                # every axis before it has learned anything
                start = policy_out_dim - num_bernoulli
                self.trunk.fc_out.bias[start:policy_out_dim] += selection_init_bias
        # no parameter when the std is state dependent, and none for an all-Bernoulli policy:
        # a zero-element parameter would never enter the graph, so its grad would stay None
        # and BlockAdamW (which requires every parameter to have a gradient) would raise
        self.log_std = (
            None
            if use_state_dependent_std or num_continuous == 0
            else nn.Parameter(log_std_init.clone())
        )

    def forward(self, observations: torch.Tensor):
        out = self.trunk(observations)
        mean = out[..., : self.policy_out_dim]
        if self.use_state_dependent_std:
            log_std = out[..., self.policy_out_dim :]
        elif self.log_std is None:  # all-Bernoulli: no continuous dims, no sigma
            log_std = out.new_zeros(observations.shape[0], 0)
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
        selection_distribution: str = "product",
        selection_init_bias: float = 0.0,
        pos_component_dims: Optional[List[int]] = None,
        force_component_dims: Optional[List[int]] = None,
        selection_names: Optional[List[str]] = None,
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
        self._g_distribution = None  # the last act()'s Gaussian, for get_entropy
        self._b_distribution = None  # the last act()'s Bernoulli, for get_entropy

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

        self.selection_distribution = str(selection_distribution)
        if self.selection_distribution not in SELECTION_DISTRIBUTIONS:
            raise ValueError(
                f"selection_distribution must be one of {SELECTION_DISTRIBUTIONS}, got "
                f"{self.selection_distribution!r}"
            )
        self._build_pairs(pos_component_dims, force_component_dims, continuous, device)
        #: one label per selection dim, used only to name the logged probabilities. The
        #: model factory fills it with the controller's force-eligible axis names; the
        #: fallback is the bit's own index, so logging never depends on a controller.
        self.selection_names = (
            [str(name) for name in selection_names]
            if selection_names
            else [str(index) for index in range(self.num_bernoulli)]
        )
        if len(self.selection_names) != self.num_bernoulli:
            raise ValueError(
                f"selection_names has {len(self.selection_names)} entries but there are "
                f"{self.num_bernoulli} selection dims"
            )

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
                selection_init_bias=float(selection_init_bias),
                num_bernoulli=self.num_bernoulli,
            )

        self.net = VmapEnsemble(build, num_agents, device=device)

    # ---- the gated pairs (match) ----
    def _build_pairs(self, pos_dims, force_dims, continuous, device) -> None:
        """Columns of each (pose, force) pair, derived from the layout — never hand-written.

        ``pos_dims`` and ``force_dims`` are **action** indices, in the same order as the
        selection dims. What the density needs is their position among the *continuous*
        columns, since that is the matrix the per-dim log probability lives in.
        """
        pos_dims = list(pos_dims or [])
        force_dims = list(force_dims or [])
        if len(pos_dims) != len(force_dims):
            raise ValueError(
                f"pos_component_dims and force_component_dims must pair up one to one; got "
                f"{pos_dims} and {force_dims}"
            )
        if pos_dims and len(pos_dims) != self.num_bernoulli:
            raise ValueError(
                f"a (pose, force) pair needs a selection dim each: {len(pos_dims)} pairs but "
                f"{self.num_bernoulli} selection dims"
            )
        if self.selection_distribution == "match" and not pos_dims:
            raise ValueError(
                "selection_distribution='match' conditions the continuous density on the "
                "selection, so it needs the (pose, force) pairs: enable the controller's force "
                "branch, which is what derives them"
            )

        column = {dim: index for index, dim in enumerate(continuous)}
        for name, dims in (("pos_component_dims", pos_dims), ("force_component_dims", force_dims)):
            missing = [dim for dim in dims if dim not in column]
            if missing:
                raise ValueError(
                    f"{name} {missing} are not continuous action dims (they are Bernoulli, "
                    "force-zero, or out of range), so they have no density to gate"
                )
        paired = {column[dim] for dim in pos_dims} | {column[dim] for dim in force_dims}
        self._pair_pos_out = torch.as_tensor(
            [column[dim] for dim in pos_dims], dtype=torch.long, device=device
        )
        self._pair_force_out = torch.as_tensor(
            [column[dim] for dim in force_dims], dtype=torch.long, device=device
        )
        self._free_cont_out = torch.as_tensor(
            [index for index in range(self.num_continuous) if index not in paired],
            dtype=torch.long,
            device=device,
        )
        self._selection_prob = None

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
        self._g_distribution = None
        self._b_distribution = None
        self._selection_prob = None
        self._selection_sample = None

        match = self.selection_distribution == "match"
        continuous_log_prob = None  # per dim, only built for match
        if self.num_continuous > 0:
            cont_mean = raw_out.index_select(-1, self._cont_out_idx)
            distribution = Normal(cont_mean, log_std.exp())
            self._g_distribution = distribution  # GaussianMixin.get_entropy reads this
            if taken_actions is None:
                u = distribution.rsample()
            else:
                u = safe_atanh(taken_actions.index_select(-1, self._cont_action_idx))
            if match:
                # the same density, left per dim so the selection can pick a column. Summing
                # it would NOT reproduce the product path bit for bit (float addition is not
                # associative), which is exactly why product keeps its own expression below.
                continuous_log_prob = distribution.log_prob(u) - squash_correction_per_dim(u)
            else:
                log_prob_parts.append(
                    distribution.log_prob(u).sum(dim=-1, keepdim=True)
                    - squash_log_prob_correction(u).unsqueeze(-1)
                )
            actions.index_copy_(-1, self._cont_action_idx, torch.tanh(u))
            mean_actions.index_copy_(-1, self._cont_action_idx, torch.tanh(cont_mean))

        if self.num_bernoulli > 0:
            probability = torch.sigmoid(raw_out.index_select(-1, self._bern_out_idx))
            bernoulli_distribution = torch.distributions.Bernoulli(probs=probability)
            self._b_distribution = bernoulli_distribution  # get_entropy reads this
            if taken_actions is None:
                with torch.no_grad():
                    sample = (torch.rand_like(probability) < probability).float()
            else:
                taken = taken_actions.index_select(-1, self._bern_action_idx)
                sample = ((taken + 1.0) / 2.0).round().clamp(0.0, 1.0)
            # straight through: forward is the sample, backward is the probability
            through = (sample - probability).detach() + probability
            log_prob_parts.append(
                bernoulli_distribution.log_prob(sample).sum(dim=-1, keepdim=True)
            )
            self._selection_prob = probability
            self._selection_sample = sample
            actions.index_copy_(-1, self._bern_action_idx, 2.0 * through - 1.0)
            mean_actions.index_copy_(
                -1, self._bern_action_idx, 2.0 * (probability > 0.5).float() - 1.0
            )

        if match and continuous_log_prob is not None:
            # only the SELECTED member of each pair is a random variable: the other is a
            # controller input the env ignores on that axis, so scoring it would charge the
            # policy for a number that did nothing
            free = continuous_log_prob.index_select(-1, self._free_cont_out).sum(
                dim=-1, keepdim=True
            )
            # bit 1 == force-controlled (SELECTION_BIT_IS_FORCE): the force component is the
            # live one there, the pose component on the other branch
            selected = self._selection_sample
            gated = torch.where(
                selected > 0.5,
                continuous_log_prob.index_select(-1, self._pair_force_out),
                continuous_log_prob.index_select(-1, self._pair_pos_out),
            ).sum(dim=-1, keepdim=True)
            log_prob_parts.insert(0, free + gated)

        outputs["log_prob"] = (
            log_prob_parts[0] if len(log_prob_parts) == 1 else sum(log_prob_parts)
        )
        outputs["mean_actions"] = mean_actions
        #: per-axis probability of the selection bit, i.e. of the axis being FORCE
        #: controlled (SELECTION_BIT_IS_FORCE). This is what the supervised selection loss
        #: predicts with and what the learner logs. None when there are no selection dims.
        outputs["selection_prob"] = self._selection_prob
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        """Entropy of the last ``act()``.

        ``product``: per dim, ``(rows, num_continuous + num_bernoulli)`` — the Gaussian
        columns are the pre-squash entropy (the tanh-squashed density has no closed form),
        the Bernoulli columns are exact. Both carry gradients, so PPO's entropy bonus
        regularizes the Bernoulli dims too.

        ``match``: one column, ``(rows, 1)``, because the gated pairs contribute a mixture
        rather than a per-dim quantity. See :meth:`_match_entropy`.
        """
        parts = []
        if self._g_distribution is not None:
            parts.append(self._g_distribution.entropy())
        if self._b_distribution is not None:
            parts.append(self._b_distribution.entropy())
        if not parts:
            return torch.tensor(0.0, device=self.device)
        if self.selection_distribution == "match":
            return self._match_entropy(parts)
        return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)).to(self.device)

    def _match_entropy(self, parts) -> torch.Tensor:
        """``H(S) + E_S[H(continuous | S)]``, one column per sample.

        The gated pairs contribute the selection-probability-weighted mix of their two
        components, because only one of them is live in any given sample. Returned summed
        (``(rows, 1)``) rather than per dim: a mix is not a per-dim quantity, and PPO's
        reshape treats whatever columns it gets as dims to average.
        """
        continuous = parts[0] if self._g_distribution is not None else None
        selection = parts[-1] if self._b_distribution is not None else None
        if continuous is None:
            return (selection.sum(dim=-1, keepdim=True)).to(self.device)
        free = continuous.index_select(-1, self._free_cont_out).sum(dim=-1, keepdim=True)
        # p is the probability of force control, so it weights the FORCE component
        probability = self._selection_prob
        gated = (
            (1.0 - probability) * continuous.index_select(-1, self._pair_pos_out)
            + probability * continuous.index_select(-1, self._pair_force_out)
        ).sum(dim=-1, keepdim=True)
        total = free + gated
        if selection is not None:
            total = total + selection.sum(dim=-1, keepdim=True)
        return total.to(self.device)


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

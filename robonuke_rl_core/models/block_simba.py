"""Block-parallel SimBa networks for SAC and PPO.

One network holds every agent's parameters with a leading ``num_agents`` dimension, so all
agents run in a single forward pass and no operation mixes two agents' rows.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``models/block_simba.py``. The
hybrid force/position actor is left out for now: it depends on the controller action layout,
which gets its own design pass. The per-agent slicing helpers live in ``block_utils.py``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


# -----------------------------
#  Squashed Gaussian utilities
# -----------------------------
def squash_log_prob_correction(u: torch.Tensor) -> torch.Tensor:
    # log(1 - tanh(u)^2) summed over last dim; numerically stable form
    return (2.0 * math.log(2.0) - 2.0 * u - 2.0 * F.softplus(-2.0 * u)).sum(dim=-1)


def safe_atanh(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.atanh(torch.clamp(a, -1.0 + eps, 1.0 - eps))


# -----------------------------
#  Block-parallel primitives
# -----------------------------
class BlockLinear(nn.Module):
    def __init__(self, num_blocks: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_blocks, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(num_blocks, out_features))
        for i in range(num_blocks):
            nn.init.kaiming_normal_(self.weight[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_blocks, batch, in_features)
        return torch.einsum("nbi,noi->nbo", x, self.weight) + self.bias[:, None, :]


class BlockLayerNorm(nn.Module):
    def __init__(self, num_blocks: int, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_blocks, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_blocks, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        return out * self.weight[:, None, :] + self.bias[:, None, :]


class BlockMLP(nn.Module):
    def __init__(self, num_blocks: int, in_dim: int, hidden_dim: int, out_dim: int, activation=None):
        super().__init__()
        self.fc1 = BlockLinear(num_blocks, in_dim, hidden_dim)
        self.fc2 = BlockLinear(num_blocks, hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(F.relu(self.fc1(x)))
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        return out


class BlockResidualBlock(nn.Module):
    def __init__(self, num_blocks: int, dim: int):
        super().__init__()
        self.ln = BlockLayerNorm(num_blocks, dim)
        self.fc1 = BlockLinear(num_blocks, dim, 4 * dim)
        self.fc2 = BlockLinear(num_blocks, 4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


# -----------------------------
#  BlockSimBa backbone
# -----------------------------
class BlockSimBa(nn.Module):
    """Block-parallel SimBa: input proj -> N residual blocks -> LN -> output proj."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        device,
        num_blocks: int = 2,
        use_state_dependent_std: bool = False,
    ):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim
        self.num_blocks = num_blocks
        self.use_state_dependent_std = use_state_dependent_std
        self.std_out_dim = act_dim if use_state_dependent_std else 0

        # Output layout per row (along last dim):
        #   [0 : act_dim)                                  -> action mean
        #   [act_dim : act_dim + std_out_dim)              -> per-action log_std (state-dep std)
        total_out = act_dim + self.std_out_dim
        self.fc_in = BlockLinear(num_agents, obs_dim, hidden_dim)
        self.resblocks = nn.ModuleList(
            [BlockResidualBlock(num_agents, hidden_dim) for _ in range(num_blocks)]
        )
        self.ln_out = BlockLayerNorm(num_agents, hidden_dim)
        self.fc_out = BlockLinear(num_agents, hidden_dim, total_out)

    def forward(self, obs_flat: torch.Tensor, num_envs: int):
        """Return ``(actions, log_std)``.

        ``log_std`` is ``None`` unless ``use_state_dependent_std`` was set; when present its
        shape is ``(num_agents * num_envs, std_out_dim)``.
        """
        obs = obs_flat.view(self.num_agents, num_envs, -1)
        x = self.fc_in(obs)
        for block in self.resblocks:
            x = block(x)
        out = self.fc_out(self.ln_out(x))

        actions = out[..., : self.act_dim]
        if self.std_out_dim > 0:
            log_std = out[..., self.act_dim : self.act_dim + self.std_out_dim].reshape(
                -1, self.std_out_dim
            )
        else:
            log_std = None

        return actions.reshape(-1, actions.shape[-1]), log_std


# -----------------------------
#  Squashed-Gaussian actor
# -----------------------------
class BlockSimBaActor(GaussianMixin, Model):
    """SAC policy: hybrid continuous + discrete (Bernoulli) action distribution,
    block-parallel across agents.

    Most action dims use a tanh-squashed Gaussian (standard SAC). Indices listed
    in ``bernoulli_action_dims`` are sampled from a Bernoulli (binary) instead;
    the {0,1} sample is mapped to {-1,+1} so Isaac Lab's BinaryJointAction sees
    the right sign convention. A straight-through estimator carries the critic's
    gradient back through the soft probability so SAC's reparameterized policy
    gradient still works for those dims.

    Reads ``inputs["observations"]`` per skrl SAC convention. ``act()`` returns
    the (mixed) action vector and a combined log_prob = continuous-squashed-
    Gaussian log_prob + Bernoulli log_prob.
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
        bernoulli_action_dims: list[int] | None = None,
        force_zero_action_dims: list[int] | None = None,
        scale_down_action_dims: list[int] | None = None,
        second_act_init_std: float | None = None,
        second_act_init_std_dims: list[int] | None = None,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
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

        # Resolve which action dims are continuous vs Bernoulli vs force-zero.
        # Indices into the full action vector that the env consumes (so env-side
        # ordering is preserved when we reassemble below). The three sets are
        # disjoint and partition range(num_actions).
        bdims = sorted(set(bernoulli_action_dims or []))
        zdims = sorted(set(force_zero_action_dims or []))
        for d in bdims:
            if d < 0 or d >= self.num_actions:
                raise ValueError(
                    f"bernoulli_action_dims index {d} out of range [0, {self.num_actions})"
                )
        for d in zdims:
            if d < 0 or d >= self.num_actions:
                raise ValueError(
                    f"force_zero_action_dims index {d} out of range [0, {self.num_actions})"
                )
        if set(bdims) & set(zdims):
            raise ValueError(
                "bernoulli_action_dims and force_zero_action_dims must be disjoint; "
                f"overlap = {sorted(set(bdims) & set(zdims))}"
            )
        self.bernoulli_dims: list[int] = bdims
        self.force_zero_dims: list[int] = zdims
        self.continuous_dims: list[int] = [
            d for d in range(self.num_actions) if d not in bdims and d not in zdims
        ]
        self.num_bernoulli = len(self.bernoulli_dims)
        self.num_force_zero = len(self.force_zero_dims)
        self.num_continuous = len(self.continuous_dims)

        # The backbone produces only `num_continuous + num_bernoulli` action outputs
        # (force-zero dims have no model parameters at all). Within that compressed
        # output, the layout is: [continuous_means | bernoulli_logits].
        self._policy_out_dim = self.num_continuous + self.num_bernoulli
        # Backbone-output indices (where to slice from raw_out).
        self._cont_out_idx = torch.arange(
            0, self.num_continuous, dtype=torch.long, device=device
        )
        self._bern_out_idx = torch.arange(
            self.num_continuous, self._policy_out_dim, dtype=torch.long, device=device
        )
        # Action-vector indices (where to scatter into the env-facing action tensor).
        self._cont_action_idx = torch.as_tensor(self.continuous_dims, dtype=torch.long, device=device)
        self._bern_action_idx = torch.as_tensor(self.bernoulli_dims, dtype=torch.long, device=device)
        self._zero_action_idx = torch.as_tensor(self.force_zero_dims, dtype=torch.long, device=device)

        self.actor_mean = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations,
            hidden_dim=actor_latent,
            act_dim=self._policy_out_dim,   # ← shrunk: no params allocated for force-zero dims
            device=device,
            num_blocks=actor_n,
            use_state_dependent_std=use_state_dependent_std,
        ).to(device)

        # log_std parameters cover ONLY continuous dims (Bernoulli has no σ). Every continuous dim
        # starts at act_init_std, EXCEPT env-facing dims listed in second_act_init_std_dims, which
        # start at second_act_init_std instead (e.g. damp initial exploration on the rotation/gain
        # dims while keeping the pose deltas at the larger default).
        std_init = torch.full((self.num_continuous,), float(act_init_std), device=device)
        if second_act_init_std_dims:
            if second_act_init_std is None:
                raise ValueError(
                    "second_act_init_std_dims was given but second_act_init_std is None"
                )
            cont_pos = {d: i for i, d in enumerate(self.continuous_dims)}
            for d in sorted(set(second_act_init_std_dims)):
                if d < 0 or d >= self.num_actions:
                    raise ValueError(
                        f"second_act_init_std_dims index {d} out of range [0, {self.num_actions})"
                    )
                pos = cont_pos.get(d)  # None if d is a selection/Bernoulli or force-zero dim (no σ)
                if pos is not None:
                    std_init[pos] = float(second_act_init_std)
        log_std_init = torch.log(std_init)  # (num_continuous,)

        if use_state_dependent_std:
            with torch.no_grad():
                # State-dep std rows live at [act_dim : act_dim + std_out_dim] in the
                # backbone, where act_dim = _policy_out_dim. We restrict consumption
                # to the continuous-only slice (self._cont_out_idx) at runtime.
                self.actor_mean.fc_out.bias[:, self._policy_out_dim:] = math.log(act_init_std)
                self.actor_mean.fc_out.bias[
                    :, self._policy_out_dim:self._policy_out_dim + self.num_continuous
                ] = log_std_init
                self.actor_mean.fc_out.weight[:, self._policy_out_dim:, :] *= 0.1
            self.actor_logstd = None
        else:
            # A BLOCK parameter (num_agents, num_continuous), not one entry per agent: with the
            # leading agent dim every per-agent helper works on it unchanged — gradient clipping,
            # checkpoint slicing, and the optimizer-state slicing (an optimizer state tensor of
            # shape (1, k) could not be attributed to its agent).
            self.actor_logstd = nn.Parameter(
                log_std_init.clone().view(1, -1).repeat(num_agents, 1).to(device)
            )

        with torch.no_grad():
            # Down-scale the final-layer mean weights at init. By default EVERY policy-output
            # row (first _policy_out_dim) is scaled uniformly. If scale_down_action_dims is
            # given, only the listed env-facing action dims are scaled and the rest keep scale
            # 1.0 — so configs can damp e.g. the pose deltas without also shrinking the gain /
            # selection dims toward zero (which otherwise leaves their actions stuck near the
            # zero-action midpoint and unexplored).
            w = self.actor_mean.fc_out.weight
            if scale_down_action_dims is None:
                w[:, : self._policy_out_dim, :] *= last_layer_scale
            else:
                # Map each requested action-vector dim to its backbone-output row
                # ([continuous_means | bernoulli_logits]); force-zero dims have no row.
                cont_pos = {d: i for i, d in enumerate(self.continuous_dims)}
                bern_pos = {d: self.num_continuous + i for i, d in enumerate(self.bernoulli_dims)}
                mult = torch.ones(self._policy_out_dim, device=device)
                for d in sorted(set(scale_down_action_dims)):
                    if d < 0 or d >= self.num_actions:
                        raise ValueError(
                            f"scale_down_action_dims index {d} out of range [0, {self.num_actions})"
                        )
                    row = cont_pos.get(d, bern_pos.get(d))
                    if row is not None:  # force-zero dims carry no weights -> nothing to scale
                        mult[row] = last_layer_scale
                w[:, : self._policy_out_dim, :] *= mult.view(1, -1, 1)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        num_envs = obs.size(0) // self.num_agents
        raw_out, log_std = self.actor_mean(obs, num_envs)
        # raw_out shape: (N*B, _policy_out_dim) where the layout is
        #   [0 : num_continuous)                          -> continuous Gaussian means
        #   [num_continuous : num_continuous+num_bernoulli) -> Bernoulli logits
        # Force-zero action dims are NOT produced by the model — they're inserted
        # as 0 in the env-facing action vector inside act().

        if not self.use_state_dependent_std:
            batch_size = raw_out.size(0) // self.num_agents
            # (N, k) -> (N, batch, k) -> (N*batch, k): each agent's rows get its own log_std
            log_std = self.actor_logstd.unsqueeze(1).expand(
                self.num_agents, batch_size, self.num_continuous
            ).reshape(-1, self.num_continuous)
        elif self.num_continuous < self._policy_out_dim:
            # State-dep std emits one std per backbone-output dim (continuous +
            # bernoulli). Restrict to continuous-only since Bernoulli has no σ.
            log_std = log_std.index_select(-1, self._cont_out_idx)

        outputs = {"log_std": log_std}
        return raw_out, outputs

    def act(self, inputs, *, role: str = ""):
        # Hybrid continuous (squashed Gaussian) + discrete (Bernoulli) sampling.
        # Returns (actions, outputs) per skrl 2.x convention; outputs["log_prob"]
        # is the combined log-prob used by SAC's policy / entropy losses.
        raw_out, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]  # (N*B, num_continuous)

        if self._g_clip_log_std:
            log_std = torch.clamp(log_std, min=self._g_min_log_std, max=self._g_max_log_std)
            outputs["log_std"] = log_std

        taken_actions = inputs.get("taken_actions", None)
        batch = raw_out.shape[0]
        log_prob_parts: list[torch.Tensor] = []
        actions = raw_out.new_zeros((batch, self.num_actions))
        cont_dist = None

        # ---- continuous head (squashed Gaussian on continuous_dims) ----
        # Read from the FIRST num_continuous columns of the (compressed) backbone
        # output; scatter into the env-facing action positions self._cont_action_idx.
        if self.num_continuous > 0:
            cont_mean = raw_out.index_select(-1, self._cont_out_idx)     # (N*B, num_continuous)
            sigma = log_std.exp()
            cont_dist = Normal(cont_mean, sigma)
            self._g_distribution = cont_dist  # for GaussianMixin.get_entropy() compat
            if taken_actions is None:
                u = cont_dist.rsample()
            else:
                # Replay path: recover pre-tanh u from stored continuous actions in (-1, 1).
                taken_cont = taken_actions.index_select(-1, self._cont_action_idx)
                u = safe_atanh(taken_cont)
            a_cont = torch.tanh(u)
            # log p(a_cont) = log p(u) - sum log(1 - tanh^2(u))   (Jacobian correction)
            lp_cont = (
                cont_dist.log_prob(u).sum(dim=-1, keepdim=True)
                - squash_log_prob_correction(u).unsqueeze(-1)
            )
            log_prob_parts.append(lp_cont)
            actions.index_copy_(-1, self._cont_action_idx, a_cont)

        # ---- Bernoulli head (binary on bernoulli_dims, mapped to {-1,+1}) ----
        # Read from the SECOND block of backbone output columns; scatter into the
        # env-facing Bernoulli action positions self._bern_action_idx.
        if self.num_bernoulli > 0:
            bern_logit = raw_out.index_select(-1, self._bern_out_idx)    # (N*B, num_bernoulli)
            bern_prob = torch.sigmoid(bern_logit)
            if taken_actions is None:
                # Fresh sample: draw a Bernoulli sample, route gradient through prob via
                # straight-through estimator. forward = sample, backward = bern_prob.
                with torch.no_grad():
                    bern_sample = (torch.rand_like(bern_prob) < bern_prob).float()
                bern_st = (bern_sample - bern_prob).detach() + bern_prob
                a_bern = 2.0 * bern_st - 1.0                              # {-1, +1} forward
            else:
                # Replay path: stored action is in {-1, +1}; decode to {0, 1} for log_prob.
                # Forward action goes back to env shape; gradient flows through bern_prob
                # via straight-through so the critic Q-grad reaches the policy.
                taken_bern = taken_actions.index_select(-1, self._bern_action_idx)
                bern_sample = ((taken_bern + 1.0) / 2.0).round().clamp(0.0, 1.0)
                bern_st = (bern_sample - bern_prob).detach() + bern_prob
                a_bern = 2.0 * bern_st - 1.0
            bern_dist = torch.distributions.Bernoulli(probs=bern_prob)
            lp_bern = bern_dist.log_prob(bern_sample).sum(dim=-1, keepdim=True)
            log_prob_parts.append(lp_bern)
            actions.index_copy_(-1, self._bern_action_idx, a_bern)

        log_prob = log_prob_parts[0] if len(log_prob_parts) == 1 else sum(log_prob_parts)

        outputs["log_prob"] = log_prob
        # Deterministic (mean) action in the SAME env-facing layout as ``actions`` (see the
        # hybrid actor for the rationale): tanh-squashed continuous means scattered into their
        # positions, Bernoulli gates thresholded to {-1,+1}, force-zero dims left at 0. Avoids
        # feeding the bare ``raw_out`` (compressed/pre-tanh) to skrl's deterministic eval.
        mean_actions = actions.new_zeros((batch, self.num_actions))
        if self.num_continuous > 0:
            mean_actions.index_copy_(-1, self._cont_action_idx, torch.tanh(cont_mean))
        if self.num_bernoulli > 0:
            mean_actions.index_copy_(-1, self._bern_action_idx, 2.0 * (bern_prob > 0.5).float() - 1.0)
        outputs["mean_actions"] = mean_actions
        return actions, outputs

    def get_entropy(self, *, role: str = ""):
        # Continuous-Gaussian entropy as a proxy; the squashed/Bernoulli mixture has
        # no clean closed form and SAC uses log_prob (not entropy) in the gradient.
        if self._g_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)


# -----------------------------
#  Q-critic (state, action -> scalar)
# -----------------------------
class BlockSimBaQCritic(DeterministicMixin, Model):
    """SAC Q-function: concatenates observation and action, returns scalar Q per (o, a).

    skrl SAC calls this via `critic.act({**inputs, "taken_actions": actions})`, where
    `inputs["observations"]` carries the observation and `inputs["taken_actions"]` the action.
    """

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
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.q_net = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations + self.num_actions,
            hidden_dim=critic_latent,
            act_dim=1,
            device=device,
            num_blocks=critic_n,
            use_state_dependent_std=False,
        ).to(device)

        torch.nn.init.constant_(self.q_net.fc_out.bias, critic_output_init_mean)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        actions = inputs["taken_actions"]
        x = torch.cat([obs, actions], dim=-1)
        num_envs = x.size(0) // self.num_agents
        value, _ = self.q_net(x, num_envs)  # backbone returns (out, log_std)
        return value, {}


# -----------------------------
#  State-value critic (state -> scalar) — PPO
# -----------------------------
class BlockSimBaValueCritic(DeterministicMixin, Model):
    """PPO state-value function V(obs) -> scalar, block-parallel across agents.

    Mirrors :class:`BlockSimBaQCritic` but consumes observations ONLY (no action
    concatenation), since PPO's critic estimates V(s) rather than Q(s, a). Reuses
    the same ``BlockSimBa`` backbone and accepts the same ``model_cfg.critic``
    kwargs, so the per-agent save/load slicing helpers apply unchanged.

    skrl PPO calls this via ``value.act({"observations": obs}, role="value")``.
    """

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
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.v_net = BlockSimBa(
            num_agents=num_agents,
            obs_dim=self.num_observations,   # obs only — no +num_actions
            hidden_dim=critic_latent,
            act_dim=1,
            device=device,
            num_blocks=critic_n,
            use_state_dependent_std=False,
        ).to(device)

        torch.nn.init.constant_(self.v_net.fc_out.bias, critic_output_init_mean)

    def compute(self, inputs, role):
        obs = inputs["observations"]
        num_envs = obs.size(0) // self.num_agents
        value, _ = self.v_net(obs, num_envs)  # backbone returns (out, log_std)
        return value, {}

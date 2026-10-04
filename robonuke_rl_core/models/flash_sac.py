"""FlashSAC networks and math, block-parallel across agents.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``models/flash_sac.py``:
the FlashSAC (Kim et al., 2026) model family for this repo's block-parallel SAC. Mirrors the authors' released architecture
(``github.com/Holiday-Robot/FlashSAC``, ``flash_rl/agents/flashSAC``) but makes the
leading tensor dimension ``num_agents`` (one independent agent per block) instead of
the reference's 2-critic ensemble — the einsum layout is identical, so N agents train
in a single fused pass exactly like ``models/block_simba.py``.

Design constraints:
* Does NOT modify or subclass the SimBa models. The ONLY things imported from
  ``block_simba`` are the shared squashed-Gaussian utilities.
* Every parameter and buffer carries a leading ``num_agents`` dim, so the existing
  per-agent checkpoint slicers (``slice_block_state_dict`` etc.) apply unchanged.

Components:
* :class:`BlockUnitLinear` / :class:`BlockBatchNorm` / :class:`BlockRMSNorm` — the
  block-parallel "unit" layers (weight-normalized linear, pre-activation BN, post RMS).
* :class:`FlashSimBaActor` — state-dependent tanh-squashed Gaussian policy.
* :class:`FlashCategoricalQCritic` — distributional (categorical / C51) Q critic.
* Free functions: :func:`categorical_td_target`, :func:`categorical_ce_loss`,
  :func:`min_q_log_probs`, :func:`normalize_flash_parameters`,
  :func:`block_cross_cat` / :func:`block_cross_split`, :func:`build_zeta_cdf`.
* :class:`FlashReturnScaler` — adaptive reward scaling (Eq. 6), reusing skrl's
  ``RunningStandardScaler`` for the running return variance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# The ONLY block_simba imports: shared squashed-Gaussian utilities (no reimplementation).
from .block_simba import safe_atanh, squash_log_prob_correction


# =============================================================================
#  Block-parallel "unit" layers (leading dim = num_agents)
# =============================================================================
class BlockUnitLinear(nn.Module):
    """Block-parallel linear ``(N, out, in)`` with einsum, orthogonal init, no bias by
    default. ``normalize_parameters`` projects each output row onto the unit sphere
    (FlashSAC weight normalization). Optional per-block bias (used by the output heads)."""

    def __init__(self, num_blocks: int, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_blocks, out_features, in_features))
        for i in range(num_blocks):
            nn.init.orthogonal_(self.weight[i], gain=1)
        self.bias = nn.Parameter(torch.zeros(num_blocks, out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_blocks, batch, in_features) -> (num_blocks, batch, out_features)
        out = torch.einsum("nbi,noi->nbo", x, self.weight)
        if self.bias is not None:
            out = out + self.bias[:, None, :]
        return out

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class BlockBatchNorm(nn.Module):
    """Block-parallel BatchNorm over the batch dim (dim=1), per ``(block, feature)``.

    Hand-rolled because ``F.batch_norm`` cannot take a leading ensemble/agent dim.
    Running stats are ``(num_blocks, dim)`` so the existing checkpoint slicers carry
    them per agent. ``normalize_parameters`` projects ``(gamma, beta)`` jointly to
    norm ``sqrt(d)``. Running stats are kept in float32 (BN convention under AMP)."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_blocks: int, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_blocks, dim))
        self.bias = nn.Parameter(torch.zeros(num_blocks, dim))
        self.register_buffer("running_mean", torch.zeros(num_blocks, dim))
        self.register_buffer("running_var", torch.ones(num_blocks, dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        # x: (num_blocks, batch, dim)
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                bessel = B / (B - 1) if B > 1 else 1.0
                self.running_var.lerp_((var.squeeze(1) * bessel).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        d = scale.shape[-1]
        norm_factor = math.sqrt(d) * torch.rsqrt(torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True) + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class BlockRMSNorm(nn.Module):
    """Block-parallel RMSNorm over the feature dim, per block. ``normalize_parameters``
    projects gamma to norm ``sqrt(d)``."""

    def __init__(self, num_blocks: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_blocks, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale = self.weight.data
        d = scale.shape[-1]
        norm_factor = math.sqrt(d) * torch.rsqrt(torch.sum(scale * scale, dim=-1, keepdim=True) + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


# =============================================================================
#  Backbone (Fig. 2): BN-embed -> inverted-residual blocks -> RMSNorm
# =============================================================================
class FlashEmbedder(nn.Module):
    def __init__(self, num_blocks: int, in_features: int, hidden_dim: int):
        super().__init__()
        self.norm = BlockBatchNorm(num_blocks, in_features)
        self.w = BlockUnitLinear(num_blocks, in_features, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return self.w(self.norm(x, training))


class FlashBlock(nn.Module):
    """Inverted-residual block: w1 -> BN -> ReLU -> w2 -> BN -> ReLU, + residual."""

    def __init__(self, num_blocks: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = BlockUnitLinear(num_blocks, hidden_dim, hidden_dim * expansion)
        self.w2 = BlockUnitLinear(num_blocks, hidden_dim * expansion, hidden_dim)
        self.norm1 = BlockBatchNorm(num_blocks, hidden_dim * expansion)
        self.norm2 = BlockBatchNorm(num_blocks, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.w1(x), training))
        x = F.relu(self.norm2(self.w2(x), training))
        return x + residual


class FlashBackbone(nn.Module):
    def __init__(self, num_agents: int, in_dim: int, hidden_dim: int, num_blocks: int):
        super().__init__()
        self.num_agents = num_agents
        self.embedder = FlashEmbedder(num_agents, in_dim, hidden_dim)
        self.blocks = nn.ModuleList([FlashBlock(num_agents, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = BlockRMSNorm(num_agents, hidden_dim)

    def forward(self, flat: torch.Tensor, num_envs: int, training: bool) -> torch.Tensor:
        x = flat.view(self.num_agents, num_envs, -1)
        x = self.embedder(x, training)
        for block in self.blocks:
            x = block(x, training)
        return self.post_norm(x)  # (num_agents, num_envs, hidden)


# =============================================================================
#  Actor — state-dependent tanh-squashed Gaussian
# =============================================================================
class FlashSimBaActor(GaussianMixin, Model):
    """FlashSAC policy: state-dependent mean + log_std, tanh-squashed, block-parallel.

    Implements the skrl ``act()`` contract used by SAC/eval:
    ``act(inputs) -> (actions, {"log_prob", "mean_actions", "log_std"})``. A
    ``training`` flag may be passed via ``inputs["training"]`` (defaults to
    ``self.training``) so the caller controls BN batch-vs-running statistics; this is
    how the FlashSAC agent runs the cross-batch actor forward under batch stats.
    Continuous actions only (no Bernoulli/force-zero dims — matches the flat_full_random
    control interface)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        actor_n: int = 2,
        actor_latent: int = 128,
        min_log_std: float = -10.0,
        max_log_std: float = 2.0,
        last_layer_scale: float = 1.0,
        scale_down_action_dims: list[int] | None = None,
        **_ignored,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=True,
            min_log_std=min_log_std, max_log_std=max_log_std, reduction="sum",
        )
        self.num_agents = num_agents
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.backbone = FlashBackbone(num_agents, self.num_observations, actor_latent, actor_n).to(device)
        self.mean_w = BlockUnitLinear(num_agents, actor_latent, self.num_actions, bias=True).to(device)
        # Log-std head, FlashSAC-style: raw output -> tanh map into [min_log_std, max_log_std].
        # The bias initializes to 0 (BlockUnitLinear default), matching the reference
        # NormalTanhPolicy (std_bias = 0); at init raw ~ 0 -> sigma starts at the tanh midpoint
        # and the entropy mechanism grows it. No act_init_std targeting (the reference has none).
        self.logstd_w = BlockUnitLinear(num_agents, actor_latent, self.num_actions, bias=True).to(device)

        # Mean-head output scaling (SimBa's ``last_layer_scale``): shrink the action-mean
        # output by this factor so typical/initial actions start small (reduces tanh
        # saturation). Applied as a PERSISTENT forward-time multiply — NOT an init weight
        # scaling — because weight normalization would otherwise re-normalize it away each
        # step. ``scale_down_action_dims`` scales only those env-action indices (else all).
        # A per-dim (num_actions,) constant vector; plain attribute (not a buffer) so it is
        # not sliced/saved by the per-agent checkpoint helpers.
        scale_vec = torch.ones(self.num_actions, device=device)
        if last_layer_scale != 1.0:
            if scale_down_action_dims:
                idx = [d for d in scale_down_action_dims if 0 <= d < self.num_actions]
                scale_vec[idx] = last_layer_scale
            else:
                scale_vec[:] = last_layer_scale
        self._mean_scale = scale_vec  # (num_actions,), broadcasts over (N, ne, act)

        # For the SAC agent's continuous-L2 diagnostic (all dims are continuous here).
        self._cont_action_idx = torch.arange(self.num_actions, dtype=torch.long, device=device)

    def get_mean_and_std(self, obs: torch.Tensor, training: bool):
        num_envs = obs.size(0) // self.num_agents
        feat = self.backbone(obs, num_envs, training)          # (N, ne, hidden)
        mean = (self.mean_w(feat) * self._mean_scale).reshape(-1, self.num_actions)  # (N*ne, act)
        raw_log_std = self.logstd_w(feat).reshape(-1, self.num_actions)
        # Bounded log_std via tanh map into [min, max] (FlashSAC NormalTanhPolicy).
        log_std = self.min_log_std + (self.max_log_std - self.min_log_std) * 0.5 * (1.0 + torch.tanh(raw_log_std))
        return mean, log_std

    def compute(self, inputs, role):
        obs = inputs["observations"]
        mean, log_std = self.get_mean_and_std(obs, inputs.get("training", self.training))
        return mean, {"log_std": log_std}

    def act(self, inputs, *, role: str = ""):
        obs = inputs["observations"]
        mean, log_std = self.get_mean_and_std(obs, inputs.get("training", self.training))
        std = log_std.exp()
        dist = Normal(mean, std)
        self._g_distribution = dist

        taken_actions = inputs.get("taken_actions", None)
        if taken_actions is None:
            u = dist.rsample()
        else:
            u = safe_atanh(taken_actions)
        a = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True) - squash_log_prob_correction(u).unsqueeze(-1)

        outputs = {"log_std": log_std, "log_prob": log_prob, "mean_actions": torch.tanh(mean)}
        return a, outputs

    def get_entropy(self, *, role: str = ""):
        if getattr(self, "_g_distribution", None) is None:
            return torch.tensor(0.0, device=self.device)
        return self._g_distribution.entropy().to(self.device)


# =============================================================================
#  Critic — distributional (categorical / C51)
# =============================================================================
class FlashCategoricalQCritic(DeterministicMixin, Model):
    """FlashSAC Q critic: categorical distribution over ``n_atoms`` on ``[v_min, v_max]``.

    ``act(inputs) -> (expected_value, {"log_prob": (N*B, n_atoms)})`` where the info
    carries the per-atom log-probabilities the categorical Bellman loss needs. A
    ``training`` flag via ``inputs["training"]`` (default ``self.training``) selects BN
    batch-vs-running stats (the cross-batch update forces batch stats)."""

    atoms: torch.Tensor

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_agents: int = 1,
        critic_n: int = 2,
        critic_latent: int = 256,
        n_atoms: int = 101,
        v_min: float = -5.0,
        v_max: float = 5.0,
        clip_actions: bool = False,
        **_ignored,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_agents = num_agents
        self.n_atoms = n_atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        self.backbone = FlashBackbone(
            num_agents, self.num_observations + self.num_actions, critic_latent, critic_n
        ).to(device)
        self.head_w = BlockUnitLinear(num_agents, critic_latent, n_atoms, bias=True).to(device)
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms, device=device))

    def forward_dist(self, obs: torch.Tensor, actions: torch.Tensor, training: bool):
        """Return ``(expected_value (N*B,1), log_prob (N*B, n_atoms))``."""
        x = torch.cat([obs, actions], dim=-1)
        num_envs = x.size(0) // self.num_agents
        feat = self.backbone(x, num_envs, training)             # (N, ne, hidden)
        logits = self.head_w(feat).reshape(-1, self.n_atoms)    # (N*ne, n_atoms)
        log_prob = F.log_softmax(logits, dim=-1)
        value = (log_prob.exp() * self.atoms).sum(dim=-1, keepdim=True)
        return value, log_prob

    def act(self, inputs, *, role: str = ""):
        value, log_prob = self.forward_dist(
            inputs["observations"], inputs["taken_actions"], inputs.get("training", self.training)
        )
        return value, {"log_prob": log_prob}

    def compute(self, inputs, role):
        value, _ = self.forward_dist(
            inputs["observations"], inputs["taken_actions"], inputs.get("training", self.training)
        )
        return value, {}


# =============================================================================
#  Block-parallel cross-batch (cat/split) preserving per-agent ordering
# =============================================================================
def block_cross_cat(cur: torch.Tensor, nxt: torch.Tensor, num_agents: int) -> torch.Tensor:
    """Concatenate current + next rows per agent so a ``(num_agents, ne, dim)`` reshape
    keeps each agent's data contiguous.

    Inputs are ``(N*B, D)`` ordered ``[a0(B) | a1(B) | ...]``. Output is
    ``(N*2B, D)`` ordered ``[a0(2B) | a1(2B) | ...]`` with each agent block laid out
    ``[cur(B), nxt(B)]`` — so downstream BatchNorm (reducing over the batch dim per
    agent) shares statistics across the current/next halves within each agent."""
    D = cur.shape[-1]
    B = cur.shape[0] // num_agents
    cur = cur.view(num_agents, B, D)
    nxt = nxt.view(num_agents, B, D)
    return torch.cat([cur, nxt], dim=1).reshape(num_agents * 2 * B, D)


def block_cross_split(x: torch.Tensor, num_agents: int):
    """Inverse of :func:`block_cross_cat`. ``x`` is ``(N*2B, D)`` per-agent
    ``[cur(B), nxt(B)]``; returns ``(cur (N*B, D), nxt (N*B, D))`` in the original
    ``[a0(B) | a1(B) | ...]`` ordering."""
    two_b = x.shape[0] // num_agents
    B = two_b // 2
    D = x.shape[-1]
    x = x.view(num_agents, two_b, D)
    cur = x[:, :B, :].reshape(num_agents * B, D)
    nxt = x[:, B:, :].reshape(num_agents * B, D)
    return cur, nxt


# =============================================================================
#  Categorical (C51) target projection, cross-entropy, min-select
# =============================================================================
def categorical_td_target(
    next_log_probs: torch.Tensor,   # (N*B, n_atoms) chosen critic's next-state log-probs
    reward: torch.Tensor,           # (N*B, 1) or (N*B,)
    done: torch.Tensor,             # (N*B, 1) or (N*B,)
    actor_entropy: torch.Tensor,    # (N*B, 1) or (N*B,)  == alpha * log_pi(a'|s')
    gamma: float,
    n_atoms: int,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """Project the distributional Bellman target (max-ent term folded into the support)
    onto the fixed atom grid. Returns target probabilities ``(N*B, n_atoms)``."""
    # Do the discrete projection in float32: the scatter_add requires target and src
    # to share a dtype, and fp16 bin projection is numerically poor. Casting here also
    # makes the function safe under autocast/mixed precision.
    next_log_probs = next_log_probs.float()
    reward = reward.reshape(-1, 1).float()
    done = done.reshape(-1, 1).float()
    actor_entropy = actor_entropy.reshape(-1, 1).float()

    bin_width = (v_max - v_min) / (n_atoms - 1)
    bin_values = torch.linspace(
        v_min, v_max, n_atoms, device=next_log_probs.device, dtype=torch.float32
    ).view(1, -1)

    target_bins = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bins = torch.clamp(target_bins, v_min, v_max)

    b = (target_bins - v_min) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, n_atoms - 1)
    frac = b - lower.float()

    p = next_log_probs.exp()
    m_l = p * (1.0 - frac)
    m_u = p * frac
    target = torch.zeros_like(p)
    target.scatter_add_(1, lower, m_l)
    target.scatter_add_(1, upper, m_u)
    return target


def categorical_ce_loss(target_probs: torch.Tensor, pred_log_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of one critic's predicted log-probs against the (detached) target
    distribution. ``target_probs`` / ``pred_log_probs`` are ``(N*B, n_atoms)``."""
    return -(target_probs.float() * pred_log_probs.float()).sum(dim=-1).mean()


def min_q_log_probs(q1: torch.Tensor, q2: torch.Tensor,
                    log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
    """Per-sample select the log-prob distribution of the critic with the smaller
    expected Q. ``q*`` are ``(N*B, 1)``; ``log_prob*`` are ``(N*B, n_atoms)``."""
    pick2 = (q2 < q1).reshape(-1, 1)
    return torch.where(pick2, log_prob2, log_prob1)


# =============================================================================
#  Weight normalization sweep
# =============================================================================
@torch.no_grad()
def normalize_flash_parameters(*modules: nn.Module) -> None:
    """Call ``normalize_parameters`` on every submodule that defines it (unit-sphere
    projection for linears, sqrt(d) projection for norm params). Run after each
    optimizer step."""
    for m in modules:
        if m is None:
            continue
        for sub in m.modules():
            fn = getattr(sub, "normalize_parameters", None)
            if callable(fn) and sub is not m:
                fn()


# =============================================================================
#  Noise repetition (Zeta-distributed repeat lengths)
# =============================================================================
def build_zeta_cdf(mu: float, max_n: int, device) -> torch.Tensor:
    """Truncated Zeta CDF over repeat lengths ``1..max_n`` with pmf ``~ n^(-mu)``."""
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / pmf.sum()
    return torch.cumsum(pmf, dim=0)


def sample_zeta_lengths(cdf: torch.Tensor, shape) -> torch.Tensor:
    """Sample repeat lengths (int) from a Zeta CDF, one per element of ``shape``."""
    u = torch.rand(shape, device=cdf.device).unsqueeze(-1)          # (*shape, 1)
    # count how many CDF thresholds u exceeds -> index; +1 for 1-based length
    return (u >= cdf).sum(dim=-1).to(torch.long) + 1


# =============================================================================
#  Adaptive reward scaling (Eq. 6), per agent, reusing skrl RunningStandardScaler
# =============================================================================
class FlashReturnScaler:
    """Per-agent adaptive reward scaling from the running discounted-return variance.

    Maintains per-env discounted return ``G_r <- gamma*(1-done)*G_r + r`` and, per
    agent, the running variance of ``G_r`` (via skrl's ``RunningStandardScaler``, so we
    do not reimplement Welford) and the running max magnitude ``G_r_max``. The scale
    applied to sampled rewards is ``r / max(sqrt(var+eps), G_r_max / g_max)`` (Eq. 6)."""

    def __init__(self, num_agents: int, envs_per_agent: int, gamma: float, g_max: float,
                 device, eps: float = 1e-8):
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        self.num_agents = num_agents
        self.envs_per_agent = envs_per_agent
        self.gamma = float(gamma)
        self.g_max = float(g_max)
        self.eps = float(eps)
        self.device = device
        self.G_r = torch.zeros(num_agents * envs_per_agent, device=device)
        self.G_r_max = torch.zeros(num_agents, device=device)
        self._rms = [RunningStandardScaler(size=1, device=device) for _ in range(num_agents)]

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        r = rewards.reshape(-1).float()
        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1)).float()
        self.G_r = self.gamma * (1.0 - done) * self.G_r + r
        epa = self.envs_per_agent
        for a in range(self.num_agents):
            sl = self.G_r[a * epa:(a + 1) * epa].view(-1, 1)
            self._rms[a](sl, train=True)   # updates running mean/variance
            self.G_r_max[a] = torch.maximum(self.G_r_max[a], sl.abs().max())

    @torch.no_grad()
    def denominators(self) -> torch.Tensor:
        """Per-agent Eq. 6 denominator, shape (num_agents,)."""
        out = []
        for a in range(self.num_agents):
            var = self._rms[a].running_variance.reshape(())
            out.append(torch.maximum(torch.sqrt(var + self.eps), self.G_r_max[a] / self.g_max))
        return torch.stack(out)

    def scale(self, rewards_flat: torch.Tensor, B: int) -> torch.Tensor:
        """Scale sampled rewards ``(N*B, 1)`` (per-agent ordering) by the current per-agent
        denominators. Differentiable-safe (denominators are detached constants).

        NOTE: skrl's ``RunningStandardScaler`` keeps its running stats in float64, so
        ``denominators()`` is float64. We cast the result back to the reward's dtype
        (float32) so the scaled reward does NOT silently upcast to float64 and propagate
        into the critic target / replay buffer / logging."""
        denom = self.denominators().detach().to(rewards_flat.dtype).view(self.num_agents, 1, 1)
        return (rewards_flat.view(self.num_agents, B, 1) / denom).reshape(-1, 1)

    def state_dict(self) -> dict:
        return {
            "G_r": self.G_r.detach().cpu(),
            "G_r_max": self.G_r_max.detach().cpu(),
            "rms": [
                {"mean": s.running_mean.detach().cpu(),
                 "var": s.running_variance.detach().cpu(),
                 "count": s.current_count.detach().cpu() if hasattr(s, "current_count") else None}
                for s in self._rms
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        self.G_r = state["G_r"].to(self.device)
        self.G_r_max = state["G_r_max"].to(self.device)
        for s, saved in zip(self._rms, state["rms"]):
            s.running_mean.copy_(saved["mean"].to(self.device))
            s.running_variance.copy_(saved["var"].to(self.device))
            if saved.get("count") is not None and hasattr(s, "current_count"):
                s.current_count.copy_(saved["count"].to(self.device))

"""FlashSAC: ``FlashSAC(SAC)`` overriding only the hooks it changes.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/flash_sac_agent.py``.
Everything else — optimizers, memory, temperature, checkpoints, per-agent clipping — is SAC's.
All FlashSAC tensor math lives in ``models/flash_sac.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import gymnasium
import numpy as np
import torch
from torch.distributions import Normal

from ..models.flash_sac import (
    FlashReturnScaler,
    block_cross_cat,
    block_cross_split,
    build_zeta_cdf,
    categorical_ce_loss,
    categorical_td_target,
    min_q_log_probs,
    normalize_flash_parameters,
    sample_zeta_lengths,
    squash_log_prob_correction,
)
from .sac import SAC

__all__ = ["FlashSAC"]


class FlashSAC(SAC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._n_step = 1  # n-step return exponent for the categorical target (paper uses 1)

        # FlashSAC's unified entropy target: 0.5 * |A| * log(2*pi*e*sigma^2)
        if (
            self.cfg.learn_entropy
            and self.cfg.target_entropy is None
            and self.cfg.target_entropy_mode == "unified"
        ):
            if isinstance(self.action_space, gymnasium.spaces.Box):
                action_dim = float(np.prod(self.action_space.shape))
            else:
                action_dim = 1.0
            sigma = float(self.cfg.entropy_sigma_target)
            self._target_entropy = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * sigma * sigma)

        self._flash_grad_steps = 0

        # noise repetition state, sized on first use
        self._zeta_cdf = build_zeta_cdf(
            self.cfg.noise_repeat_zeta_mu, self.cfg.noise_repeat_max, self.device
        )
        self._nr_noise = None
        self._nr_count = None
        self._nr_len = None

        self._return_scaler: FlashReturnScaler | None = None

    def _learner_name(self) -> str:
        return "flash_sac"

    # ------------------------------------------------------------------ rollout
    def _sample_rollout_action(self, inputs: dict, *, timestep: int):
        """Temporally-correlated noise: hold a noise draw for a Zeta-sampled number of steps."""
        if not self.cfg.noise_repeat_enabled or not self.training:
            return self.policy.act(inputs, role="policy")

        mean, log_std = self.policy.get_mean_and_std(
            inputs["observations"], training=self.policy.training
        )
        std = log_std.exp()
        rows = mean.shape[0]

        if self._nr_noise is None or self._nr_noise.shape != mean.shape:
            self._nr_noise = torch.randn_like(mean)
            self._nr_count = torch.zeros(rows, 1, dtype=torch.long, device=self.device)
            self._nr_len = sample_zeta_lengths(self._zeta_cdf, (rows,)).view(rows, 1)

        reinit = (self._nr_count == 0) | (self._nr_count >= self._nr_len)
        self._nr_noise = torch.where(reinit, torch.randn_like(mean), self._nr_noise)
        self._nr_len = torch.where(
            reinit, sample_zeta_lengths(self._zeta_cdf, (rows,)).view(rows, 1), self._nr_len
        )
        self._nr_count = torch.where(reinit, torch.zeros_like(self._nr_count), self._nr_count) + 1

        u = mean + std * self._nr_noise
        log_prob = Normal(mean, std).log_prob(u).sum(dim=-1, keepdim=True) - squash_log_prob_correction(
            u
        ).unsqueeze(-1)
        return torch.tanh(u), {
            "log_prob": log_prob,
            "mean_actions": torch.tanh(mean),
            "log_std": log_std,
        }

    def _update_return_stats(self, *, rewards, terminated, truncated) -> None:
        """Running per-agent discounted-return statistics for adaptive reward scaling."""
        if not self.training or not self.cfg.reward_scaling_enabled:
            return
        if self._return_scaler is None:
            self._return_scaler = FlashReturnScaler(
                num_agents=self.num_agents,
                envs_per_agent=rewards.shape[0] // self.num_agents,
                gamma=self.cfg.discount_factor,
                g_max=self.cfg.reward_scaling_g_max,
                device=self.device,
                eps=self.cfg.reward_scaling_eps,
            )
        self._return_scaler.update(rewards, terminated, truncated)

    # ------------------------------------------------------------------ update hooks
    def _should_update_actor(self, gradient_step: int) -> bool:
        period = max(int(self.cfg.actor_update_period), 1)
        ran = (self._flash_grad_steps % period) == 0
        self._flash_grad_steps += 1
        return ran

    def _post_optimizer_step(self) -> None:
        if self.cfg.weight_norm_enabled:
            normalize_flash_parameters(self.policy, self.critic_1, self.critic_2)

    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, rows):
        """Cross-batch categorical (C51) Bellman loss with the EMA target critics."""
        num_agents = self.num_agents
        rewards = sampled["rewards"]
        if self.cfg.reward_scaling_enabled and self._return_scaler is not None:
            rewards = self._return_scaler.scale(rewards, rows)

        gamma_n = self.cfg.discount_factor**self._n_step
        n_atoms = self.critic_1.n_atoms
        v_min, v_max = self.critic_1.v_min, self.critic_1.v_max

        with torch.no_grad():
            next_actions, next_out = self.policy.act(
                {"observations": next_inputs["observations"], "training": False}, role="policy"
            )
            entropy = self.expand_per_agent(self._entropy_coefficient, rows)
            actor_entropy = entropy * next_out["log_prob"]

            # one forward over [current ; next] per agent, so both halves share BN batch stats
            obs_all = block_cross_cat(critic_inputs["observations"], critic_next_inputs["observations"], num_agents)
            act_all = block_cross_cat(sampled["actions"], next_actions, num_agents)

            tq1_all, tlp1_all = self.target_critic_1.forward_dist(obs_all, act_all, training=True)
            tq2_all, tlp2_all = self.target_critic_2.forward_dist(obs_all, act_all, training=True)
            _, tq1_next = block_cross_split(tq1_all, num_agents)
            _, tq2_next = block_cross_split(tq2_all, num_agents)
            _, tlp1_next = block_cross_split(tlp1_all, num_agents)
            _, tlp2_next = block_cross_split(tlp2_all, num_agents)

            target_probs = categorical_td_target(
                next_log_probs=min_q_log_probs(tq1_next, tq2_next, tlp1_next, tlp2_next),
                reward=rewards,
                done=sampled["terminated"].float(),
                actor_entropy=actor_entropy,
                gamma=gamma_n,
                n_atoms=n_atoms,
                v_min=v_min,
                v_max=v_max,
            )

        pq1_all, plp1_all = self.critic_1.forward_dist(obs_all, act_all, training=True)
        pq2_all, plp2_all = self.critic_2.forward_dist(obs_all, act_all, training=True)
        plp1_cur, _ = block_cross_split(plp1_all, num_agents)
        plp2_cur, _ = block_cross_split(plp2_all, num_agents)
        critic_loss = 0.5 * (
            categorical_ce_loss(target_probs, plp1_cur) + categorical_ce_loss(target_probs, plp2_cur)
        )

        q1_cur, _ = block_cross_split(pq1_all, num_agents)
        q2_cur, _ = block_cross_split(pq2_all, num_agents)
        target_values = (target_probs * self.critic_1.atoms).sum(dim=-1, keepdim=True)
        return critic_loss, q1_cur, q2_cur, target_values

    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, rows):
        """Cross-batch actor forward + expected-value Q."""
        num_agents = self.num_agents
        next_actor_obs = self.normalize_observations(sampled["next_observations"], train=True)
        obs_all = block_cross_cat(inputs["observations"], next_actor_obs, num_agents)
        actions_all, out_all = self.policy.act(
            {"observations": obs_all, "training": True}, role="policy"
        )
        actions, _ = block_cross_split(actions_all, num_agents)
        log_prob, _ = block_cross_split(out_all["log_prob"], num_agents)
        mean_actions, _ = block_cross_split(out_all["mean_actions"], num_agents)
        log_std, _ = block_cross_split(out_all["log_std"], num_agents)

        q1, _ = self.critic_1.forward_dist(critic_inputs["observations"], actions, training=False)
        q2, _ = self.critic_2.forward_dist(critic_inputs["observations"], actions, training=False)

        entropy = self.expand_per_agent(self._entropy_coefficient, rows)
        policy_loss = (entropy * log_prob - torch.min(q1, q2)).mean()
        outputs = {"log_prob": log_prob, "mean_actions": mean_actions, "log_std": log_std}
        return policy_loss, actions, log_prob, q1, q2, outputs

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_extras(self, agent: int) -> Dict[str, Any]:
        extras = super()._checkpoint_extras(agent)
        extras["return_scaler"] = (
            self._return_scaler.state_dict() if self._return_scaler is not None else None
        )
        return extras

    def _load_extras(self, agent: int, extras: Dict[str, Any], path) -> None:
        saved = extras.pop("return_scaler", None)
        super()._load_extras(agent, extras, path)
        if saved is not None and self._return_scaler is not None:
            self._return_scaler.load_state_dict(saved)

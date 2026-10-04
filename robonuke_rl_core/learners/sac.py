"""Block-parallel SAC.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/sac.py``, minus AMP,
the distributed branches, the auxiliary-loss manager, the contact/rotation supervision and
the per-axis action metrics. Changed for per-agent independence: gradients are clipped per
agent and the observation/state normalizers are per agent (``BlockRunningNorm``).

Every tensor op either keeps the rows separate (the block networks reduce within a block) or
reduces with a constant factor, so agent ``i``'s update never depends on agent ``j``'s data.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F

from ..models.block_utils import clip_grad_norm_per_agent
from ..models.factory import build_models
from ..models.normalizer import BlockRunningNorm
from ..losses.losses import LossContext
from .base import LearnerBase

__all__ = ["SAC"]


class SAC(LearnerBase):
    def __init__(self, *, model_cfg=None, **kwargs) -> None:
        """``model_cfg`` is only needed when ``sac.periodic_reset_enabled`` rebuilds models."""
        super().__init__(**kwargs)
        self._model_cfg = model_cfg

        required = ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2")
        missing = [k for k in required if self.models.get(k) is None]
        if missing:
            raise ValueError(f"SAC requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]

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
            # no weight decay on log_alpha: pulling it toward 0 has no meaning
            self.entropy_optimizer = torch.optim.AdamW(
                [self.log_entropy_coefficient], lr=self.cfg.entropy_lr, weight_decay=0.0
            )

        self._build_optimizers()

        self.target_critic_1.freeze_parameters(True)
        self.target_critic_2.freeze_parameters(True)
        self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        if self.cfg.normalize_observations:
            self.observation_normalizer = self.make_normalizer(self.observation_space)
            self.state_normalizer = (
                self.make_normalizer(self.state_space) if self._asymmetric else None
            )
        else:
            self.observation_normalizer = None
            self.state_normalizer = None

    # ------------------------------------------------------------------ setup helpers
    def _resolve_target_entropy(self) -> float:
        if self.cfg.target_entropy is not None:
            return float(self.cfg.target_entropy)
        if isinstance(self.action_space, gymnasium.spaces.Box):
            return float(-np.prod(self.action_space.shape))
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            return float(-self.action_space.n)
        raise ValueError(
            f"cannot derive a target entropy for action space {self.action_space}; set "
            "sac.target_entropy"
        )

    def _build_optimizers(self) -> None:
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.actor_lr, weight_decay=self.cfg.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.policy_scheduler = None
        self.critic_scheduler = None
        self._lr_built = False

    def _build_cosine_lr(self, timesteps: int) -> None:
        """Cosine LR for actor and critic, decayed to ``lr_end`` over the run's gradient steps."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        update_steps = max(1, int(timesteps) - int(self.cfg.learning_starts))
        t_max = max(1, update_steps * int(self.cfg.gradient_steps))
        self.policy_scheduler = CosineAnnealingLR(
            self.policy_optimizer, T_max=t_max, eta_min=self.cfg.lr_end
        )
        self.critic_scheduler = CosineAnnealingLR(
            self.critic_optimizer, T_max=t_max, eta_min=self.cfg.lr_end
        )

    def _create_memory_tensors(self) -> None:
        if self.memory is None:
            raise ValueError("SAC needs a replay memory")
        self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self._tensors_names = ["observations", "actions", "rewards", "next_observations", "terminated"]
        if self._asymmetric:
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self._tensors_names += ["states", "next_states"]

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
    def act(self, observations, states, *, timestep: int, timesteps: int):
        if self.training and timestep < self.cfg.random_timesteps:
            return self.random_actions(observations.shape[0]), {}
        inputs = {"observations": self.normalize_observations(observations)}
        # no_grad: these actions only drive the env; the update re-runs the policy on replay
        # batches. A live graph here would be spliced into the env's action buffers and grow.
        with torch.no_grad():
            return self._sample_rollout_action(inputs, timestep=timestep)

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
        self._forward_env_metrics(infos, timestep)
        self._track_episodes(rewards, terminated, truncated)
        self._update_return_stats(rewards=rewards, terminated=terminated, truncated=truncated)

        if not self.training:
            return
        extra: Dict[str, torch.Tensor] = {}
        if self._asymmetric:
            if states is None or next_states is None:
                raise RuntimeError(
                    "asymmetric SAC needs states and next_states from the trainer "
                    f"(got states={states is not None}, next_states={next_states is not None})"
                )
            extra["states"] = states
            extra["next_states"] = next_states
        self.memory.add_samples(
            observations=observations,
            actions=actions,
            rewards=self.shape_rewards(rewards, timestep, timesteps),
            next_observations=next_observations,
            terminated=terminated,
            **extra,
        )

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        if timestep >= self.cfg.learning_starts:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)
        self._maybe_periodic_reset(timestep)

    # ------------------------------------------------------------------ hooks FlashSAC overrides
    def _sample_rollout_action(self, inputs: dict, *, timestep: int):
        """Rollout sampling. Base: a stochastic squashed-Gaussian sample."""
        return self.policy.act(inputs, role="policy")

    def _should_update_actor(self, gradient_step: int) -> bool:
        """Actor-update gate. Base: every gradient step."""
        return True

    def _post_optimizer_step(self) -> None:
        """After both optimizer steps and the target update. Base: nothing."""

    def _update_return_stats(self, *, rewards, terminated, truncated) -> None:
        """Rollout statistics hook. Base: nothing (FlashSAC's reward scaling uses it)."""

    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, rows):
        """Min-twin bootstrapped target + MSE over both critics."""
        with torch.no_grad():
            next_actions, outputs = self.policy.act(next_inputs, role="policy")
            target_q1, _ = self.target_critic_1.act(
                {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_1"
            )
            target_q2, _ = self.target_critic_2.act(
                {**critic_next_inputs, "taken_actions": next_actions}, role="target_critic_2"
            )
            entropy = self.expand_per_agent(self._entropy_coefficient, rows)
            target_q = torch.min(target_q1, target_q2) - entropy * outputs["log_prob"]
            target_values = sampled["rewards"] + self.cfg.discount_factor * sampled[
                "terminated"
            ].logical_not() * target_q

        critic_1_values, _ = self.critic_1.act(
            {**critic_inputs, "taken_actions": sampled["actions"]}, role="critic_1"
        )
        critic_2_values, _ = self.critic_2.act(
            {**critic_inputs, "taken_actions": sampled["actions"]}, role="critic_2"
        )
        critic_loss = (
            F.mse_loss(critic_1_values, target_values) + F.mse_loss(critic_2_values, target_values)
        ) / 2
        return critic_loss, critic_1_values, critic_2_values, target_values

    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, rows):
        """``alpha * log_pi - min(Q1, Q2)``."""
        actions, outputs = self.policy.act(inputs, role="policy")
        log_prob = outputs["log_prob"]
        critic_1_pi, _ = self.critic_1.act({**critic_inputs, "taken_actions": actions}, role="critic_1")
        critic_2_pi, _ = self.critic_2.act({**critic_inputs, "taken_actions": actions}, role="critic_2")
        entropy = self.expand_per_agent(self._entropy_coefficient, rows)
        policy_loss = (entropy * log_prob - torch.min(critic_1_pi, critic_2_pi)).mean()
        return policy_loss, actions, log_prob, critic_1_pi, critic_2_pi, outputs

    # ------------------------------------------------------------------ update
    def update(self, *, timestep: int, timesteps: int) -> None:
        rows = self.cfg.batch_size  # per agent; the memory returns N * batch_size rows
        if self.cfg.lr_schedule == "cosine" and not self._lr_built:
            self._build_cosine_lr(timesteps)
            self._lr_built = True

        for gradient_step in range(self.cfg.gradient_steps):
            sampled = dict(
                zip(
                    self._tensors_names,
                    self.memory.sample(names=self._tensors_names, batch_size=rows)[0],
                )
            )
            inputs = {"observations": self.normalize_observations(sampled["observations"], train=True)}
            next_inputs = {
                "observations": self.normalize_observations(sampled["next_observations"], train=True)
            }
            if self._asymmetric:
                critic_inputs = {"observations": self.normalize_states(sampled["states"], train=True)}
                critic_next_inputs = {
                    "observations": self.normalize_states(sampled["next_states"], train=True)
                }
            else:
                critic_inputs, critic_next_inputs = inputs, next_inputs

            critic_loss, critic_1_values, critic_2_values, target_values = self._compute_critic_loss(
                sampled=sampled,
                inputs=inputs,
                next_inputs=next_inputs,
                critic_inputs=critic_inputs,
                critic_next_inputs=critic_next_inputs,
                rows=rows,
            )
            aux = self.compute_aux_loss(
                LossContext(
                    learner=self,
                    step=timestep,
                    target="critic",
                    sampled=sampled,
                    critic_inputs=critic_inputs,
                    critic_1_values=critic_1_values,
                    critic_2_values=critic_2_values,
                    target_values=target_values,
                )
            )
            if aux is not None:
                critic_loss = critic_loss + aux

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norms = clip_grad_norm_per_agent(
                [self.critic_1, self.critic_2], self.num_agents, self.cfg.grad_norm_clip
            )
            self.critic_optimizer.step()

            actor_ran = self._should_update_actor(gradient_step)
            policy_grad_norms = log_prob = None
            if actor_ran:
                (
                    policy_loss,
                    actions,
                    log_prob,
                    critic_1_pi,
                    critic_2_pi,
                    outputs,
                ) = self._compute_actor_loss(
                    sampled=sampled, inputs=inputs, critic_inputs=critic_inputs, rows=rows
                )
                aux = self.compute_aux_loss(
                    LossContext(
                        learner=self,
                        step=timestep,
                        target="policy",
                        sampled=sampled,
                        inputs=inputs,
                        actions=actions,
                        log_prob=log_prob,
                        policy_outputs=outputs,
                    )
                )
                if aux is not None:
                    policy_loss = policy_loss + aux

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_grad_norms = clip_grad_norm_per_agent(
                    self.policy, self.num_agents, self.cfg.grad_norm_clip
                )
                self.policy_optimizer.step()

                if self.cfg.learn_entropy:
                    self._update_entropy(log_prob, rows)

            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
            self._post_optimizer_step()

            if self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

            if self.on_log:
                self._emit_update_metrics(
                    timestep=timestep,
                    rows=rows,
                    critic_1_values=critic_1_values,
                    critic_2_values=critic_2_values,
                    target_values=target_values,
                    critic_grad_norms=critic_grad_norms,
                    policy_grad_norms=policy_grad_norms,
                    log_prob=log_prob,
                )

    def _update_entropy(self, log_prob: torch.Tensor, rows: int) -> None:
        """Per-agent temperature step: each agent's loss term holds only its own rows."""
        log_prob_per_agent = log_prob.view(self.num_agents, rows, 1).mean(dim=1)  # (N, 1)
        if self.cfg.entropy_loss_form == "alpha":
            # FlashSAC's form: alpha * (H_hat - H_target), adaptation rate ~ alpha
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
        self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

    def _emit_update_metrics(
        self,
        *,
        timestep: int,
        rows: int,
        critic_1_values,
        critic_2_values,
        target_values,
        critic_grad_norms,
        policy_grad_norms,
        log_prob,
    ) -> None:
        with torch.no_grad():
            metrics = {
                "loss/critic": 0.5
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
                "lr/actor": torch.full(
                    (self.num_agents,),
                    self.policy_optimizer.param_groups[0]["lr"],
                    device=self.device,
                ),
                "lr/critic": torch.full(
                    (self.num_agents,),
                    self.critic_optimizer.param_groups[0]["lr"],
                    device=self.device,
                ),
            }
            if policy_grad_norms is not None:
                metrics["grad_norm/policy"] = policy_grad_norms
            if log_prob is not None:
                metrics["policy/log_prob"] = self.per_agent_mean(log_prob, rows)
            self.emit_per_agent(metrics, timestep)

    # ------------------------------------------------------------------ SimBa periodic reset
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
            self._learner_name(),
            self._model_cfg,
            self.observation_space,
            self.state_space,
            self.action_space,
            self.num_agents,
            self.device,
        )
        for key in ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"):
            self.models[key] = fresh[key]
            setattr(self, key, fresh[key])
        self.target_critic_1.freeze_parameters(True)
        self.target_critic_2.freeze_parameters(True)
        self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        self._build_optimizers()
        self._entropy_coefficient = torch.full(
            (self.num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            self.log_entropy_coefficient = torch.log(
                self._entropy_coefficient.clone()
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.AdamW(
                [self.log_entropy_coefficient], lr=self.cfg.entropy_lr, weight_decay=0.0
            )

    def _learner_name(self) -> str:
        return "sac"

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_model_keys(self) -> List[str]:
        # the target critics are saved too: resuming must keep their EMA lag, and loading one
        # slot must not touch any other agent's targets
        return ["policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"]

    def _checkpoint_optimizer_keys(self) -> List[str]:
        return ["policy_optimizer", "critic_optimizer"]

    def _checkpoint_normalizers(self) -> Dict[str, BlockRunningNorm]:
        normalizers = {}
        if self.observation_normalizer is not None:
            normalizers["observations"] = self.observation_normalizer
        if self.state_normalizer is not None:
            normalizers["states"] = self.state_normalizer
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

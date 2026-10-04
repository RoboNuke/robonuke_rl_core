"""Block-parallel PPO.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/ppo.py``, minus AMP, the
distributed branches, the auxiliary-loss manager and the KL-adaptive LR (one optimizer over
block parameters holds a single LR, so a data-driven LR would couple the agents). Changed for
per-agent independence: per-agent gradient clipping, a per-agent value normalizer, and the
per-agent KL early stop from RoboNuke/Continuous_Force_RL ``agents/block_ppo.py``
(``keep_mask``). ``value_update_ratio`` comes from the same file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..models.block_utils import clip_grad_norm_per_agent, step_with_frozen_agents
from ..models.normalizer import BlockRunningNorm
from ..losses.losses import LossContext
from .base import LearnerBase

__all__ = ["PPO", "compute_gae"]


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float,
    lambda_coefficient: float,
    time_limit_bootstrap: bool,
    num_agents: int,
    envs_per_agent: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with **per-agent** advantage normalization.

    Tensors are ``(rollout, total_envs, 1)`` with the env axis laid out
    ``[agent 0 envs, agent 1 envs, ...]``. GAE runs per env over time; the advantages are then
    standardized within each agent's block, so one agent's returns never shift another's
    advantages (stock PPO normalizes globally).
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_done = ((terminated | truncated) if time_limit_bootstrap else terminated).logical_not()
    memory_size = rewards.shape[0]

    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage

    returns = advantages + values

    rollout = advantages.shape[0]
    blocks = advantages.view(rollout, num_agents, envs_per_agent, -1)
    mean = blocks.mean(dim=(0, 2), keepdim=True)
    std = blocks.std(dim=(0, 2), keepdim=True)
    advantages = ((blocks - mean) / (std + 1e-8)).view(rollout, num_agents * envs_per_agent, -1)
    return returns, advantages


class PPO(LearnerBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        required = ("policy", "value")
        missing = [k for k in required if self.models.get(k) is None]
        if missing:
            raise ValueError(f"PPO requires models {required}; missing or None: {missing}")
        self.policy = self.models["policy"]
        self.value = self.models["value"]

        self._asymmetric = self.state_space is not None

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.policy_lr, weight_decay=self.cfg.weight_decay
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=self.cfg.value_lr, weight_decay=self.cfg.weight_decay
        )
        self.policy_scheduler = None
        self.value_scheduler = None
        self._lr_built = False

        if self.cfg.normalize_observations:
            self.observation_normalizer = self.make_normalizer(self.observation_space)
            self.state_normalizer = (
                self.make_normalizer(self.state_space) if self._asymmetric else None
            )
        else:
            self.observation_normalizer = None
            self.state_normalizer = None
        # per agent, unlike V's single shared value scaler
        self.value_normalizer = (
            BlockRunningNorm(
                self.num_agents, 1, clip_threshold=self.cfg.normalizer_clip, device=self.device
            )
            if self.cfg.normalize_values
            else None
        )

        self._rollout = 0
        self._current_log_prob = None
        self._current_values = None
        self._next_observations = None
        self._next_states = None

    # ------------------------------------------------------------------ setup
    def _create_memory_tensors(self) -> None:
        if self.memory is None:
            raise ValueError("PPO needs a rollout memory")
        self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
        self._tensors_names = ["observations", "actions", "log_prob", "values", "returns", "advantages"]
        if self._asymmetric:
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self._tensors_names.insert(1, "states")

    def _build_cosine_lr(self, timesteps: int) -> None:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        updates = max(1, (int(timesteps) - int(self.cfg.learning_starts)) // max(1, self.cfg.rollouts))
        t_max = max(1, updates * self.cfg.learning_epochs)
        self.policy_scheduler = CosineAnnealingLR(
            self.policy_optimizer, T_max=t_max, eta_min=self.cfg.lr_end
        )
        self.value_scheduler = CosineAnnealingLR(
            self.value_optimizer, T_max=t_max, eta_min=self.cfg.lr_end
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

    def normalize_values(self, values: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Per-agent value normalization for flat ``(N*rows, 1)`` tensors."""
        if self.value_normalizer is None:
            return values
        return self.value_normalizer(values, train=train, inverse=inverse)

    def normalize_value_block(self, values: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Same, for a rollout-shaped ``(T, total_envs, 1)`` tensor (agent axis is dim 1)."""
        if self.value_normalizer is None:
            return values
        rollout, envs, dim = values.shape
        # env-major flattening puts every agent's rows together, which is what the normalizer wants
        flat = values.permute(1, 0, 2).reshape(envs * rollout, dim)
        out = self.value_normalizer(flat, train=train, inverse=inverse)
        return out.view(envs, rollout, dim).permute(1, 0, 2)

    def _value_inputs(self, observations: torch.Tensor, states: torch.Tensor, *, train: bool = False) -> dict:
        if self._asymmetric:
            return {"observations": self.normalize_states(states, train=train)}
        return {"observations": self.normalize_observations(observations, train=train)}

    # ------------------------------------------------------------------ interaction
    def act(self, observations, states, *, timestep: int, timesteps: int):
        inputs = {"observations": self.normalize_observations(observations)}
        random = self.training and timestep < self.cfg.random_timesteps
        with torch.no_grad():
            if random:
                actions = self.random_actions(observations.shape[0])
                _, outputs = self.policy.act({**inputs, "taken_actions": actions}, role="policy")
            else:
                actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]
            if self.training:
                values, _ = self.value.act(self._value_inputs(observations, states), role="value")
                self._current_values = self.normalize_values(values, inverse=True)
        return actions, outputs

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

        if not self.training:
            return
        self._next_observations = next_observations
        self._next_states = next_states

        shaped = self.shape_rewards(rewards, timestep, timesteps)
        if self.cfg.time_limit_bootstrap and bool(truncated.any()):
            with torch.no_grad():
                next_values, _ = self.value.act(
                    self._value_inputs(next_observations, next_states), role="value"
                )
                next_values = self.normalize_values(next_values, inverse=True)
            shaped = shaped + self.cfg.discount_factor * next_values * truncated

        samples: Dict[str, torch.Tensor] = dict(
            observations=observations,
            actions=actions,
            rewards=shaped,
            terminated=terminated,
            truncated=truncated,
            log_prob=self._current_log_prob,
            values=self._current_values,
        )
        if self._asymmetric:
            if states is None:
                raise RuntimeError("asymmetric PPO needs states from the trainer (env.state())")
            samples["states"] = states
        self.memory.add_samples(**samples)

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        if self._rollout % self.cfg.rollouts == 0 and timestep >= self.cfg.learning_starts:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)

    # ------------------------------------------------------------------ update
    def update(self, *, timestep: int, timesteps: int) -> None:
        num_agents = self.num_agents
        if self.cfg.lr_schedule == "cosine" and not self._lr_built:
            self._build_cosine_lr(timesteps)
            self._lr_built = True

        with torch.no_grad():
            last_values, _ = self.value.act(
                self._value_inputs(self._next_observations, self._next_states), role="value"
            )
            last_values = self.normalize_values(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
            num_agents=num_agents,
            envs_per_agent=self.memory.num_envs // num_agents,
        )
        self.memory.set_tensor_by_name("values", self.normalize_value_block(values, train=True))
        self.memory.set_tensor_by_name("returns", self.normalize_value_block(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        accumulated: Dict[str, List[torch.Tensor]] = {}

        def accumulate(name: str, value: torch.Tensor) -> None:
            accumulated.setdefault(name, []).append(value.detach())

        for epoch in range(self.cfg.learning_epochs):
            # Per-agent KL early stop: once an agent is dropped it stays frozen for this epoch.
            # Frozen means its policy weights and Adam moments do not change at all (see
            # step_with_frozen_agents). The value network is not masked: every agent's critic
            # trains on every minibatch, so no agent's critic depends on another agent's KL.
            keep = torch.ones(num_agents, dtype=torch.bool, device=self.device)
            any_kept = True

            for batch in self.memory.sample_all(
                names=self._tensors_names, mini_batches=self.cfg.mini_batches, shuffle=True
            ):
                sampled = dict(zip(self._tensors_names, batch))
                rows = sampled["observations"].shape[0] // num_agents

                inputs = {
                    "observations": self.normalize_observations(
                        sampled["observations"], train=(epoch == 0)
                    )
                }
                value_inputs = (
                    {"observations": self.normalize_states(sampled["states"], train=(epoch == 0))}
                    if self._asymmetric
                    else inputs
                )

                self.policy_optimizer.zero_grad(set_to_none=False)
                self.value_optimizer.zero_grad()
                total_loss = torch.zeros((), device=self.device)

                if any_kept:
                    _, outputs = self.policy.act(
                        {**inputs, "taken_actions": sampled["actions"]}, role="policy"
                    )
                    ratio_log = outputs["log_prob"] - sampled["log_prob"]
                    with torch.no_grad():
                        kl = (ratio_log.exp() - 1.0 - ratio_log).view(num_agents, rows, -1).mean(dim=(1, 2))
                    accumulate("ppo/kl", kl)
                    if self.cfg.kl_threshold > 0:
                        keep = keep & (kl <= self.cfg.kl_threshold)
                        any_kept = bool(keep.any())
                    keep_f = keep.to(ratio_log.dtype)

                    # Masked losses keep the FULL denominator (num_agents), so dropping one
                    # agent does not rescale another agent's gradient. With every agent kept
                    # this is exactly the mean over the flat batch.
                    ratio = ratio_log.exp()
                    surrogate = sampled["advantages"] * ratio
                    surrogate_clipped = sampled["advantages"] * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_terms = -torch.min(surrogate, surrogate_clipped)
                    policy_per_agent = policy_terms.view(num_agents, rows, -1).mean(dim=(1, 2))
                    total_loss = total_loss + (keep_f * policy_per_agent).sum() / num_agents

                    entropy_per_agent = (
                        self.policy.get_entropy(role="policy").view(num_agents, rows, -1).mean(dim=(1, 2))
                    )
                    if self.cfg.entropy_loss_scale:
                        total_loss = total_loss - (
                            self.cfg.entropy_loss_scale * (keep_f * entropy_per_agent).sum() / num_agents
                        )

                    aux = self.compute_aux_loss(
                        LossContext(
                            learner=self,
                            step=timestep,
                            target="policy",
                            sampled=sampled,
                            inputs=inputs,
                            log_prob=outputs["log_prob"],
                            policy_outputs=outputs,
                        )
                    )
                    if aux is not None:
                        total_loss = total_loss + aux

                    if self.on_log:
                        accumulate("loss/policy", policy_per_agent)
                        accumulate("policy/entropy", entropy_per_agent)
                        accumulate("ppo/clip_fraction", (
                            ((ratio - 1.0).abs() > self.cfg.ratio_clip).to(ratio.dtype)
                        ).view(num_agents, rows, -1).mean(dim=(1, 2)))
                # every agent dropped: the policy only takes the (frozen) step below, so its
                # step counter advances like every other minibatch

                value_loss, value_per_agent = self._value_loss(sampled, value_inputs, rows)
                total_loss = total_loss + value_loss
                aux = self.compute_aux_loss(
                    LossContext(
                        learner=self,
                        step=timestep,
                        target="critic",
                        sampled=sampled,
                        critic_inputs=value_inputs,
                        target_values=sampled["returns"],
                    )
                )
                if aux is not None:
                    total_loss = total_loss + aux

                # one combined backward, two disjoint optimizer steps
                total_loss.backward()
                for param in self.policy.parameters():
                    if param.grad is None:  # the first minibatch had no policy loss
                        param.grad = torch.zeros_like(param)
                policy_norms = clip_grad_norm_per_agent(
                    self.policy, num_agents, self.cfg.grad_norm_clip
                )
                value_norms = clip_grad_norm_per_agent(
                    self.value, num_agents, self.cfg.grad_norm_clip
                )
                if self.cfg.kl_threshold > 0:
                    step_with_frozen_agents(self.policy_optimizer, self.policy, num_agents, keep)
                else:
                    self.policy_optimizer.step()
                self.value_optimizer.step()

                # extra value-only passes over the same minibatch
                for _ in range(self.cfg.value_update_ratio - 1):
                    extra_loss, extra_per_agent = self._value_loss(sampled, value_inputs, rows)
                    self.value_optimizer.zero_grad()
                    extra_loss.backward()
                    clip_grad_norm_per_agent(self.value, num_agents, self.cfg.grad_norm_clip)
                    self.value_optimizer.step()
                    value_per_agent = value_per_agent + extra_per_agent
                if self.cfg.value_update_ratio > 1:
                    value_per_agent = value_per_agent / self.cfg.value_update_ratio

                if self.on_log:
                    accumulate("loss/value", value_per_agent)
                    accumulate("grad_norm/policy", policy_norms)
                    accumulate("grad_norm/value", value_norms)
                    accumulate("ppo/kept", keep.to(value_per_agent.dtype))

            if self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if self.value_scheduler is not None:
                self.value_scheduler.step()

        if self.on_log:
            metrics = {name: torch.stack(values).mean(dim=0) for name, values in accumulated.items()}
            metrics["lr/policy"] = torch.full(
                (num_agents,), self.policy_optimizer.param_groups[0]["lr"], device=self.device
            )
            metrics["lr/value"] = torch.full(
                (num_agents,), self.value_optimizer.param_groups[0]["lr"], device=self.device
            )
            self.emit_per_agent(metrics, timestep)

    def _value_loss(self, sampled: dict, value_inputs: dict, rows: int):
        """Clipped MSE value loss. Not KL-masked: an agent's critic keeps learning."""
        predicted, _ = self.value.act(value_inputs, role="value")
        if self.cfg.value_clip > 0:
            predicted = sampled["values"] + torch.clip(
                predicted - sampled["values"], min=-self.cfg.value_clip, max=self.cfg.value_clip
            )
        squared = (sampled["returns"] - predicted) ** 2
        per_agent = squared.view(self.num_agents, rows, -1).mean(dim=(1, 2)) * self.cfg.value_loss_scale
        return self.cfg.value_loss_scale * squared.mean(), per_agent

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_model_keys(self) -> List[str]:
        return ["policy", "value"]

    def _checkpoint_optimizer_keys(self) -> List[str]:
        return ["policy_optimizer", "value_optimizer"]

    def _checkpoint_normalizers(self) -> Dict[str, BlockRunningNorm]:
        normalizers = {}
        if self.observation_normalizer is not None:
            normalizers["observations"] = self.observation_normalizer
        if self.state_normalizer is not None:
            normalizers["states"] = self.state_normalizer
        if self.value_normalizer is not None:
            normalizers["values"] = self.value_normalizer
        return normalizers

    def _checkpoint_extras(self, agent: int) -> Dict[str, Any]:
        return {}

    def _load_extras(self, agent: int, extras: Dict[str, Any], path) -> None:
        if extras:
            raise KeyError(f"{path} holds learner extras {sorted(extras)} but PPO saves none")

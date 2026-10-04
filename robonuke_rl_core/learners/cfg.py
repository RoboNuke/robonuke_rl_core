"""Learner and trainer configuration: the `trainer`, `sac`, `ppo` and `flash_sac` sections.

Hyperparameters come from V's ``SAC_CFG`` / ``PPO_CFG``, minus everything the learner plan
removes (AMP, distributed, auxiliary losses, contact/rotation supervision, per-axis action
metrics, KL-adaptive LR, TensorBoard/wandb writers). OmegaConf-supported types only: a
callable is a ``"module:name"`` string, resolved with ``config.resolve_callable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ["TrainerCfg", "SACCfg", "PPOCfg", "FlashSACCfg"]

LEARNERS = ("sac", "ppo", "flash_sac")


@dataclass
class TrainerCfg:
    """What to run, for how long, and how often to write."""

    #: which learner drives the run; only that learner's section is used
    learner: str = MISSING
    total_timesteps: int = MISSING
    #: env steps between episode-stat flushes through the ``on_log`` hooks
    write_interval: int = 1000
    #: env steps between per-agent checkpoints (0 disables)
    checkpoint_interval: int = 10000
    output_dir: str = "runs"

    def validate(self, cfg: Any) -> None:
        if self.learner not in LEARNERS:
            raise ValueError(f"trainer.learner must be one of {LEARNERS}, got {self.learner!r}")
        if self.total_timesteps < 1:
            raise ValueError(f"trainer.total_timesteps must be >= 1, got {self.total_timesteps}")
        if self.write_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError(
                "trainer.write_interval and trainer.checkpoint_interval must be >= 0, got "
                f"{self.write_interval} and {self.checkpoint_interval}"
            )
        total_envs = cfg.root.task.cfg.scene.num_envs
        num_agents = cfg.experiment.num_agents
        if total_envs % num_agents != 0:
            raise ValueError(
                f"task.cfg.scene.num_envs ({total_envs}) must be divisible by "
                f"experiment.num_agents ({num_agents}): each agent owns a contiguous block of envs"
            )


@dataclass
class _CommonCfg:
    """Fields every learner shares."""

    discount_factor: float = 0.99
    #: env steps of uniform random actions before the policy is used
    random_timesteps: int = 0
    #: env steps before the first update
    learning_starts: int = 0
    weight_decay: float = 0.0
    #: per-agent gradient-norm clip (0 disables); each agent is clipped by its own norm
    grad_norm_clip: float = 0.0
    #: running normalization of observations (and states, when asymmetric), per agent
    normalize_observations: bool = True
    normalizer_clip: float = 5.0
    #: ``"module:name"`` of a ``f(rewards, timestep, timesteps) -> rewards`` shaper
    rewards_shaper: Optional[str] = None


@dataclass
class SACCfg(_CommonCfg):
    """Soft Actor-Critic."""

    gradient_steps: int = 1
    #: transitions sampled **per agent** per gradient step
    batch_size: int = 64
    polyak: float = 0.005
    actor_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3
    entropy_lr: float = 1.0e-3
    #: "constant" or "cosine" (cosine decays actor and critic LR to ``lr_end``)
    lr_schedule: str = "constant"
    lr_end: float = 1.5e-4
    learn_entropy: bool = True
    initial_entropy_value: float = 0.2
    #: target entropy; None means -|A| (see also FlashSAC's ``target_entropy_mode``)
    target_entropy: Optional[float] = None
    #: "log_alpha" (skrl) or "alpha" (FlashSAC's gentler form)
    entropy_loss_form: str = "log_alpha"
    #: SimBa periodic reset: rebuild networks + optimizers, keep the replay buffer
    periodic_reset_enabled: bool = False
    periodic_reset_frequency: int = 0
    periodic_reset_max: int = 0

    def validate(self, cfg: Any) -> None:
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"{self._name()}.lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if self.entropy_loss_form not in ("log_alpha", "alpha"):
            raise ValueError(
                f"{self._name()}.entropy_loss_form must be 'log_alpha' or 'alpha', got "
                f"{self.entropy_loss_form!r}"
            )
        if self.batch_size < 1 or self.gradient_steps < 1:
            raise ValueError(
                f"{self._name()}.batch_size and gradient_steps must be >= 1, got "
                f"{self.batch_size} and {self.gradient_steps}"
            )
        if not 0.0 < self.polyak <= 1.0:
            raise ValueError(f"{self._name()}.polyak must be in (0, 1], got {self.polyak}")
        if self.periodic_reset_enabled and self.periodic_reset_frequency <= 0:
            raise ValueError(
                f"{self._name()}.periodic_reset_enabled needs periodic_reset_frequency > 0, got "
                f"{self.periodic_reset_frequency}"
            )

    def _name(self) -> str:
        return "sac"


@dataclass
class FlashSACCfg(SACCfg):
    """FlashSAC: distributional critic, weight normalization, noise repetition."""

    #: gradient steps between actor updates (the critic updates every step)
    actor_update_period: int = 1
    weight_norm_enabled: bool = True
    noise_repeat_enabled: bool = True
    noise_repeat_zeta_mu: float = 2.0
    noise_repeat_max: int = 16
    #: adaptive reward scaling (Eq. 6); ``g_max`` must equal the critic's ``v_max``
    reward_scaling_enabled: bool = False
    reward_scaling_g_max: float = 5.0
    reward_scaling_eps: float = 1.0e-8
    #: "neg_action_dim" (SAC's -|A|) or "unified" (FlashSAC's sigma-based target)
    target_entropy_mode: str = "neg_action_dim"
    entropy_sigma_target: float = 0.15

    def validate(self, cfg: Any) -> None:
        super().validate(cfg)
        if self.target_entropy_mode not in ("neg_action_dim", "unified"):
            raise ValueError(
                "flash_sac.target_entropy_mode must be 'neg_action_dim' or 'unified', got "
                f"{self.target_entropy_mode!r}"
            )
        if self.actor_update_period < 1:
            raise ValueError(
                f"flash_sac.actor_update_period must be >= 1, got {self.actor_update_period}"
            )
        if self.reward_scaling_enabled and self.reward_scaling_g_max != cfg.model.critic.v_max:
            raise ValueError(
                f"flash_sac.reward_scaling_g_max ({self.reward_scaling_g_max}) must equal "
                f"model.critic.v_max ({cfg.model.critic.v_max}) so normalized returns land "
                "inside the categorical support"
            )

    def _name(self) -> str:
        return "flash_sac"


@dataclass
class PPOCfg(_CommonCfg):
    """Proximal Policy Optimization."""

    #: rollout length per env; the buffer holds this many steps before an update
    rollouts: int = 16
    learning_epochs: int = 8
    mini_batches: int = 2
    gae_lambda: float = 0.95
    policy_lr: float = 3.0e-4
    value_lr: float = 3.0e-4
    lr_schedule: str = "constant"
    lr_end: float = 1.5e-4
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 1.0
    #: per-agent KL early stop: an agent above this stops updating for the rest of the epoch
    #: (0 disables)
    kl_threshold: float = 0.0
    time_limit_bootstrap: bool = False
    #: running normalization of values and returns, per agent
    normalize_values: bool = True
    #: extra value-only updates per minibatch after the combined update (1 = none)
    value_update_ratio: int = 1

    def validate(self, cfg: Any) -> None:
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"ppo.lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if self.rollouts < 1 or self.learning_epochs < 1 or self.mini_batches < 1:
            raise ValueError(
                "ppo.rollouts, learning_epochs and mini_batches must be >= 1, got "
                f"{self.rollouts}, {self.learning_epochs}, {self.mini_batches}"
            )
        if self.value_update_ratio < 1:
            raise ValueError(f"ppo.value_update_ratio must be >= 1, got {self.value_update_ratio}")
        # the memory partitions each AGENT's rows into minibatches, so the per-agent row
        # count is what has to divide evenly (rollouts * total_envs would let a minibatch
        # come out empty when num_agents is large)
        total_envs = cfg.root.task.cfg.scene.num_envs
        envs_per_agent = total_envs // cfg.experiment.num_agents
        rows = self.rollouts * envs_per_agent
        if rows % self.mini_batches != 0:
            raise ValueError(
                f"ppo: rollouts * envs_per_agent ({self.rollouts} * {envs_per_agent} = {rows}) "
                f"must be divisible by mini_batches ({self.mini_batches}) so every minibatch "
                "holds the same number of rows per agent"
            )

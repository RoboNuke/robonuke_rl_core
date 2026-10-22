"""Learner and trainer configuration: the `trainer`, `sac` and `ppo` sections.

Hyperparameters come from V's ``SAC_CFG`` / ``PPO_CFG``, minus everything the learner plan
removes (AMP, distributed, auxiliary losses, contact/rotation supervision, per-axis action
metrics, KL-adaptive LR, TensorBoard/wandb writers). OmegaConf-supported types only: a
callable is a ``"module:name"`` string, resolved with ``config.resolve_callable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

__all__ = ["TrainerCfg", "SACCfg", "PPOCfg", "LEARNERS", "register_learner", "known_learners"]

#: the package's own learners: ``models/factory`` builds their networks and eval loads them
LEARNERS = ("sac", "ppo")

#: learners a project registers with :func:`register_learner`. They bring their own entry
#: script and training loop; the package only has to accept the name in ``trainer.learner``.
PROJECT_LEARNERS: list = []


def register_learner(name: str) -> None:
    """Accept ``name`` as a ``trainer.learner`` value. Call it in the entry script's ``setup``."""
    if name in LEARNERS or name in PROJECT_LEARNERS:
        raise ValueError(f"learner '{name}' is already registered; known: {known_learners()}")
    PROJECT_LEARNERS.append(name)


def known_learners() -> tuple:
    """Every value ``trainer.learner`` accepts: the package's plus the registered ones."""
    return LEARNERS + tuple(PROJECT_LEARNERS)


@dataclass
class TrainerCfg:
    """What to run, for how long, and how often to write."""

    #: which learner drives the run; only that learner's section is used
    learner: str = MISSING
    #: env steps for the package's learners; a registered project learner defines its own
    #: unit (its README says which)
    total_timesteps: int = MISSING
    #: env steps between episode-stat flushes through the ``on_log`` hooks
    write_interval: int = 1000
    #: 0: ``episode/success`` is the mean over the episodes that finished in the write
    #: interval. N > 0: the mean over each agent's last N finished episodes, republished every
    #: interval (a rolling window, not cleared at the flush), so a short interval still has a rate
    success_window: int = 0
    #: env steps between per-agent checkpoints (0 disables)
    checkpoint_interval: int = 10000
    output_dir: str = "runs"
    #: publish the learner's memory stats (``stats/ram_mb``, ``stats/gpu_used_mb``,
    #: ``stats/gpu_torch_reserved_mb``); ``stats/update_time_ms`` is published either way
    log_memory_stats: bool = True

    def validate(self, cfg: Any) -> None:
        if self.learner not in known_learners():
            raise ValueError(
                f"trainer.learner must be one of {known_learners()}, got {self.learner!r}"
            )
        if self.total_timesteps < 1:
            raise ValueError(f"trainer.total_timesteps must be >= 1, got {self.total_timesteps}")
        if self.write_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError(
                "trainer.write_interval and trainer.checkpoint_interval must be >= 0, got "
                f"{self.write_interval} and {self.checkpoint_interval}"
            )
        if self.success_window < 0:
            raise ValueError(f"trainer.success_window must be >= 0, got {self.success_window}")
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
    #: target entropy; None means -|A|
    target_entropy: Optional[float] = None
    #: "log_alpha" (skrl's form) or "alpha" (the gentler multiplicative form)
    entropy_loss_form: str = "log_alpha"
    #: the entropy bonus in the critics' bootstrap target (``- alpha * log_pi(a'|s')``):
    #: classic SAC; false leaves the target pure reward (SERL's ``backup_entropy=False``), the
    #: entropy then shaping only the actor
    backup_entropy: bool = True
    #: where the temperature's entropy estimate comes from: "current" (the actor step's batch
    #: and its freshly sampled actions) or "next" (the critic step's batch at its next
    #: observations, the policy's next actions it already sampled for the target, demo rows
    #: included inside the demo window: SERL's temperature loss)
    entropy_observations: str = "current"
    #: SimBa periodic reset: rebuild networks + optimizers, keep the replay buffer
    periodic_reset_enabled: bool = False
    periodic_reset_frequency: int = 0
    periodic_reset_max: int = 0
    #: Q critics in the ensemble (``critic_1 .. critic_N``, each with a target); 2 is the
    #: classic twin, 10 is REDQ / RLPD
    num_critics: int = 2
    #: critics the bootstrap target takes its min over, drawn at random per agent each
    #: gradient step (REDQ's in-target minimization); equal to num_critics = min over all
    critic_subset: int = 2
    #: how the actor's loss reduces the critics' Q: "min" (over all of them; classic SAC) or
    #: "mean" (REDQ / RLPD, for a large ensemble)
    actor_q_reduction: str = "min"
    #: the actor (and the temperature) update once every this many gradient steps; with
    #: gradient_steps G and policy_delay G the critic gets G updates per env step, the actor 1
    policy_delay: int = 1
    #: env step of the first actor update; None means learning_starts. Before it only the
    #: critics train (learning_starts stays the first critic update)
    policy_learning_starts: Optional[int] = None
    #: a ``torch.save``d dict of demo transitions in the agent's own spaces (observations,
    #: actions, rewards, next_observations, terminated, truncated; states / next_states when
    #: asymmetric), mixed into the CRITIC's batches only, during the window below
    demo_transitions: Optional[str] = None
    #: env steps [demo_sampling_start, demo_sampling_end) in which each critic batch is half
    #: demo, half online ("symmetric sampling"); while the replay memory holds fewer than half
    #: a batch, demos fill the rest. The actor always trains on online data only
    demo_sampling_start: int = 0
    demo_sampling_end: int = 0
    #: capture each kind of gradient step (critic only / with the actor, with or without demo
    #: rows) as a CUDA graph and replay it as one launch: the update is launch-bound at small
    #: batch sizes. CUDA only (ignored on CPU); updates with aux losses, and mixed batches
    #: while the memory holds under half a batch, stay eager
    cuda_graph: bool = False

    ACTOR_Q_REDUCTIONS = ("min", "mean")
    ENTROPY_OBSERVATIONS = ("current", "next")

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
        if self.entropy_observations not in self.ENTROPY_OBSERVATIONS:
            raise ValueError(
                f"{self._name()}.entropy_observations must be one of {self.ENTROPY_OBSERVATIONS}, "
                f"got {self.entropy_observations!r}"
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
        if self.num_critics < 2:
            raise ValueError(f"{self._name()}.num_critics must be >= 2, got {self.num_critics}")
        if not 1 <= self.critic_subset <= self.num_critics:
            raise ValueError(
                f"{self._name()}.critic_subset must be in [1, num_critics={self.num_critics}], "
                f"got {self.critic_subset}"
            )
        if self.actor_q_reduction not in self.ACTOR_Q_REDUCTIONS:
            raise ValueError(
                f"{self._name()}.actor_q_reduction must be one of {self.ACTOR_Q_REDUCTIONS}, got "
                f"{self.actor_q_reduction!r}"
            )
        if self.policy_delay < 1:
            raise ValueError(f"{self._name()}.policy_delay must be >= 1, got {self.policy_delay}")
        if self.policy_delay > self.gradient_steps:
            raise ValueError(
                f"{self._name()}.policy_delay ({self.policy_delay}) above gradient_steps "
                f"({self.gradient_steps}) would skip whole env steps of actor updates; keep it "
                "<= gradient_steps"
            )
        if (
            self.policy_learning_starts is not None
            and self.policy_learning_starts < self.learning_starts
        ):
            raise ValueError(
                f"{self._name()}.policy_learning_starts ({self.policy_learning_starts}) is before "
                f"learning_starts ({self.learning_starts}), the first update of anything"
            )
        window = self.demo_sampling_end > self.demo_sampling_start
        if self.demo_sampling_start < 0 or self.demo_sampling_end < self.demo_sampling_start:
            raise ValueError(
                f"{self._name()}.demo_sampling_start / demo_sampling_end must satisfy "
                f"0 <= start <= end, got {self.demo_sampling_start} / {self.demo_sampling_end}"
            )
        if window and not self.demo_transitions:
            raise ValueError(
                f"{self._name()}.demo_sampling_start..end ({self.demo_sampling_start}.."
                f"{self.demo_sampling_end}) mixes demos into the critic's batches, but "
                f"{self._name()}.demo_transitions names no file"
            )
        if self.demo_transitions and not window:
            raise ValueError(
                f"{self._name()}.demo_transitions is set but the sampling window "
                f"[{self.demo_sampling_start}, {self.demo_sampling_end}) is empty, so the demos "
                "would never be used"
            )
        if window and self.batch_size < 2:
            raise ValueError(
                f"{self._name()}.batch_size must be >= 2 to split a batch between demo and "
                f"online rows, got {self.batch_size}"
            )

    def _name(self) -> str:
        return "sac"


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

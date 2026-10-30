"""Learner and trainer configuration: the `trainer`, `sac` and `ppo` sections.

Hyperparameters come from V's ``SAC_CFG`` / ``PPO_CFG``, minus everything the learner plan
removes (AMP, distributed, auxiliary losses, contact/rotation supervision, per-axis action
metrics, KL-adaptive LR, TensorBoard/wandb writers). OmegaConf-supported types only: a
callable is a ``"module:name"`` string, resolved with ``config.resolve_callable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

__all__ = ["TrainerCfg", "SACCfg", "PPOCfg", "LEARNERS", "register_learner", "known_learners", "sac_critic_outputs"]

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
    #: more ``infos["episode_metrics_to_log"]`` channels averaged over the same last ``success_window`` episodes
    #: (e.g. per-phase pass rates): exact names or ``fnmatch`` patterns (``phase/passed_*``), each republished at
    #: every flush instead of per episode. An episode whose value is NaN (e.g. a phase it never reached) is not
    #: counted: the window holds the last ``success_window`` episodes with a value. Needs ``success_window`` > 0
    windowed_metrics: List[str] = field(default_factory=list)
    #: what picks ``ckpt_best`` and ``episode/best_success_rate``: "return" (the highest mean episode return of an
    #: interval; ``best_success_rate`` is that interval's success rate, so it can go down when return and success
    #: diverge, e.g. under shaped rewards) or "success" (the highest ``episode/success`` of an interval, the windowed
    #: one with ``success_window``; ``best_success_rate`` is then its running maximum)
    best_by: str = "return"
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
        if self.windowed_metrics and self.success_window < 1:
            raise ValueError("trainer.windowed_metrics averages over the last success_window episodes: set success_window > 0")
        if self.best_by not in ("return", "success"):
            raise ValueError(f"trainer.best_by must be 'return' or 'success', got {self.best_by!r}")
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
    #: run the first this many episodes (counted over every env) on zero actions -- for a residual, the base alone,
    #: no exploration noise -- storing them as ordinary transitions, with no updates until they have finished, so
    #: the memory starts with the base's own outcomes (its successes among them). 0 is off
    warmup_episodes: int = 0
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
    #: scale the stored reward by its per-agent running std (ResiP's ``RunningMeanStdClip``:
    #: ``r / sqrt(var + 1e-8)``, no mean subtraction, the variance updated from every env
    #: step's batch of rewards), after the shaper; episode returns and ``reward/step`` keep
    #: the raw reward
    normalize_rewards: bool = False
    #: the scaled reward is clipped to +-this
    reward_clip: float = 5.0

    def _validate_common(self, name: str) -> None:
        if self.reward_clip <= 0.0:
            raise ValueError(f"{name}.reward_clip must be > 0, got {self.reward_clip}")


@dataclass
class PotentialTermCfg:
    """One channel of ``sac.potential_terms``: ``weight`` times the ``channel`` the env publishes in its infos, in
    the potential of the residual of phase ``route`` (``"*"``: an unrouted SAC)."""

    route: str = MISSING
    channel: str = MISSING
    weight: float = MISSING


@dataclass
class SACCfg(_CommonCfg):
    """Soft Actor-Critic."""

    #: gradient steps per update round
    gradient_steps: int = 1
    #: env steps between update rounds, once ``learning_starts`` is passed: a round of
    #: ``gradient_steps`` runs at every env step ``t >= learning_starts`` with
    #: ``(t - learning_starts) % update_every == 0``, so the update-to-data ratio is
    #: ``gradient_steps / (update_every * num_envs)`` (Policy Decorator: 16 steps every 4)
    update_every: int = 1
    #: transitions sampled **per agent** per gradient step
    batch_size: int = 64
    polyak: float = 0.005
    actor_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3
    entropy_lr: float = 1.0e-3
    #: "constant" or "cosine" (cosine decays actor and critic LR to ``lr_end``)
    lr_schedule: str = "constant"
    lr_end: float = 1.5e-4
    #: linear LR warm-up, in gradient steps, before the schedule proper (``optim.lr_at``);
    #: separately for the actor and the critics. 0 is none
    actor_lr_warmup: int = 0
    critic_lr_warmup: int = 0
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
    #: draw the critic_subset target critics with replacement (SERL's ``jax.random.randint``: a critic may be drawn
    #: twice, so the min is sometimes over fewer distinct critics). False: distinct critics (REDQ)
    critic_subset_replacement: bool = False
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
    #: inside the demo window, the actor trains on the critic's mixed (half demo) batch too -- one batch for both,
    #: as RLPD does -- instead of its own online-only batch
    demo_batch_shared: bool = False
    #: the demo rows of a mixed batch count as successes in PARTS' clone term (TD3+BC's anchor to the dataset's
    #: actions: the demo action, clipped to the actor's [-1, 1], is the clone target) and never as failures. False:
    #: demo rows are in neither term
    clone_demo_rows: bool = False
    #: an L2 penalty on the actor's mean action, ``action_l2_weight * mean_d(a_mean_d^2)`` over each batch's rows
    #: (the demo rows of a memory_critic memory excluded): pulls the actor toward zero -- for a residual, the base
    #: alone -- wherever the critic's gain for a larger action does not outweigh it. 0 is off
    action_l2_weight: float = 0.0
    #: where the demos go. "sample": a separate set mixed into the critic's batches in the window above.
    #: "memory": written into the replay memory once, before the first env step, as ordinary transitions
    #: (overwritten when the memory wraps, like any other row; no window, no symmetric sampling); with
    #: success_clone_weight / failure_anchor_weight each demo episode is an episode of its own, a success when it
    #: holds a positive reward, so the actor's PARTS terms read them as any other row.
    #: "memory_critic": written into the replay memory as with "memory", each row flagged as a demo: the critics
    #: sample every row, the actor samples its own batch from the online rows only, so it never sees a demo and
    #: always gets a full batch
    demo_storage: str = "sample"
    #: warmup_episodes: let the critics train on the warm-up's transitions (from learning_starts) while the actor
    #: stays frozen until the warm-up has ended. False: nothing trains during the warm-up
    warmup_trains_critics: bool = False
    #: no update at all (critics or actor) until every agent's memory holds this many of its own online rows (demo
    #: rows not counted): a routed residual starts training only once its phase has given it this much data, however
    #: long that takes. 0 is off (learning_starts alone)
    learning_starts_rows: int = 0
    #: capture each kind of gradient step (critic only / with the actor, with or without demo
    #: rows) as a CUDA graph and replay it as one launch: the update is launch-bound at small
    #: batch sizes. CUDA only (ignored on CPU); updates with aux losses, and mixed batches
    #: while the memory holds under half a batch, stay eager
    cuda_graph: bool = False
    #: Policy Decorator's progressive exploration over this many env steps (its H): while
    #: training, each env runs this step on the policy's actions with probability
    #: ``min(timestep / H, 1)`` (one draw per env, every env step) and on zeros otherwise (for
    #: a residual policy: the base alone); the zeros are what is executed and stored.
    #: ``random_timesteps`` actions too. 0 is off
    progressive_exploration_steps: int = 0
    #: env step the progressive exploration ramp starts at: before it every env runs on zeros (share 0), from
    #: it the share is ``min((timestep - start) / progressive_exploration_steps, 1)``. 0 starts the ramp at once
    progressive_exploration_start: int = 0
    #: n-step TD targets: each critic target sums the next ``n_step`` rewards (fewer where the
    #: episode ends first) and bootstraps ``gamma ** m * Q`` that many steps later
    #: (``learners/n_step.py``); online rows and demo rows alike, uncorrected. 1 is classic
    #: 1-step SAC
    n_step: int = 1
    #: FlashSAC (arXiv:2604.04539; github.com/Holiday-Robot/FlashSAC), all off by default.
    #: "scalar" (a Q value, MSE to the target) or "categorical": each critic predicts a
    #: distribution over ``critic_num_bins`` atoms spread evenly on [critic_min_value,
    #: critic_max_value], trained by cross-entropy to the target distribution projected onto
    #: the atoms (C51). Its Q is the distribution's mean; the bootstrap takes the distribution
    #: of the critic with the lowest Q, row by row
    critic_distribution: str = "scalar"
    critic_num_bins: int = 101
    critic_min_value: float = -5.0
    critic_max_value: float = 5.0
    #: FlashSAC's adaptive reward scaling: the critics' rewards (online and demo, at sampling
    #: time) are divided by ``max(sqrt(var(G) + 1e-8), max|G| / adaptive_reward_g_max)``, with G
    #: each env's running discounted return of the stored reward, its variance and max
    #: magnitude kept per agent from every stored env step. The memory keeps the raw reward
    adaptive_reward_scaling: bool = False
    adaptive_reward_g_max: float = 5.0
    #: FlashSAC's weight normalization: after every optimizer step (and once at build) each
    #: Linear weight row of the policy and the critics is projected onto the unit sphere, and
    #: each normalization layer's (scale, shift) to norm sqrt(width). Linear biases are left as
    #: they are; the targets are only normalized at build (their Polyak average is not)
    weight_normalization: bool = False
    #: FlashSAC's unified entropy target: ``0.5 |A| log(2 pi e sigma^2)``, the entropy of a
    #: Gaussian of std sigma on every action dim (FlashSAC: 0.15). None leaves
    #: ``target_entropy``; setting both is an error
    target_entropy_sigma: Optional[float] = None
    #: FlashSAC's cross-batch value prediction: the critics (and the target critics) take the
    #: current and next rows of a batch in ONE call, ``[s, a] ++ [s', a']`` per agent, and so
    #: does the actor (``s ++ s'``); each loss reads its own half. Only a layer that uses batch
    #: statistics (BatchNorm) sees a difference; for SimBa and the MLP it changes nothing but
    #: the cost
    cross_batch_value: bool = False
    #: FlashSAC's noise repetition: while training, each agent's envs hold one Gaussian noise
    #: draw for k env steps, k drawn from a Zeta(noise_repeat_zeta_s) truncated at
    #: noise_repeat_max (P(k) ~ k^-s), then draw both afresh; the action is the policy's
    #: ``tanh(mean + std * noise)``. The uniform ``random_timesteps`` actions are not held
    noise_repeat: bool = False
    noise_repeat_zeta_s: float = 2.0
    noise_repeat_max: int = 16
    #: train an unsquashed actor (``model.architecture: mlp`` with ``model.actor.squash: false``): a plain
    #: Gaussian, its samples unbounded (the env gets them as they are) and its log-probability the Gaussian's
    #: own, no tanh correction. False refuses that actor for SAC, whose actor is otherwise tanh-squashed
    unsquashed_actor: bool = False
    #: log the acting policy's std per env group: the ``infos`` channel holding each env's integer label every step,
    #: ``"<key>/<name>"`` for ``infos[key][name]`` (e.g. ``eval_state/phase/id``, one_leg's task phase), or a plain
    #: top-level key. Each env's std at an observation (the mean over its action dims) is grouped by the label the
    #: env published for that observation and logged as ``policy/std_<std_label_names[label]>``; labels outside
    #: ``std_label_names`` (an episode's end codes) are skipped. None: off
    std_by_label: Optional[str] = None
    std_label_names: List[str] = field(default_factory=list)
    #: per-env ``infos["metrics_to_log"]`` channels to log per ``std_by_label`` group too, as ``<name>_<group>``: each
    #: env's value for the step goes to the group of the label of the observation it acted on, and only the envs that
    #: ran the policy that step count (progressive exploration's zeros left out; every env when it is off)
    metrics_by_label: List[str] = field(default_factory=list)
    #: PARTS' actor terms (Su et al., arXiv:2609.21788, eq. 4), off at 0. Every stored transition carries its
    #: episode's id; when an episode ends its outcome (``outcome_metric`` from ``infos["episode_metrics_to_log"]``,
    #: 1 success / 0 failure) is recorded, and the actor's loss gains, per agent over the batch's rows of finished
    #: episodes: ``success_clone_weight * mean_{B+} ||a_mean - a_stored||^2`` (clone the residuals executed in
    #: successful episodes) + ``failure_anchor_weight * mean_{B-} ||a_mean||^2`` (pull the actor toward zero, the base
    #: alone, on states of failed episodes), ``a_mean`` the actor's deterministic (mean) action. Rows of episodes still
    #: running count in neither
    success_clone_weight: float = 0.0
    #: the demos in the actor too (with ``demo_transitions``): each actor step also clones ``batch_size`` demo rows,
    #: per agent, as successes: ``success_clone_weight * mean ||a_mean(s_demo) - clip(a_demo, -1, 1)||^2`` (logged
    #: ``actor/demo_clone_loss``). The critics take the demos through the demo sampling window as before
    demo_actor_clone: bool = False
    #: how PARTS' clone and anchor terms reduce over the action dims: "sum" (||.||^2, eq. 4 as written; grows with
    #: the action width, e.g. 4x for a 4-step chunk) or "mean" (TD3+BC's per-dim mean squared error, so lambda_Q =
    #: actor_q_scale / mean|Q| and lambda+ / lambda- keep TD3+BC's balance whatever the width)
    parts_term_reduction: str = "sum"
    failure_anchor_weight: float = 0.0
    outcome_metric: str = "episode/success"
    #: PARTS' lambda_Q in TD3+BC's form (Fujimoto & Gu 2021): the actor's Q term is scaled by
    #: ``actor_q_scale / mean|Q|`` (each agent's batch mean, no gradient), so the clone / anchor terms keep their
    #: weight whatever the value scale (TD3+BC: 2.5). None leaves the Q term as it is
    actor_q_scale: Optional[float] = None
    #: an ``infos`` key holding, per env and step, 1 where the env executed the agent's action and 0 where it ignored
    #: it (a residual switched off for that step): the stored action is multiplied by it, so the memory and the
    #: critics see the action that was executed. None: every stored action is the agent's
    executed_action_mask: Optional[str] = None
    #: ``std_label_names`` groups whose transitions the memory stores: a transition is added only when the label of
    #: the observation it starts from (``std_by_label``) is one of these (e.g. one_leg's ``insert`` and ``screw``, a
    #: residual acting only after the grasp), so the memory holds only what is trained on; the agents then fill at
    #: their own pace (``MultiRandomMemory.add_rows``). Empty: every transition is stored
    store_labels: List[str] = field(default_factory=list)
    #: PARTS' per-phase residuals: one complete SAC (actor, critics, targets, optimizers, replay memory) per listed
    #: ``std_label_names`` label, all with this section's settings (``learners/routed.py``). Each env's action comes
    #: from the residual of the label its observation is in (zeros, the base alone, in an unlisted one); each residual
    #: stores only the transitions starting in its label, rewarded by its local outcome: 1 when the next observation's
    #: label is a LATER entry of ``std_label_names`` (the phase passed, or e.g. ``success``), its attempt then ending;
    #: an episode ending first ends it with 0. Empty: one SAC for every step
    route_labels: List[str] = field(default_factory=list)
    #: set per residual by the routing (one of ``route_labels``); a SAC with it stores, rewards and ends its attempts
    #: as described there. Not set by hand
    route_label: Optional[str] = None
    #: potential-based reward shaping (Ng et al. 1999), per routed residual: a list of ``{route, channel, weight}``
    #: (:class:`PotentialTermCfg`; ``route`` a route label, ``"*"`` for an unrouted SAC). Every stored transition's reward gains ``gamma Phi(s') - Phi(s)``
    #: (``discount_factor``), ``Phi = -sum weight * value`` over the channels the env published with each observation
    #: (``"<key>/<name>"`` reads ``infos[key][name]``, e.g. one_leg's ``eval_state/phase/tip_mm``): a dense reward for
    #: closing the distance. ``Phi(s')`` is the next observation's on an attempt's last transition too (not 0), so an
    #: attempt that ends far off pays for it. No shaping on the step after an env's episode ended (the values
    #: published with its first observation are the ended episode's). A residual with no entry is unshaped. Empty: off
    potential_terms: List[PotentialTermCfg] = field(default_factory=list)
    #: bootstrap through a time limit (``truncated`` without ``terminated``: ``r + gamma Q(s', a')``), as SB3 does with
    #: the true final observation. False ends every episode's last transition, ``r`` alone: Isaac Lab resets an env
    #: inside the step that ends it and returns the next episode's first observation, so ``s'`` there is not the
    #: state the episode reached
    bootstrap_time_limits: bool = True
    #: PARTS' success-reweighted retraining, every ``retrain_every`` env steps (0: off): the replay memory keeps every
    #: transition of a successful episode (``outcome_metric``) and those of a random ``retrain_failure_fraction`` of
    #: the failed ones (episodes still running are kept), the networks, optimizers and temperature start fresh, and
    #: ``retrain_gradient_steps`` updates run on that curated memory before online training resumes on it
    retrain_every: int = 0
    #: count ``learning_starts``, ``update_every`` and ``retrain_every`` in this residual's own steps (the env steps
    #: whose starting label is its ``route_label``), and update only on those steps: a routed residual never updates
    #: while its phase is not running (nor in phases with no residual). Needs ``route_labels``
    route_clock: bool = False
    #: hold transitions while the task makes no progress: the ``infos`` channel (``"<key>/<name>"`` for
    #: ``infos[key][name]``) holding, per env, 1 on a step that made no progress (e.g. one_leg's
    #: ``eval_state/phase/still``). Such transitions are held back, written once progress resumes; if the episode is
    #: ended for lack of progress (``hold_stall_key`` 1 on its last step), only the first held transition is written,
    #: as a terminal failure, and the rest are dropped, so the memory never fills with the repeated stalled states.
    #: Set both or neither
    hold_still_key: Optional[str] = None
    hold_stall_key: Optional[str] = None
    #: PARTS' reference dropout: in each actor update, zero the last ``reference_columns`` columns of the actor's
    #: observation (the base action, for a residual that observes it last) for this fraction of the batch rows, so
    #: the actor cannot learn its correction from the reference alone. 0 is off
    reference_dropout: float = 0.0
    reference_columns: int = 0
    retrain_failure_fraction: float = 0.1
    retrain_gradient_steps: int = 0

    ACTOR_Q_REDUCTIONS = ("min", "mean")
    ENTROPY_OBSERVATIONS = ("current", "next")
    CRITIC_DISTRIBUTIONS = ("scalar", "categorical")

    def validate(self, cfg: Any) -> None:
        self._validate_common(self._name())
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"{self._name()}.lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if self.actor_lr_warmup < 0 or self.critic_lr_warmup < 0:
            raise ValueError(
                f"{self._name()}.actor_lr_warmup and critic_lr_warmup must be >= 0 gradient steps, "
                f"got {self.actor_lr_warmup} and {self.critic_lr_warmup}"
            )
        if self.update_every < 1:
            raise ValueError(f"{self._name()}.update_every must be >= 1 env step, got {self.update_every}")
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
        if self.demo_batch_shared and not (self.demo_transitions and window and self.demo_storage == "sample"):
            raise ValueError(f"{self._name()}.demo_batch_shared shares the demo window's mixed batch: set demo_transitions, "
                             "a demo_sampling_start..end window and demo_storage: sample")
        if self.action_l2_weight < 0:
            raise ValueError(f"{self._name()}.action_l2_weight must be >= 0, got {self.action_l2_weight}")
        if self.clone_demo_rows and not (self.demo_transitions and window and self.success_clone_weight > 0):
            raise ValueError(f"{self._name()}.clone_demo_rows clones the mixed batch's demo rows: set demo_transitions, a "
                             "demo_sampling_start..end window and success_clone_weight > 0")
        if self.demo_storage not in ("sample", "memory", "memory_critic"):
            raise ValueError(f"{self._name()}.demo_storage must be 'sample', 'memory' or 'memory_critic', got {self.demo_storage!r}")
        if self.demo_storage in ("memory", "memory_critic"):
            if not self.demo_transitions:
                raise ValueError(f"{self._name()}.demo_storage: memory writes sac.demo_transitions into the memory; set it")
            if window:
                raise ValueError(
                    f"{self._name()}.demo_storage: memory puts the demos in the replay memory, not in a sampling "
                    f"window: leave demo_sampling_start..end empty (got {self.demo_sampling_start}..{self.demo_sampling_end})"
                )
        if self.demo_transitions and not window and self.demo_storage == "sample":
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
        if self.n_step < 1:
            raise ValueError(f"{self._name()}.n_step must be >= 1 (1 is 1-step TD), got {self.n_step}")
        if self.progressive_exploration_start < 0:
            raise ValueError(
                f"{self._name()}.progressive_exploration_start must be >= 0, got {self.progressive_exploration_start}"
            )
        if self.progressive_exploration_start and self.progressive_exploration_steps <= 0:
            raise ValueError(
                f"{self._name()}.progressive_exploration_start ({self.progressive_exploration_start}) delays a ramp "
                "that is off: set progressive_exploration_steps > 0"
            )
        if self.progressive_exploration_steps < 0:
            raise ValueError(
                f"{self._name()}.progressive_exploration_steps must be >= 0 (0 is off), got "
                f"{self.progressive_exploration_steps}"
            )
        name = self._name()
        if self.critic_distribution not in self.CRITIC_DISTRIBUTIONS:
            raise ValueError(
                f"{name}.critic_distribution must be one of {self.CRITIC_DISTRIBUTIONS}, got "
                f"{self.critic_distribution!r}"
            )
        if self.critic_distribution == "categorical":
            if self.critic_num_bins < 2:
                raise ValueError(f"{name}.critic_num_bins must be >= 2, got {self.critic_num_bins}")
            if not self.critic_min_value < self.critic_max_value:
                raise ValueError(
                    f"{name}.critic_min_value ({self.critic_min_value}) must be below "
                    f"critic_max_value ({self.critic_max_value})"
                )
            if self.critic_subset != self.num_critics:
                raise ValueError(
                    f"{name}.critic_distribution: categorical bootstraps from the lowest-Q critic of "
                    f"all of them; critic_subset ({self.critic_subset}) must equal num_critics "
                    f"({self.num_critics})"
                )
        if self.adaptive_reward_scaling:
            if self.adaptive_reward_g_max <= 0.0:
                raise ValueError(
                    f"{name}.adaptive_reward_g_max must be > 0, got {self.adaptive_reward_g_max}"
                )
            if self.normalize_rewards:
                raise ValueError(
                    f"{name}.adaptive_reward_scaling and normalize_rewards both scale the reward; "
                    "use one"
                )
        if self.target_entropy_sigma is not None:
            if self.target_entropy is not None:
                raise ValueError(
                    f"{name}.target_entropy ({self.target_entropy}) and target_entropy_sigma "
                    f"({self.target_entropy_sigma}) both set the target entropy; set one"
                )
            if self.target_entropy_sigma <= 0.0:
                raise ValueError(
                    f"{name}.target_entropy_sigma must be > 0, got {self.target_entropy_sigma}"
                )
        for route in list(self.route_labels) + ([self.route_label] if self.route_label else []):
            if self.std_by_label is None or route not in self.std_label_names:
                raise ValueError(
                    f"{name}.route_labels / route_label {route!r} must be one of std_label_names "
                    f"{list(self.std_label_names)} (with std_by_label set)"
                )
        if self.store_labels:
            if self.std_by_label is None:
                raise ValueError(f"{name}.store_labels filters by std_by_label's labels; set std_by_label")
            unknown = [label for label in self.store_labels if label not in self.std_label_names]
            if unknown:
                raise ValueError(f"{name}.store_labels {unknown} are not in std_label_names {list(self.std_label_names)}")
        if self.retrain_every < 0 or self.retrain_gradient_steps < 0:
            raise ValueError(f"{name}.retrain_every and retrain_gradient_steps must be >= 0")
        if not 0.0 <= self.retrain_failure_fraction <= 1.0:
            raise ValueError(f"{name}.retrain_failure_fraction must be in [0, 1], got {self.retrain_failure_fraction}")
        routed = list(self.route_labels) + ([self.route_label] if self.route_label else [])
        seen = set()
        for term in self.potential_terms:
            key = term.route
            if key == "*" and routed:
                raise ValueError(f"{name}.potential_terms route '*' is an unrouted SAC's; a routed one names its route labels")
            if key != "*" and key not in self.std_label_names:
                raise ValueError(f"{name}.potential_terms route {key!r} must be '*' or one of std_label_names {list(self.std_label_names)}")
            if key != "*" and not routed:
                raise ValueError(f"{name}.potential_terms route {key!r} shapes a routed residual: set route_labels (or use '*')")
            if not term.channel:
                raise ValueError(f"{name}.potential_terms: an entry for {key!r} names no channel")
            if (key, term.channel) in seen:
                raise ValueError(f"{name}.potential_terms lists {term.channel!r} twice for {key!r}")
            seen.add((key, term.channel))
        if self.route_clock and not (self.route_labels or self.route_label):
            raise ValueError(f"{name}.route_clock counts a routed residual's own steps: set route_labels")
        if bool(self.hold_still_key) != bool(self.hold_stall_key):
            raise ValueError(f"{name}.hold_still_key and hold_stall_key go together: set both or neither")
        if not 0.0 <= self.reference_dropout <= 1.0:
            raise ValueError(f"{name}.reference_dropout must be in [0, 1], got {self.reference_dropout}")
        if self.reference_dropout > 0 and self.reference_columns < 1:
            raise ValueError(f"{name}.reference_dropout zeroes the actor observation's last reference_columns columns: set it >= 1")
        if self.retrain_every and self.retrain_gradient_steps < 1:
            raise ValueError(f"{name}.retrain_every retrains fresh networks: set retrain_gradient_steps >= 1")
        if self.success_clone_weight < 0 or self.failure_anchor_weight < 0:
            raise ValueError(
                f"{name}.success_clone_weight and failure_anchor_weight must be >= 0, got "
                f"{self.success_clone_weight}, {self.failure_anchor_weight}"
            )
        if self.actor_q_scale is not None and self.actor_q_scale <= 0:
            raise ValueError(f"{name}.actor_q_scale must be > 0, got {self.actor_q_scale}")
        if self.learning_starts_rows < 0:
            raise ValueError(f"{name}.learning_starts_rows must be >= 0, got {self.learning_starts_rows}")
        if self.warmup_episodes < 0:
            raise ValueError(f"{name}.warmup_episodes must be >= 0, got {self.warmup_episodes}")
        if self.parts_term_reduction not in ("sum", "mean"):
            raise ValueError(f"{name}.parts_term_reduction must be 'sum' or 'mean', got {self.parts_term_reduction!r}")
        if self.demo_actor_clone and self.demo_storage in ("memory", "memory_critic"):
            raise ValueError(f"{name}.demo_actor_clone samples the separate demo set; with demo_storage: memory the demos are memory rows the clone term already reads")
        if self.demo_actor_clone and not (self.demo_transitions and self.success_clone_weight > 0):
            raise ValueError(f"{name}.demo_actor_clone clones the demos with success_clone_weight: set demo_transitions and success_clone_weight > 0")
        if self.metrics_by_label and self.std_by_label is None:
            raise ValueError(f"{name}.metrics_by_label groups by std_by_label's labels; set std_by_label")
        if self.std_by_label is not None and not self.std_label_names:
            raise ValueError(f"{name}.std_by_label ({self.std_by_label!r}) needs std_label_names, one per label value")
        if self.noise_repeat:
            if self.noise_repeat_max < 1:
                raise ValueError(f"{name}.noise_repeat_max must be >= 1, got {self.noise_repeat_max}")
            if self.noise_repeat_zeta_s <= 0.0:
                raise ValueError(
                    f"{name}.noise_repeat_zeta_s must be > 0, got {self.noise_repeat_zeta_s}"
                )

    def _name(self) -> str:
        return "sac"


def sac_critic_outputs(sac_cfg: Any) -> int:
    """Columns each SAC critic emits: 1 (a Q value) or ``critic_num_bins`` logits
    (``critic_distribution: categorical``); what the model factory builds the critics with."""
    if getattr(sac_cfg, "critic_distribution", "scalar") == "categorical":
        return int(sac_cfg.critic_num_bins)
    return 1


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
    #: linear LR warm-up, in updates (rollouts), before the schedule proper (``optim.lr_at``);
    #: separately for the policy and the value net (ResiP: 5 for the actor, 0 for the critic).
    #: 0 is none
    policy_lr_warmup: int = 0
    value_lr_warmup: int = 0
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
    #: reset EVERY env right after each update, so each rollout starts from fresh initial
    #: conditions (ResiP's ``reset_every_iteration``). Needs the package's trainer
    #: (``learners/trainer.py``, what ``train.py`` runs): it performs the reset and hands the
    #: fresh observations to the next rollout. The episodes cut by the reset are not counted
    #: in the episode statistics
    reset_every_iteration: bool = False

    def validate(self, cfg: Any) -> None:
        self._validate_common("ppo")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"ppo.lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if self.policy_lr_warmup < 0 or self.value_lr_warmup < 0:
            raise ValueError(
                "ppo.policy_lr_warmup and value_lr_warmup must be >= 0 updates, got "
                f"{self.policy_lr_warmup} and {self.value_lr_warmup}"
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

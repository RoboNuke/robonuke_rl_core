"""Deep Deterministic Policy Gradient (``trainer.learner: ddpg``; Lillicrap et al. 2016), with the TD3 / REDQ / ResFiT
pieces as options (``learners/cfg.py`` ``DDPGCfg``).

The actor is deterministic, ``a = mid + half * tanh(mu(s))`` over ``action_range`` ``[lo, hi]`` (the model's squashed
mean; ``tanh(mu) = 0`` is the range's midpoint). Every noise draw goes through :class:`~robonuke_rl_core.learners.noise.
ActionNoise`: ``warmup_noise`` until this learner starts learning, ``train_noise`` after, each with the same options
(gaussian or uniform, about the actor or the midpoint, scheduled scale, clip, adaptive factor, held draws, target
smoothing); eval acts without noise. The action is clamped to the range (``clip_actions``). The critics regress on

    y = r + gamma^n * (1 - done) * reduce_target(Q'_k(s', a')),   a' = mu'(s') [+ the phase's target smoothing]

(``target_actor``: the target actor mu', Polyak-averaged as the target critics; the reduction a min over a random ``critic_subset`` of the ``num_critics`` target critics), by ``mse`` (the
mean over critics of each one's squared error) or ResFiT's ``mean_abs_squared``. The actor maximizes
``reduce_actor(Q_k(s, mu(s)))`` (``actor_q_reduction``: ``mean`` over the critics, ResFiT's ``ensemble_mean``; ``min``;
``min_random_pair``; ``q1``), plus ``action_l2_reg_weight * mean(sum_d a_d^2)``; it updates every ``policy_delay``-th
critic gradient step and its target follows after each of its steps.

Everything else is :class:`~robonuke_rl_core.learners.sac.SAC`'s, unchanged: per-agent stacked networks and
optimizers (``num_agents``), the replay memory, n-step rows, demos (``demo_transitions``, mixed into batches, the actor
on the same mixed batch with ``demo_batch_shared``), per-phase routing (``route_labels``, each phase its own learner and
clocks), held rows, CUDA graphs, checkpoints and the update metrics -- minus the entropy temperature. On top of SAC's:
the warm-up phase until this learner starts learning (``warmup_noise``; ResFiT's: uniform about the midpoint) and the
critic warm-up (``critic_warmup_updates`` critic-only gradient steps at once before the actor learns, its target
smoothing the warm-up's), both counted per learner, so a routed phase's clocks are its own.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..losses.losses import LossContext
from ..optim import BlockAdamW, clip_grad_norm_per_agent
from .base import CHECKPOINT_CRITIC_WARMUP, MEMORY_CRITIC_WARMUP, rollout_checkpoint_name, rollout_memory_name
from .noise import ActionNoise
from .sac import SAC, masked_lerp_

__all__ = ["DDPG"]


class DDPG(SAC):
    """See the module docstring. ``models`` holds ``policy`` and ``target_policy`` (same architecture), ``critics``
    and ``target_critics`` (``num_critics`` stacked)."""

    HAS_TEMPERATURE = False
    #: the critic warm-up burst publishes the metrics of its last this-many gradient steps only
    WARMUP_LOGGED_UPDATES = 1000

    def __init__(self, **kwargs) -> None:
        models = kwargs.get("models") or {}
        if models.get("target_policy") is None:
            raise ValueError("DDPG requires a 'target_policy' model (the target actor): build the models for ddpg")
        super().__init__(**kwargs)
        self.target_policy = self.models["target_policy"]
        self._sync_target_policy()
        #: action_range's midpoint and half width: the actor's action is mid + half * tanh(mu)
        lo, hi = (float(v) for v in self.cfg.action_range)
        self._range = (lo, hi)
        self._mid, self._half = (lo + hi) / 2.0, (hi - lo) / 2.0
        #: every noise draw: the warm-up's and the training's (learners/noise.py)
        self.noise = ActionNoise(self.cfg.warmup_noise, self.cfg.train_noise, action_range=self._range,
                                 num_agents=self.num_agents, device=self.device)
        self.noise.load_smoothing("train", 0)
        #: the training noise's scale at the last acting step (logged as policy/exploration_std)
        self._explore_std = torch.full((), self.noise.scale("train", 0), device=self.device)
        #: critic_warmup_updates, per agent: its burst has run (or none is asked for); _in_critic_warmup while one runs
        self._agent_warmed = [int(self.cfg.critic_warmup_updates) == 0] * self.num_agents
        self._in_critic_warmup = False
        #: init_rollout_checkpoint_saves, per agent: its env episodes finished when its warm-up ended (None before),
        #: the count its last rollout checkpoint saw, and how many it has written
        self._rollout_seen: List[Optional[int]] = [None] * self.num_agents
        self._rollout_saves = [0] * self.num_agents
        #: the critics' spread (std across the ensemble of Q(s, a) on the last batch), per agent
        self._q_spread = torch.zeros(self.num_agents, device=self.device)

    # ------------------------------------------------------------------ optimizers
    def _actor_parameters(self):
        """The actor's parameters the deterministic action uses: all but its Gaussian std's (a std parameter or head,
        ``log_std`` in the name), which no DDPG loss reaches."""
        return [p for name, p in self.policy.named_parameters() if "log_std" not in name]

    def _build_optimizers(self) -> None:
        super()._build_optimizers()
        self.policy_optimizer = BlockAdamW(
            self._actor_parameters(), self.num_agents, lr=self.cfg.actor_lr, weight_decay=self.cfg.weight_decay
        )

    # ------------------------------------------------------------------ targets
    def _sync_target_policy(self) -> None:
        self.target_policy.freeze_parameters(True)
        self.target_policy.update_parameters(self.policy, polyak=1)

    @property
    def _critic_warmed(self) -> bool:
        """Every agent's critic warm-up has run."""
        return all(self._agent_warmed)

    def _polyak_target_policy(self) -> None:
        """``target_policy <- target_policy + polyak * (policy - target_policy)``, after each actor step; with more than
        one agent only for the agents whose actor stepped (:attr:`_keep_actor`)."""
        with torch.no_grad():
            if self._masked:
                masked_lerp_(self.target_policy.parameters(), self.policy.parameters(), float(self.cfg.polyak), self._keep_actor)
                for target, online in zip(self.target_policy.buffers(), self.policy.buffers()):
                    keep = self._keep_actor.view((-1,) + (1,) * (target.dim() - 1)) if target.dim() else self._keep_actor.all()
                    target.copy_(torch.where(keep, online, target))
                return
            torch._foreach_lerp_(
                list(self.target_policy.parameters()), list(self.policy.parameters()), float(self.cfg.polyak)
            )
            for target, online in zip(self.target_policy.buffers(), self.policy.buffers()):
                target.copy_(online)

    def _checkpoint_model_keys(self):
        return ["policy", "target_policy", "critics", "target_critics"]

    def _upgrade_checkpoint_models(self, models: Dict[str, Any], path) -> Dict[str, Any]:
        """SAC's member-count check reads the stacked critics' leading dimension, which a single critic has none of."""
        if self._num_critics == 1:
            return models
        return super()._upgrade_checkpoint_models(models, path)

    # ------------------------------------------------------------------ acting
    @property
    def exploration_scale(self):
        """The adaptive factor, ``observations -> (rows, A)`` (set by the trainer from ``register_exploration_scale``)."""
        return self.noise.factor

    @exploration_scale.setter
    def exploration_scale(self, fn) -> None:
        self.noise.factor = fn

    def _agent_started(self, timestep: int):
        """Per agent, whether it may update: its own online rows (``learning_starts_rows``) and its own clock
        (``learning_starts``: env steps, or with ``route_clock`` its own steps in the phase) are there."""
        if self._base_only():
            return [False] * self.num_agents
        ready = self._agent_online_rows_ready()
        clocks = self._agent_route_steps if self.cfg.route_clock else [int(timestep)] * self.num_agents
        return [r and int(c) >= int(self.cfg.learning_starts) for r, c in zip(ready, clocks)]

    def _learning_started(self, timestep: int) -> bool:
        """Every agent may update (:meth:`_agent_started`)."""
        return all(self._agent_started(timestep))

    def _agent_random_phase(self, timestep: int):
        """The warm-up phase, per agent (``warmup_noise``): until that agent starts learning (or for ``random_timesteps``)."""
        if timestep < int(self.cfg.random_timesteps):
            return [True] * self.num_agents
        return [not started for started in self._agent_started(timestep)]

    def _random_phase(self, timestep: int) -> bool:
        """Every agent is in its warm-up phase (:meth:`_agent_random_phase`)."""
        return all(self._agent_random_phase(timestep))

    def _random_agent_mask(self, random_agents) -> torch.Tensor:
        """``random_agents`` (a host list) as a ``(num_agents,)`` bool device tensor, rebuilt only when the list
        changes: ``torch.tensor(list, device=cuda)`` every step is a host-to-device copy, i.e. a host sync."""
        key = tuple(bool(v) for v in random_agents)
        cached = getattr(self, "_random_mask_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, torch.tensor(key, dtype=torch.bool, device=self.device))
            self._random_mask_cache = cached
        return cached[1]

    def _clamp(self, actions: torch.Tensor) -> torch.Tensor:
        """``actions`` clamped to ``action_range`` (``clip_actions``), else as they are."""
        return actions.clamp(*self._range) if self.cfg.clip_actions else actions

    def _deterministic(self, model, inputs) -> tuple:
        """``(mid + half * tanh(mu(s)), outputs)`` of ``model`` (the actor or its target)."""
        _, outputs = model.act(inputs, role="policy")
        if "mean_actions" not in outputs:
            raise RuntimeError("DDPG's actor must return mean_actions (a squashed Gaussian actor's tanh(mu))")
        return self._mid + self._half * outputs["mean_actions"], outputs

    def act(self, observations, states, *, timestep: int, timesteps: int):
        if self.training and timestep == 0 and not getattr(self, "_initial_logged", False):
            self._log_initial_policy(observations)
        rows = observations.shape[0]
        width = self._space_width(self.action_space)
        if self.training and self._base_only():
            return torch.full((rows, width), self.neutral_action, device=self.device), {}
        if not self.training:
            return self.eval_actions(observations), {}
        random_agents = self._agent_random_phase(timestep)
        # the phases in use this step: the warm-up for the agents not yet learning, the training noise for the others
        # (agent i owns envs [i * envs_per_agent, (i + 1) * envs_per_agent))
        phases = (["warmup"] if any(random_agents) else []) + (["train"] if not all(random_agents) else [])
        mean, outputs = None, {}
        if any(self.noise.needs_actor(phase) for phase in phases):
            inputs = {"observations": self.normalize_observations(observations)}
            with torch.no_grad(), self._batch_mode(self.policy, False):
                mean, outputs = self._deterministic(self.policy, inputs)
        std = self.noise.scale("train", timestep)
        self._explore_std.fill_(std)
        with torch.no_grad():
            drawn = {phase: self.noise.draw(phase, mean, observations, rows, width, timestep) for phase in phases}
            if len(drawn) == 2:
                per_env = self._random_agent_mask(random_agents).repeat_interleave(rows // self.num_agents)
                actions = torch.where(per_env[:, None], drawn["warmup"], drawn["train"])
            else:
                actions = next(iter(drawn.values()))
            actions = self._clamp(actions)
        if mean is not None:
            self._emit_std_by_label({"log_std": torch.full_like(mean, math.log(max(std, 1e-12)))}, timestep)
            outputs = {**outputs, "mean_actions": mean}
        return actions, outputs

    def eval_actions(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """The eval policy's action: the scaled, clamped mean. DDPG evaluates without noise."""
        if not deterministic:
            raise ValueError("DDPG evaluates its deterministic action only: eval has no noise (deterministic: true)")
        with torch.no_grad(), self._batch_mode(self.policy, False):
            mean, _ = self._deterministic(self.policy, {"observations": self.normalize_observations(observations)})
            return self._clamp(mean)

    def _log_initial_policy(self, observations: torch.Tensor) -> None:
        """The actor as it starts, before any update, published at step 0: ``policy/mean_action`` (the mean |action|
        it would take without noise), ``policy/exploration_std`` and the learning rates."""
        self._initial_logged = True
        if not self.on_log:
            return
        rows = observations.shape[0] // self.num_agents
        with torch.no_grad(), self._batch_mode(self.policy, False):
            mean, _ = self._deterministic(self.policy, {"observations": self.normalize_observations(observations)})
        std = torch.full((self.num_agents,), self.noise.scale("train", 0), device=self.device)
        metrics = {
            "policy/mean_action": mean.abs().reshape(self.num_agents, rows, -1).mean(dim=(1, 2)),
            "policy/exploration_std": std,
            "policy/std": std,
            "lr/actor": self.policy_optimizer.lr.to(torch.float32),
            "lr/critic": self.critic_optimizer.lr.to(torch.float32),
        }
        self.emit_per_agent(metrics, 0)
        for hook in self.on_flush:  # published now, as wandb step 0
            hook(0)

    # ------------------------------------------------------------------ the update schedule
    def _actor_may_learn(self, timestep: int) -> bool:
        if self._in_critic_warmup or not self._critic_warmed:
            return False
        return super()._actor_may_learn(timestep)

    def _agent_actor_may_learn(self, agent: int, timestep: int) -> bool:
        """Agent ``agent``'s actor learns once its own critic warm-up has run."""
        if self._in_critic_warmup or not self._agent_warmed[agent]:
            return False
        return SAC._actor_may_learn(self, timestep)

    def _pool_warmup_rows(self, timestep: int) -> bool:
        """``shared_warmup``: until the agents start learning, every env's row goes to every agent's memory (the
        warm-up acts on the actors' untrained starting point, the same for every agent), so they reach
        ``learning_starts_rows`` together. Every agent's route clock and ``<phase>_step`` count the pooled steps too
        (``SAC.record_transition``), so they also reach a step-based ``learning_starts`` together."""
        return bool(self.cfg.shared_warmup) and not all(self._agent_started(timestep))

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        started = self._agent_started(timestep)
        # shared_warmup: one critic warm-up for every agent at once, on the step they start together; otherwise each
        # agent's own (with route_clock, on a step it is in its phase)
        newly = [s and not w and (self.cfg.shared_warmup or self._agent_route_now[a] or not self.cfg.route_clock)
                 for a, (s, w) in enumerate(zip(started, self._agent_warmed))]
        if any(newly):
            # ResFiT's critic warm-up, per agent: critic_warmup_updates critic-only gradient steps now, before the next
            # env step, for the agents that just started (the others untouched)
            self._in_critic_warmup = True
            self.noise.load_smoothing("warmup", timestep)  # the burst's target smoothing is the warm-up's
            rounds = -(-int(self.cfg.critic_warmup_updates) // int(self.cfg.gradient_steps))
            # only the burst's end is published: its last WARMUP_LOGGED_UPDATES gradient steps, one point per update
            # as usual, so the interval it lands in holds the critic as the actor will meet it
            logged = -(-self.WARMUP_LOGGED_UPDATES // int(self.cfg.gradient_steps))
            for round_index in range(rounds):
                self._suppress_update_metrics = round_index < rounds - logged
                self._run_update(newly, timestep, timesteps)
            self._suppress_update_metrics = False
            self._in_critic_warmup = False
            self._agent_warmed = [w or n for w, n in zip(self._agent_warmed, newly)]
            self._save_critic_warmup_checkpoints(newly, timestep)
            # the rollouts counted for init_rollout_checkpoint_saves are the ones that end after the burst
            self._rollout_seen = [self._agent_finished_episodes[a] if n else s for a, (s, n) in enumerate(zip(self._rollout_seen, newly))]
            return
        # agents that have not started or not warmed up take no step (their rows-ready / clock gate in SAC's schedule)
        self.noise.load_smoothing("train", timestep)
        super()._update_if_ready(timestep=timestep, timesteps=timesteps)
        if self.cfg.init_rollout_checkpoint_saves > 0:
            self._save_rollout_checkpoints(started, timestep)

    def _save_rollout_checkpoints(self, started, timestep: int) -> None:
        """``init_rollout_checkpoint_saves``: ``ckpt_rollout_<k>_<step>.ckpt`` for each agent one of whose envs
        ended an episode this step, after this step's update, for its first ``init_rollout_checkpoint_saves`` such
        steps after its warm-up (no warm-up: after it starts learning); ``save_memory_with_last_rollout_checkpoint``:
        the last with the agent's replay memory beside it. Host lists only (the step's host read)."""
        if self.trainer_cfg.checkpoint_interval <= 0 or self.run_dirs is None:
            return
        for agent in range(self.num_agents):
            if not (started[agent] and self._agent_warmed[agent]):
                continue
            finished = self._agent_finished_episodes[agent]
            if self._rollout_seen[agent] is None:  # no warm-up burst: counting starts when the agent starts learning
                self._rollout_seen[agent] = finished
            if finished <= self._rollout_seen[agent] or self._rollout_saves[agent] >= self.cfg.init_rollout_checkpoint_saves:
                continue
            self._rollout_seen[agent] = finished
            self._rollout_saves[agent] += 1
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / rollout_checkpoint_name(self._rollout_saves[agent], timestep)
            torch.save(self._build_checkpoint(agent, timestep), path)
            last = self._rollout_saves[agent] == self.cfg.init_rollout_checkpoint_saves
            if last and self.cfg.save_memory_with_last_rollout_checkpoint:
                self.save_memory(agent, timestep, name=rollout_memory_name(self._rollout_saves[agent], timestep))
            self._announce_checkpoint(agent, timestep, path)

    def _save_critic_warmup_checkpoints(self, agents, timestep: int) -> None:
        """``ckpt_critic_warmup.ckpt`` for each agent in ``agents`` (the ones whose burst just ran): the critics right
        after the warm-up, before any update on normal data, whatever env step that is (its ``step``). Written where the
        numbered checkpoints go (with routes, each phase's own folder), and only when they are written at all;
        ``save_memory_with_critic_warmup``: with the agent's replay memory as it was trained on beside it."""
        if self.trainer_cfg.checkpoint_interval <= 0 or self.run_dirs is None:
            return
        for agent, now in enumerate(agents):
            if not now:
                continue
            directory = self.checkpoint_dir(agent)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / CHECKPOINT_CRITIC_WARMUP
            torch.save(self._build_checkpoint(agent, timestep), path)
            if self.cfg.save_memory_with_critic_warmup:
                self.save_memory(agent, timestep, name=MEMORY_CRITIC_WARMUP)
            self._announce_checkpoint(agent, timestep, path)

    # ------------------------------------------------------------------ losses
    @staticmethod
    def _members(out: torch.Tensor) -> torch.Tensor:
        """A critic module's output as ``(critics, agents * rows, 1)``: a single stacked critic returns
        ``(agents * rows, 1)``, without the critic dimension."""
        return out.unsqueeze(0) if out.dim() == 2 else out

    def _actor_q(self, q_out: torch.Tensor, rows: int) -> torch.Tensor:
        """``actor_q_reduction`` over the critics ``(critics, agents * rows, 1)`` -> ``(agents * rows, 1)``."""
        how = self.cfg.actor_q_reduction
        if how == "mean":
            return q_out.mean(dim=0)
        if how == "min":
            return q_out.min(dim=0).values
        if how == "q1":
            return q_out[0]
        # min_random_pair: the min over two critics drawn per agent (distinct)
        per_agent = q_out.view(self._num_critics, self.num_agents, rows, -1).permute(1, 0, 2, 3)
        chosen = torch.argsort(torch.rand(self.num_agents, self._num_critics, device=self.device), dim=1)[:, :2]
        index = chosen[:, :, None, None].expand(-1, -1, rows, per_agent.shape[-1])
        return per_agent.gather(1, index).min(dim=1).values.reshape(self.num_agents * rows, -1)

    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, rows):
        with torch.no_grad():
            source = self.target_policy if self.cfg.target_actor else self.policy
            with self._batch_mode(source, False):
                next_actions, _ = self._deterministic(source, next_inputs)
            next_actions = self._clamp(self.noise.smooth(next_actions, sampled["next_observations"]))
            next_critic_actions = self._critic_actions(sampled, "next_observations", next_actions)
            with self._batch_mode(self.target_critics, True):
                target_out, _ = self.target_critics.act(
                    {**critic_next_inputs, "taken_actions": next_critic_actions}, role="target_critics"
                )
            target_out = self._members(target_out)
            rewards = sampled["rewards"]
            if self.return_scaler is not None:
                rewards = rewards / self.expand_per_agent(self.return_scaler.denominator, rows)
            terminal = sampled["terminated"] & sampled["truncated"].logical_not()
            if not self.cfg.bootstrap_time_limits:
                terminal = sampled["terminated"] | sampled["truncated"]
            discount = sampled["discounts"] if self._n_step() > 1 else self.cfg.discount_factor
            target_values = rewards + discount * terminal.logical_not() * self._target_min(target_out, rows)
            if self.cfg.q_target_min is not None:
                target_values = target_values.clamp(float(self.cfg.q_target_min), float(self.cfg.q_target_max))
        critic_actions = self._critic_actions(sampled, "observations", sampled["actions"], train=True)
        with self._batch_mode(self.critics, True):
            values, _ = self.critics.act({**critic_inputs, "taken_actions": critic_actions}, role="critics")
        values = self._members(values)
        errors = values - target_values.unsqueeze(0)  # (critics, R, 1)
        if self.cfg.critic_loss == "mean_abs_squared":
            per_row = errors.abs().mean(dim=0).pow(2)  # ResFiT: (mean_k |Q_k - y|)^2
        else:
            per_row = errors.pow(2).mean(dim=0)
        critic_loss = per_row.mean()
        per_agent = per_row.detach().view(self.num_agents, rows).mean(dim=1)
        with torch.no_grad():
            spread = values.std(dim=0) if self._num_critics > 1 else torch.zeros_like(values[0])
            self._q_spread = spread.view(self.num_agents, rows).mean(dim=1)
        second = values[1] if self._num_critics > 1 else values[0]
        #: the ensemble's mean prediction, for the health metrics (read back in the same gradient step)
        self._ensemble_mean = values.detach().mean(dim=0)
        return critic_loss, values[0], second, target_values, None, per_agent

    def _critic_prediction(self, critic_1_values, critic_2_values):
        """The ensemble's mean Q (all ``num_critics``) of the last critic step."""
        return self._ensemble_mean

    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, rows, next_inputs=None):
        with self._batch_mode(self.policy, True):
            actions, outputs = self._deterministic(self.policy, inputs)
        if self.cfg.clip_actions:  # a straight-through clamp to action_range: the value is clamped, the gradient passes
            actions = actions + (actions.clamp(*self._range) - actions).detach()
        with self._batch_mode(self.critics, False):
            q_out, _ = self.critics.act(
                {**critic_inputs, "taken_actions": self._critic_actions(sampled, "observations", actions)}, role="critics"
            )
        q = self._actor_q(self._members(q_out), rows)
        q_term = q
        online = (1.0 - sampled["is_demo"]).reshape(-1, 1) if sampled is not None and "is_demo" in sampled else None
        if self.cfg.actor_q_scale is not None:
            # TD3+BC's lambda = scale / mean|Q|, per agent, no gradient through it
            mean_abs = q.detach().abs().view(self.num_agents, rows).mean(dim=1).clamp(min=1e-6)
            q_term = q * self.expand_per_agent((float(self.cfg.actor_q_scale) / mean_abs).unsqueeze(-1), rows)
        per_row = -q_term
        metrics = {}
        if self.cfg.log_action_grad:
            # the critics' slope in the action at the actor's own action, |dQ/da| per row: the signal the actor follows
            slope = torch.autograd.grad(q.sum(), actions, retain_graph=True)[0]
            metrics["q/action_grad_norm"] = slope.detach().norm(dim=-1).view(self.num_agents, rows).mean(dim=1)
        if self.cfg.action_l2_reg_weight > 0:
            l2 = actions.pow(2).sum(dim=-1, keepdim=True)
            per_row = per_row + float(self.cfg.action_l2_reg_weight) * l2
            metrics["actor/action_l2"] = l2.detach().view(self.num_agents, rows).mean(dim=1)
        per_row = per_row.view(self.num_agents, rows)
        if online is None:
            per_agent = per_row.mean(dim=1)
        else:
            weights = online.view(self.num_agents, rows)
            per_agent = (per_row * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        outputs = {**outputs, "mean_actions": actions, "parts_metrics": metrics}
        return per_agent.mean(), actions, None, q, outputs

    def _actor_step(self, *, timestep: int, sampled, inputs, critic_inputs, rows: int, next_log_prob=None, next_inputs=None) -> dict:
        policy_loss, actions, _, q_pi, outputs = self._compute_actor_loss(
            sampled=sampled, inputs=inputs, critic_inputs=critic_inputs, rows=rows, next_inputs=next_inputs
        )
        ctx = LossContext(learner=self, step=timestep, target="policy", sampled=sampled, inputs=inputs, actions=actions,
                          log_prob=None, policy_outputs=outputs)
        aux = self.compute_aux_loss(ctx)
        if aux is not None:
            policy_loss = policy_loss + aux
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norms = clip_grad_norm_per_agent(self.policy, self.num_agents, self.cfg.grad_norm_clip)
        self.policy_optimizer.step(self._keep_actor if self._masked else None)
        self._normalize_weights(self.policy)
        if self.cfg.target_actor:
            self._polyak_target_policy()
        with torch.no_grad():
            per_agent_loss = self.per_agent_mean(-q_pi, rows)
        return {
            "policy_grad_norms": policy_grad_norms,
            "log_prob": None,
            "policy_loss": per_agent_loss,
            "log_std": None,
            "entropy_loss": None,
            "entropy_estimate": None,
            "mean_actions": actions.detach(),
            "actor_aux_metrics": {
                **{k: v.detach() for k, v in ctx.aux_metrics.items()},
                **(outputs.get("parts_metrics") or {}),
            },
        }

    def _emit_update_metrics(self, **kwargs) -> None:
        """SAC's update metrics, plus the training noise's scale (``policy/exploration_std``, also as ``policy/std``) and
        the critics' spread across the ensemble (``q/ensemble_std``)."""
        aux = dict(kwargs.get("aux_metrics") or {})
        std = self._explore_std.reshape(1).expand(self.num_agents).clone()
        aux.update({"policy/exploration_std": std, "policy/std": std, "q/ensemble_std": self._q_spread})
        kwargs["aux_metrics"] = aux
        super()._emit_update_metrics(**kwargs)

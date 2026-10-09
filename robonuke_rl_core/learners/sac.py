"""SAC over ``num_agents`` independent agents.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/sac.py``, minus AMP,
the distributed branches, the auxiliary-loss manager, the contact/rotation supervision and
the per-axis action metrics. Changed for per-agent independence: gradients are clipped per
agent and the observation/state normalizers are per agent (``BlockRunningNorm``).

Beyond classic SAC, all off by default:

* **A Q ensemble** (``num_critics`` > 2): the bootstrap target is the min over a random
  ``critic_subset`` of the target critics, drawn per agent (REDQ, Chen et al. 2021); the actor
  reduces the critics' Q by ``actor_q_reduction``.
* **A delayed actor** (``policy_delay``): the actor and temperature step once every so many
  gradient steps, so ``gradient_steps`` raises the critic's update-to-data ratio alone.
* **A critic-only start** (``policy_learning_starts``).
* **Demo transitions in the critic's batches** (``demo_transitions`` with the
  ``demo_sampling_*`` window): half demo, half online, RLPD's symmetric sampling — for the
  critic only; the actor's batch is always online.

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

from ..optim import BlockAdamW, clip_grad_norm_per_agent, lr_at
from ..models.factory import build_models
from ..models.normalizer import BlockRunningNorm
from ..losses.losses import LossContext
from .base import LearnerBase

__all__ = ["SAC"]


class SAC(LearnerBase):
    def __init__(self, *, model_cfg=None, controller_cfg=None, **kwargs) -> None:
        """``model_cfg`` and ``controller_cfg`` are only needed when
        ``sac.periodic_reset_enabled`` rebuilds models — a rebuild has to reproduce the same
        actor, including the derived MATCH pairs."""
        super().__init__(**kwargs)
        self._model_cfg = model_cfg
        self._controller_cfg = controller_cfg

        self._num_critics = int(self.cfg.num_critics)
        required = ("policy", *self._critic_keys(), *self._target_keys())
        missing = [k for k in required if self.models.get(k) is None]
        if missing:
            raise ValueError(
                f"SAC requires models {required} (sac.num_critics={self._num_critics}); missing "
                f"or None: {missing}"
            )
        extra = sorted(set(k for k in self.models if k.startswith(("critic_", "target_critic_"))) - set(required))
        if extra:
            raise ValueError(
                f"SAC got critics {extra} beyond sac.num_critics={self._num_critics}; build the "
                "models with the same num_critics"
            )
        self.policy = self.models["policy"]
        self._bind_critics()

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
            # no weight decay on log_alpha: pulling it toward 0 has no meaning. Its leading
            # dim is the agent, so one BlockAdamW is N independent temperature optimizers.
            self.entropy_optimizer = BlockAdamW(
                [self.log_entropy_coefficient],
                self.num_agents,
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

        self._build_optimizers()
        self._sync_targets()

        if self.cfg.normalize_observations:
            self.observation_normalizer = self.make_normalizer(self.observation_space)
            self.state_normalizer = (
                self.make_normalizer(self.state_space) if self._asymmetric else None
            )
        else:
            self.observation_normalizer = None
            self.state_normalizer = None

        # loaded with the replay memory (``_create_memory_tensors``): eval builds this learner
        # with no memory and never needs the demos
        self._demo: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ setup helpers
    def _critic_keys(self) -> List[str]:
        return [f"critic_{k}" for k in range(1, self._num_critics + 1)]

    def _target_keys(self) -> List[str]:
        return [f"target_critic_{k}" for k in range(1, self._num_critics + 1)]

    def _bind_critics(self) -> None:
        """``critic_k`` / ``target_critic_k`` attributes and the ``critics`` / ``target_critics``
        lists, from ``self.models``."""
        for key in (*self._critic_keys(), *self._target_keys()):
            setattr(self, key, self.models[key])
        self.critics = [self.models[k] for k in self._critic_keys()]
        self.target_critics = [self.models[k] for k in self._target_keys()]

    def _sync_targets(self) -> None:
        for target, critic in zip(self.target_critics, self.critics):
            target.freeze_parameters(True)
            target.update_parameters(critic, polyak=1)

    def _policy_learning_starts(self) -> int:
        if self.cfg.policy_learning_starts is None:
            return int(self.cfg.learning_starts)
        return int(self.cfg.policy_learning_starts)

    def _demo_names(self) -> List[str]:
        names = ["observations", "actions", "rewards", "next_observations", "terminated", "truncated"]
        if self._asymmetric:
            names += ["states", "next_states"]
        return names

    def _load_demo_transitions(self) -> Dict[str, torch.Tensor]:
        """The demo file's tensors on the device, checked against the agent's spaces."""
        path = self.cfg.demo_transitions
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"sac.demo_transitions: no file {path}") from None
        if not isinstance(data, dict) or "transitions" not in data:
            raise ValueError(
                f"sac.demo_transitions ({path}) must be a dict with a 'transitions' dict of "
                "tensors (and optional 'meta')"
            )
        transitions = data["transitions"]
        names = self._demo_names()
        missing = [n for n in names if n not in transitions]
        if missing:
            raise KeyError(f"sac.demo_transitions ({path}) is missing {missing}; needs {names}")
        widths = {
            "observations": self._space_width(self.observation_space),
            "next_observations": self._space_width(self.observation_space),
            "actions": self._space_width(self.action_space),
            "rewards": 1,
            "terminated": 1,
            "truncated": 1,
        }
        if self._asymmetric:
            widths["states"] = widths["next_states"] = self._space_width(self.state_space)
        rows = None
        out = {}
        for name in names:
            tensor = transitions[name]
            if tensor.dim() != 2 or tensor.shape[1] != widths[name]:
                raise ValueError(
                    f"sac.demo_transitions ({path}): '{name}' must be (rows, {widths[name]}), got "
                    f"{tuple(tensor.shape)} — was the file made for this env and agent?"
                )
            if rows is None:
                rows = tensor.shape[0]
            elif tensor.shape[0] != rows:
                raise ValueError(
                    f"sac.demo_transitions ({path}): '{name}' has {tensor.shape[0]} rows, "
                    f"'{names[0]}' has {rows}"
                )
            dtype = torch.bool if name in ("terminated", "truncated") else torch.float32
            out[name] = tensor.to(device=self.device, dtype=dtype)
        if not rows:
            raise ValueError(f"sac.demo_transitions ({path}) holds no transitions")
        # actions may lie beyond the squashed policy's [-1, 1]: only the critic sees demo rows,
        # and it takes the action as an input, so the value of a correction the actor cannot
        # reach is still a true value. They must be finite.
        for name in names:
            if out[name].is_floating_point() and not torch.isfinite(out[name]).all():
                raise ValueError(f"sac.demo_transitions ({path}): '{name}' holds non-finite values")
        print(f"[sac] {rows} demo transitions from {path}", flush=True)
        return out

    @staticmethod
    def _space_width(space) -> int:
        if isinstance(space, int):
            return int(space)
        return int(np.prod(space.shape))

    def _demo_active(self, timestep: int) -> bool:
        return self._demo is not None and (
            self.cfg.demo_sampling_start <= timestep < self.cfg.demo_sampling_end
        )

    def _sample(self, rows: int) -> Dict[str, torch.Tensor]:
        return dict(
            zip(self._tensors_names, self.memory.sample(names=self._tensors_names, batch_size=rows)[0])
        )

    def _mixed_batch(self, rows: int) -> Dict[str, torch.Tensor]:
        """The critic's batch inside the demo window: per agent, ``min(rows // 2, stored)``
        online rows, the rest demo rows. Draw shapes depend only on the step count, never on
        the data, and each agent draws its own demo rows, so agents stay independent."""
        n_online = min(rows // 2, int(self.memory.size))
        n_demo = rows - n_online
        names = self._demo_names()
        demo_rows = next(iter(self._demo.values())).shape[0]
        index = torch.randint(0, demo_rows, (self.num_agents, n_demo), device=self.device)
        online = self._sample(n_online) if n_online else None
        batch = {}
        for name in names:
            demo = self._demo[name][index]  # (num_agents, n_demo, width)
            if online is not None:
                part = online[name].view(self.num_agents, n_online, -1)
                demo = torch.cat((part.to(demo.dtype), demo), dim=1)
            batch[name] = demo.reshape(self.num_agents * rows, -1)
        return batch

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
        self.policy_optimizer = BlockAdamW(
            self.policy.parameters(),
            self.num_agents,
            lr=self.cfg.actor_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.critic_optimizer = BlockAdamW(
            itertools.chain(*(critic.parameters() for critic in self.critics)),
            self.num_agents,
            lr=self.cfg.critic_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self._updates = 0

    def _set_learning_rates(self, timesteps: int) -> None:
        """The LR for this gradient step, from the update count alone (never from the data)."""
        if self.cfg.lr_schedule == "constant":
            return
        total = max(1, (int(timesteps) - int(self.cfg.learning_starts)) * int(self.cfg.gradient_steps))
        self.policy_optimizer.set_lr(
            lr_at(self._updates, total, self.cfg.actor_lr, self.cfg.lr_end, self.cfg.lr_schedule)
        )
        self.critic_optimizer.set_lr(
            lr_at(self._updates, total, self.cfg.critic_lr, self.cfg.lr_end, self.cfg.lr_schedule)
        )

    def _create_memory_tensors(self) -> None:
        if self.memory is None:
            raise ValueError("SAC needs a replay memory")
        self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="next_observations", size=self.observation_space, dtype=torch.float32)
        self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
        # truncated is stored too, because the bootstrap must tell a terminal state from the
        # clock running out: Isaac Lab's Factory and Forge raise BOTH flags at the time limit
        self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
        self._tensors_names = [
            "observations", "actions", "rewards", "next_observations", "terminated", "truncated",
        ]
        if self._asymmetric:
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.state_space, dtype=torch.float32)
            self._tensors_names += ["states", "next_states"]
        # whatever the configured aux losses need per transition (e.g. contact flags)
        self._tensors_names += self.create_aux_memory_tensors()
        if self.cfg.demo_transitions:
            self._demo = self._load_demo_transitions()
        if self._demo is not None and self.aux_memory_keys():
            raise ValueError(
                "sac.demo_transitions cannot be mixed into batches while aux losses need per-"
                f"transition keys {sorted(self.aux_memory_keys())}: the demos do not carry them"
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

    # ------------------------------------------------------------------ interaction
    def act(self, observations, states, *, timestep: int, timesteps: int):
        if self.training and timestep < self.cfg.random_timesteps:
            return self.random_actions(observations.shape[0]), {}
        inputs = {"observations": self.normalize_observations(observations)}
        # no_grad: these actions only drive the env; the update re-runs the policy on replay
        # batches. A live graph here would be spliced into the env's action buffers and grow.
        with torch.no_grad():
            actions, outputs = self.policy.act(inputs, role="policy")
        self.emit_selection(outputs, timestep)
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
        self.observe_step(
            infos=infos,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            step=timestep,
        )

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
        extra.update(self.aux_memory_values(infos))
        self.memory.add_samples(
            observations=observations,
            actions=actions,
            rewards=self.shape_rewards(rewards, timestep, timesteps),
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
            **extra,
        )

    def _update_if_ready(self, *, timestep: int, timesteps: int) -> None:
        # learning_starts is the first update of anything (the critics); the actor joins at
        # policy_learning_starts
        if timestep >= self.cfg.learning_starts:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)
        self._maybe_periodic_reset(timestep)

    # ------------------------------------------------------------------ losses
    def _target_min(self, target_qs: List[torch.Tensor], rows: int) -> torch.Tensor:
        """Min over the target critics: all of them, or a random ``critic_subset`` per agent."""
        subset = int(self.cfg.critic_subset)
        if subset == self._num_critics:
            # the classic path draws nothing, so its RNG stream is the twin's
            return torch.stack(target_qs).min(dim=0).values if self._num_critics > 2 else torch.min(*target_qs)
        stacked = torch.stack(target_qs).view(self._num_critics, self.num_agents, rows, -1)
        per_agent = stacked.permute(1, 0, 2, 3)  # (agents, critics, rows, 1)
        chosen = torch.argsort(torch.rand(self.num_agents, self._num_critics, device=self.device), dim=1)[:, :subset]
        index = chosen[:, :, None, None].expand(-1, -1, rows, per_agent.shape[-1])
        return per_agent.gather(1, index).min(dim=1).values.reshape(self.num_agents * rows, -1)

    def _compute_critic_loss(self, *, sampled, inputs, next_inputs, critic_inputs, critic_next_inputs, rows):
        """Min-over-targets bootstrapped target + mean MSE over the critics."""
        with torch.no_grad():
            next_actions, outputs = self.policy.act(next_inputs, role="policy")
            target_qs = [
                target.act({**critic_next_inputs, "taken_actions": next_actions}, role=key)[0]
                for target, key in zip(self.target_critics, self._target_keys())
            ]
            entropy = self.expand_per_agent(self._entropy_coefficient, rows)
            target_q = self._target_min(target_qs, rows) - entropy * outputs["log_prob"]
            # Bootstrap unless the next state is genuinely terminal. A time limit is not the
            # end of the world: the episode had a future, it just stopped being observed, so
            # dropping gamma*V(next) there would teach the critic that running out the clock
            # is worth nothing. Isaac Lab's Factory and Forge raise terminated AND truncated
            # at the limit (``_get_dones`` returns one ``time_out`` tensor twice), so reading
            # ``terminated`` alone would do exactly that to every episode.
            terminal = sampled["terminated"] & sampled["truncated"].logical_not()
            target_values = (
                sampled["rewards"]
                + self.cfg.discount_factor * terminal.logical_not() * target_q
            )

        values = [
            critic.act({**critic_inputs, "taken_actions": sampled["actions"]}, role=key)[0]
            for critic, key in zip(self.critics, self._critic_keys())
        ]
        critic_loss = sum(F.mse_loss(v, target_values) for v in values) / self._num_critics
        self._critic_values = values
        return critic_loss, values[0], values[1], target_values

    def _compute_actor_loss(self, *, sampled, inputs, critic_inputs, rows):
        """``alpha * log_pi - Q``, with Q the min (or mean, ``actor_q_reduction``) over the
        critics."""
        actions, outputs = self.policy.act(inputs, role="policy")
        log_prob = outputs["log_prob"]
        q_pi = [
            critic.act({**critic_inputs, "taken_actions": actions}, role=key)[0]
            for critic, key in zip(self.critics, self._critic_keys())
        ]
        if self.cfg.actor_q_reduction == "mean":
            q = torch.stack(q_pi).mean(dim=0)
        elif self._num_critics == 2:
            q = torch.min(*q_pi)  # the twin's exact expression
        else:
            q = torch.stack(q_pi).min(dim=0).values
        entropy = self.expand_per_agent(self._entropy_coefficient, rows)
        policy_loss = (entropy * log_prob - q).mean()
        return policy_loss, actions, log_prob, q, outputs

    # ------------------------------------------------------------------ update
    def update(self, *, timestep: int, timesteps: int) -> None:
        with self.update_timer():  # publishes stats/update_time_ms with the rest
            self._update(timestep=timestep, timesteps=timesteps)

    def _update(self, *, timestep: int, timesteps: int) -> None:
        rows = self.cfg.batch_size  # per agent; the memory returns N * batch_size rows
        policy_on = timestep >= self._policy_learning_starts()
        demo_on = self._demo_active(timestep)

        for gradient_step in range(self.cfg.gradient_steps):
            self._set_learning_rates(timesteps)
            # outside the demo window one online batch serves both losses, as in classic SAC;
            # inside it the critic's batch is mixed and the actor's stays online
            online = self._sample(rows) if (policy_on or not demo_on) else None
            sampled = self._mixed_batch(rows) if demo_on else online
            inputs, next_inputs, critic_inputs, critic_next_inputs = self._batch_inputs(sampled)

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
                self.critics, self.num_agents, self.cfg.grad_norm_clip
            )
            self.critic_optimizer.step()

            # the actor (and temperature) step on every policy_delay-th gradient step
            actor_step = policy_on and (self._updates + 1) % self.cfg.policy_delay == 0
            actor_metrics = {}
            if actor_step:
                if online is not sampled:
                    # a second batch this step: it must not count twice in the normalizer
                    inputs, _, critic_inputs, _ = self._batch_inputs(online, train=False)
                actor_metrics = self._actor_step(
                    timestep=timestep, sampled=online, inputs=inputs, critic_inputs=critic_inputs, rows=rows
                )

            for target, critic in zip(self.target_critics, self.critics):
                target.update_parameters(critic, polyak=self.cfg.polyak)

            self._updates += 1

            if self.on_log:
                self._emit_update_metrics(
                    timestep=timestep,
                    rows=rows,
                    critic_1_values=critic_1_values,
                    critic_2_values=critic_2_values,
                    target_values=target_values,
                    critic_grad_norms=critic_grad_norms,
                    demo_rows=(rows - min(rows // 2, int(self.memory.size))) if demo_on else 0,
                    **actor_metrics,
                )

    def _batch_inputs(self, sampled, train: bool = True):
        """The (normalized) actor and critic inputs of a sampled batch; ``train`` updates the
        normalizers' running stats with it."""
        inputs = {"observations": self.normalize_observations(sampled["observations"], train=train)}
        next_inputs = {
            "observations": self.normalize_observations(sampled["next_observations"], train=train)
        }
        if self._asymmetric:
            critic_inputs = {"observations": self.normalize_states(sampled["states"], train=train)}
            critic_next_inputs = {
                "observations": self.normalize_states(sampled["next_states"], train=train)
            }
        else:
            critic_inputs, critic_next_inputs = inputs, next_inputs
        return inputs, next_inputs, critic_inputs, critic_next_inputs

    def _actor_step(self, *, timestep: int, sampled, inputs, critic_inputs, rows: int) -> dict:
        policy_loss, actions, log_prob, q_pi, outputs = self._compute_actor_loss(
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
        if not self.on_log:
            return {}
        with torch.no_grad():
            per_agent_loss = self.per_agent_mean(
                self.expand_per_agent(self._entropy_coefficient, rows) * log_prob - q_pi, rows
            )
        return {
            "policy_grad_norms": policy_grad_norms,
            "log_prob": log_prob,
            "policy_loss": per_agent_loss,
            "policy_outputs": outputs,
        }

    def _update_entropy(self, log_prob: torch.Tensor, rows: int) -> None:
        """Per-agent temperature step: each agent's loss term holds only its own rows."""
        log_prob_per_agent = log_prob.view(self.num_agents, rows, 1).mean(dim=1)  # (N, 1)
        if self.cfg.entropy_loss_form == "alpha":
            # the multiplicative form: alpha * (H_hat - H_target); the adaptation rate is ~alpha
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
        self._entropy_loss = entropy_loss.detach().reshape(-1)

    def _emit_update_metrics(
        self,
        *,
        timestep: int,
        rows: int,
        critic_1_values,
        critic_2_values,
        target_values,
        critic_grad_norms,
        demo_rows: int = 0,
        policy_grad_norms=None,
        log_prob=None,
        policy_loss=None,
        policy_outputs=None,
    ) -> None:
        """Critic metrics every gradient step; the actor's only on a step that updated it."""
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
                "lr/actor": self.policy_optimizer.lr.to(torch.float32),
                "lr/critic": self.critic_optimizer.lr.to(torch.float32),
            }
            if self._demo is not None:
                # demo rows per critic batch, as a fraction: 0 outside the window
                metrics["sac/demo_fraction"] = torch.full(
                    (self.num_agents,), demo_rows / rows, device=self.device
                )
            if policy_grad_norms is not None:
                metrics["grad_norm/policy"] = policy_grad_norms
            if log_prob is not None:
                metrics["policy/log_prob"] = self.per_agent_mean(log_prob, rows)
            if policy_loss is not None:
                # the actor's own objective: alpha * log_pi - min(Q1, Q2), per agent
                metrics["loss/policy"] = policy_loss
            if getattr(self, "_entropy_loss", None) is not None:
                # the temperature's loss, i.e. how hard alpha is being pushed
                metrics["loss/entropy"] = self._entropy_loss
            if policy_outputs is not None and "log_std" in policy_outputs:
                metrics["policy/std"] = (
                    policy_outputs["log_std"].exp().view(self.num_agents, rows, -1).mean(dim=(1, 2))
                )
            metrics.update(self.stats_metrics())
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
            "sac",
            self._model_cfg,
            self.observation_space,
            self.state_space,
            self.action_space,
            self.num_agents,
            self.device,
            self._controller_cfg,
            num_critics=self._num_critics,
        )
        for key in ("policy", *self._critic_keys(), *self._target_keys()):
            self.models[key] = fresh[key]
        self.policy = self.models["policy"]
        self._bind_critics()
        self._sync_targets()

        self._build_optimizers()
        self._entropy_coefficient = torch.full(
            (self.num_agents, 1), float(self.cfg.initial_entropy_value), device=self.device
        )
        if self.cfg.learn_entropy:
            self.log_entropy_coefficient = torch.log(
                self._entropy_coefficient.clone()
            ).requires_grad_(True)
            self.entropy_optimizer = BlockAdamW(
                [self.log_entropy_coefficient],
                self.num_agents,
                lr=self.cfg.entropy_lr,
                weight_decay=0.0,
            )

    # ------------------------------------------------------------------ checkpoints
    def _checkpoint_model_keys(self) -> List[str]:
        # the target critics are saved too: resuming must keep their EMA lag, and loading one
        # slot must not touch any other agent's targets
        return ["policy", *self._critic_keys(), *self._target_keys()]

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

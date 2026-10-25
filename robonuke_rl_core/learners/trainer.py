"""The training loop: skrl's sequential trainer plus the env reset a learner may ask for.

skrl's ``SequentialTrainer.train`` keeps the observations the last ``env.step`` returned and
feeds them to the next ``act``; nothing in its loop lets the agent reset the env in between.
PPO's ``reset_every_iteration`` (ResiP) needs exactly that: every env reset right after the
update, and the NEXT rollout starting from the fresh observations. So :class:`Trainer`
reproduces the single-agent loop step for step with one addition after ``post_interaction``:
when the learner raised ``env_reset_requested``, the trainer resets every env
(:func:`~robonuke_rl_core.evaluation.force_env_reset`, which also clears skrl's one-shot reset
guard and checks the clocks really went to zero), takes the fresh observations and states as
the next step's inputs and tells the learner (``forced_env_reset``). With no request the loop
is skrl's.
"""

from __future__ import annotations

import sys

import torch
import tqdm
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.utils import ScopedTimer

from ..evaluation import force_env_reset

__all__ = ["Trainer", "SequentialTrainerCfg"]


class Trainer(SequentialTrainer):
    """skrl's sequential trainer for ONE learner (which holds every agent), honouring the
    learner's ``env_reset_requested`` between env steps."""

    def train(self) -> None:
        if self.num_simultaneous_agents != 1:
            raise ValueError(
                "robonuke_rl_core's Trainer drives one learner (its agents live inside it); "
                f"got {self.num_simultaneous_agents} simultaneous skrl agents"
            )
        agent = self.agents
        timesteps = self.cfg.timesteps
        agent.enable_training_mode(True)

        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(range(timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout):
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)

            with torch.no_grad():
                with ScopedTimer() as timer:
                    actions, outputs = agent.act(observations, states, timestep=timestep, timesteps=timesteps)
                    agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    agent.track_data("Stats / Env stepping time (ms)", timer.elapsed_time_ms)

                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                agent.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )

                if self.cfg.environment_info in infos:
                    for key, value in infos[self.cfg.environment_info].items():
                        if isinstance(value, torch.Tensor) and value.numel() == 1:
                            agent.track_data(key if "/" in key else f"Info / {key}", value.item())

            agent.post_interaction(timestep=timestep, timesteps=timesteps)

            if getattr(agent, "env_reset_requested", False):
                # the learner's update is done; the next rollout starts from fresh episodes
                with torch.no_grad():
                    observations, states = force_env_reset(self.env)
                agent.forced_env_reset()
            elif self.env.num_envs > 1:
                observations, states = next_observations, next_states
            elif terminated.any() or truncated.any():
                # skrl's single-env branch: the one episode ended, start the next
                with torch.no_grad():
                    observations, infos = self.env.reset()
                    states = self.env.state()
            else:
                observations, states = next_observations, next_states

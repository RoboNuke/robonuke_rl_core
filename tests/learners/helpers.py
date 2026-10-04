"""Builders for CPU learner tests: tiny networks, no Isaac Lab, no env.

Every learner is driven directly (``update`` / ``record_transition`` / ``post_interaction``)
with a hand-filled memory, so these tests need torch, skrl and gymnasium only.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium
import numpy as np
import torch

from robonuke_rl_core.learners.base import run_dirs  # noqa: F401  (re-exported for tests)
from robonuke_rl_core.learners.cfg import FlashSACCfg, PPOCfg, SACCfg, TrainerCfg
from robonuke_rl_core.learners.flash_sac import FlashSAC
from robonuke_rl_core.learners.ppo import PPO
from robonuke_rl_core.learners.sac import SAC
from robonuke_rl_core.memory.multi_random import MultiRandomMemory
from robonuke_rl_core.models.cfg import ActorCfg, CriticCfg, ModelCfg
from robonuke_rl_core.models.factory import build_models

OBS_DIM = 4
STATE_DIM = 6
ACT_DIM = 2
LEARNER_CLASSES = {"sac": SAC, "ppo": PPO, "flash_sac": FlashSAC}


def tiny_model_cfg() -> ModelCfg:
    return ModelCfg(
        actor=ActorCfg(actor_n=1, actor_latent=8),
        critic=CriticCfg(critic_n=1, critic_latent=8, n_atoms=5, v_min=-2.0, v_max=2.0),
    )


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def learner_cfg(learner: str, **overrides) -> Any:
    """A tiny but complete cfg for one learner."""
    if learner == "sac":
        cfg = SACCfg(batch_size=4, gradient_steps=1, learning_starts=0)
    elif learner == "flash_sac":
        cfg = FlashSACCfg(batch_size=4, gradient_steps=1, learning_starts=0)
    elif learner == "ppo":
        cfg = PPOCfg(rollouts=4, learning_epochs=2, mini_batches=2, learning_starts=0)
    else:
        raise ValueError(learner)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"{learner} cfg has no field '{key}'")
        setattr(cfg, key, value)
    return cfg


def build_learner(
    learner: str,
    *,
    num_agents: int = 3,
    envs_per_agent: int = 2,
    rollout: int = 4,
    asymmetric: bool = False,
    run_dirs_list=None,
    trainer_overrides: Dict[str, Any] | None = None,
    **cfg_overrides,
):
    """A learner with tiny networks and an empty memory of depth ``rollout``."""
    torch.manual_seed(0)
    num_envs = num_agents * envs_per_agent
    cfg = learner_cfg(learner, **cfg_overrides)
    trainer_fields = {
        "learner": learner,
        "total_timesteps": 100,
        "write_interval": 4,
        "checkpoint_interval": 0,
        **(trainer_overrides or {}),
    }
    trainer_cfg = TrainerCfg(**trainer_fields)

    observation_space = box(OBS_DIM)
    state_space = box(STATE_DIM) if asymmetric else None
    action_space = box(ACT_DIM)
    model_cfg = tiny_model_cfg()
    models = build_models(
        learner, model_cfg, observation_space, state_space, action_space, num_agents, "cpu"
    )
    memory = MultiRandomMemory(
        memory_size=rollout, num_envs=num_envs, num_agents=num_agents, device="cpu"
    )
    extra = {"model_cfg": model_cfg} if learner in ("sac", "flash_sac") else {}
    instance = LEARNER_CLASSES[learner](
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device="cpu",
        cfg=cfg,
        trainer_cfg=trainer_cfg,
        num_agents=num_agents,
        num_envs=num_envs,
        run_dirs=run_dirs_list,
        **extra,
    )
    instance.init()
    instance.enable_training_mode(True)
    return instance


def fill_memory(learner, steps: int | None = None, seed: int = 1) -> None:
    """Write random transitions for every env, through the learner's own record path."""
    torch.manual_seed(seed)
    steps = steps or learner.memory.memory_size
    num_envs = learner.num_envs
    for step in range(steps):
        observations = torch.randn(num_envs, OBS_DIM)
        next_observations = torch.randn(num_envs, OBS_DIM)
        states = torch.randn(num_envs, STATE_DIM) if learner._asymmetric else None
        next_states = torch.randn(num_envs, STATE_DIM) if learner._asymmetric else None
        rewards = torch.randn(num_envs, 1)
        terminated = torch.zeros(num_envs, 1, dtype=torch.bool)
        truncated = torch.zeros(num_envs, 1, dtype=torch.bool)
        if step == steps - 1:
            terminated[:] = True  # one finished episode per env
        # the policy's own action, so PPO's stored log_prob matches the stored action
        actions, _ = learner.act(observations, states, timestep=step, timesteps=100)
        learner.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos={},
            timestep=step,
            timesteps=100,
        )
    learner._next_observations = torch.randn(num_envs, OBS_DIM)
    learner._next_states = torch.randn(num_envs, STATE_DIM) if learner._asymmetric else None

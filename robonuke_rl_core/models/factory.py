"""Model construction: one builder per learner.

:func:`build_models` is the single place networks are built. The train script calls it once;
SAC's SimBa periodic reset calls it again to rebuild fresh networks mid-run.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict

from skrl.models.torch import Model

from .block_simba import BlockSimBaActor, BlockSimBaQCritic, BlockSimBaValueCritic
from .cfg import ModelCfg
from .flash_sac import FlashCategoricalQCritic, FlashSimBaActor

__all__ = ["MODEL_BUILDERS", "build_models"]


def _kwargs(section: Any) -> dict:
    return dataclasses.asdict(section)


def _build_sac(model_cfg: ModelCfg, obs_space, state_space, action_space, num_agents, device):
    """Squashed-Gaussian actor + twin Q critics + their targets."""
    critic_space = state_space if state_space is not None else obs_space

    def make_q():
        return BlockSimBaQCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_simba_critic_kwargs(model_cfg),
        )

    return {
        "policy": BlockSimBaActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_kwargs(model_cfg.actor),
        ),
        "critic_1": make_q(),
        "critic_2": make_q(),
        "target_critic_1": make_q(),
        "target_critic_2": make_q(),
    }


def _build_ppo(model_cfg: ModelCfg, obs_space, state_space, action_space, num_agents, device):
    """Squashed-Gaussian actor + one state-value critic."""
    critic_space = state_space if state_space is not None else obs_space
    return {
        "policy": BlockSimBaActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_kwargs(model_cfg.actor),
        ),
        "value": BlockSimBaValueCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_simba_critic_kwargs(model_cfg),
        ),
    }


def _build_flash_sac(model_cfg: ModelCfg, obs_space, state_space, action_space, num_agents, device):
    """FlashSAC actor + twin categorical critics and their targets."""
    critic_space = state_space if state_space is not None else obs_space
    if model_cfg.actor.bernoulli_action_dims or model_cfg.actor.force_zero_action_dims:
        raise ValueError(
            "FlashSAC supports continuous actions only; model.actor.bernoulli_action_dims and "
            "force_zero_action_dims must be empty"
        )

    def make_q():
        return FlashCategoricalQCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_flash_critic_kwargs(model_cfg),
        )

    return {
        "policy": FlashSimBaActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_kwargs(model_cfg.actor),
        ),
        "critic_1": make_q(),
        "critic_2": make_q(),
        "target_critic_1": make_q(),
        "target_critic_2": make_q(),
    }


def _simba_critic_kwargs(model_cfg: ModelCfg) -> dict:
    """SimBa critic kwargs: the categorical support is FlashSAC-only."""
    critic = model_cfg.critic
    return {
        "critic_n": critic.critic_n,
        "critic_latent": critic.critic_latent,
        "critic_output_init_mean": critic.critic_output_init_mean,
        "clip_actions": critic.clip_actions,
    }


def _flash_critic_kwargs(model_cfg: ModelCfg) -> dict:
    critic = model_cfg.critic
    return {
        "critic_n": critic.critic_n,
        "critic_latent": critic.critic_latent,
        "clip_actions": critic.clip_actions,
        "n_atoms": critic.n_atoms,
        "v_min": critic.v_min,
        "v_max": critic.v_max,
    }


#: learner name -> builder. A new learner adds its entry here.
MODEL_BUILDERS: Dict[str, Callable[..., Dict[str, Model]]] = {
    "sac": _build_sac,
    "ppo": _build_ppo,
    "flash_sac": _build_flash_sac,
}


def build_models(
    learner: str,
    model_cfg: ModelCfg,
    observation_space,
    state_space,
    action_space,
    num_agents: int,
    device,
) -> Dict[str, Model]:
    """Build the networks ``learner`` needs.

    ``state_space`` not None means asymmetric actor-critic: the critic consumes the state
    vector while the actor keeps the policy observation.
    """
    if learner not in MODEL_BUILDERS:
        raise ValueError(
            f"no model builder for learner '{learner}'; known: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[learner](
        model_cfg, observation_space, state_space, action_space, num_agents, device
    )

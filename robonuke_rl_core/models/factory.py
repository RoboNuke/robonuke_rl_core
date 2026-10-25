"""Model construction: one builder per learner.

:func:`build_models` is the single place networks are built. The train script calls it once;
SAC's SimBa periodic reset calls it again to rebuild fresh networks mid-run.

Every model is a plain single-agent module stacked by ``VmapEnsemble``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict

from skrl.models.torch import Model

from .simba import EnsembleActor, EnsembleQCritic, EnsembleValueCritic
from .mlp import MlpActor, MlpQCritic, MlpValueCritic
from .cfg import MlpModelCfg, SimbaModelCfg

__all__ = ["MODEL_BUILDERS", "build_models", "actor_kwargs"]


def _kwargs(section: Any) -> dict:
    return dataclasses.asdict(section)


def actor_kwargs(model_cfg, controller_cfg=None) -> dict:
    """The actor's config fields plus whatever the action layout contributes.

    The gated ``(pose, force)`` pairs ``selection_distribution: match`` conditions on are
    **derived**, never configured: they follow from the controller's capabilities and its
    ``force_axes`` mask, which is the same object that decides where the selection block
    sits. An experiment that hand-wrote them could shift one index and silently gate the
    wrong axis. No controller (or one with no selection block) contributes nothing, and an
    actor asked for ``match`` then raises for want of pairs.
    """
    kwargs = _kwargs(model_cfg.actor)
    if controller_cfg is None or not getattr(controller_cfg, "enabled", False):
        return kwargs
    from ..envs.interface import ActionLayout

    layout = ActionLayout(controller_cfg)
    if not layout.selection_dim:
        return kwargs
    kwargs["pos_component_dims"] = layout.pos_component_indices
    kwargs["force_component_dims"] = layout.force_component_indices
    kwargs["selection_names"] = layout.selection_axis_names
    return kwargs


def _build_sac(
    model_cfg: SimbaModelCfg,
    obs_space,
    state_space,
    action_space,
    num_agents,
    device,
    actor,
    num_critics: int = 2,
):
    """Squashed-Gaussian actor + ``num_critics`` Q critics stacked in one module
    (``critics``, one vmapped call for all of them) and their targets (``target_critics``).
    Two is classic SAC's twin; more is a REDQ / RLPD ensemble (``sac.num_critics``)."""
    if num_critics < 2:
        raise ValueError(f"SAC needs at least 2 critics (sac.num_critics), got {num_critics}")
    critic_space = state_space if state_space is not None else obs_space

    def make_q():
        return EnsembleQCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            members=num_critics,
            **_critic_kwargs(model_cfg),
        )

    models = {
        "policy": EnsembleActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **actor,
        ),
    }
    # the critics first, then the targets: the build order fixes the RNG stream (member-major,
    # the order separate per-critic modules were built in)
    models["critics"] = make_q()
    models["target_critics"] = make_q()
    return models


def _build_ppo(
    model_cfg: SimbaModelCfg,
    obs_space,
    state_space,
    action_space,
    num_agents,
    device,
    actor,
    num_critics: int = 2,
):
    """Squashed-Gaussian actor + one state-value critic. ``num_critics`` is SAC's; PPO has
    one value critic and takes only the default."""
    if num_critics != 2:
        raise ValueError(f"num_critics is a SAC option; PPO builds one value critic (got {num_critics})")
    critic_space = state_space if state_space is not None else obs_space
    return {
        "policy": EnsembleActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **actor,
        ),
        "value": EnsembleValueCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_critic_kwargs(model_cfg),
        ),
    }


def _critic_kwargs(model_cfg: SimbaModelCfg) -> dict:
    critic = model_cfg.critic
    return {
        "critic_n": critic.critic_n,
        "critic_latent": critic.critic_latent,
        "critic_output_init_mean": critic.critic_output_init_mean,
        "clip_actions": critic.clip_actions,
    }


def _build_mlp_sac(
    model_cfg: MlpModelCfg,
    obs_space,
    state_space,
    action_space,
    num_agents,
    device,
    actor,
    num_critics: int = 2,
):
    """``architecture: mlp`` for SAC: a squashed-Gaussian MLP actor + ``num_critics`` MLP Q
    critics stacked in one module and their targets, laid out exactly as :func:`_build_sac`."""
    if num_critics < 2:
        raise ValueError(f"SAC needs at least 2 critics (sac.num_critics), got {num_critics}")
    if not actor.get("squash", True):
        raise ValueError("SAC's actor is tanh-squashed: model.actor.squash must be true for trainer.learner: sac")
    critic_space = state_space if state_space is not None else obs_space

    def make_q():
        return MlpQCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            members=num_critics,
            **_kwargs(model_cfg.critic),
        )

    models = {
        "policy": MlpActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **actor,
        ),
    }
    models["critics"] = make_q()
    models["target_critics"] = make_q()
    return models


def _build_mlp_ppo(
    model_cfg: MlpModelCfg,
    obs_space,
    state_space,
    action_space,
    num_agents,
    device,
    actor,
    num_critics: int = 2,
):
    """``architecture: mlp`` for PPO: a Gaussian MLP actor (squashed or plain) + one MLP value
    critic."""
    if num_critics != 2:
        raise ValueError(f"num_critics is a SAC option; PPO builds one value critic (got {num_critics})")
    critic_space = state_space if state_space is not None else obs_space
    return {
        "policy": MlpActor(
            observation_space=obs_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **actor,
        ),
        "value": MlpValueCritic(
            observation_space=critic_space,
            action_space=action_space,
            device=device,
            num_agents=num_agents,
            **_kwargs(model_cfg.critic),
        ),
    }


#: (architecture, learner) -> builder. A new architecture or learner adds its pairs here.
MODEL_BUILDERS: Dict[tuple, Callable[..., Dict[str, Model]]] = {
    ("simba", "sac"): _build_sac,
    ("simba", "ppo"): _build_ppo,
    ("mlp", "sac"): _build_mlp_sac,
    ("mlp", "ppo"): _build_mlp_ppo,
}


def build_models(
    learner: str,
    model_cfg,
    observation_space,
    state_space,
    action_space,
    num_agents: int,
    device,
    controller_cfg=None,
    num_critics: int = 2,
) -> Dict[str, Model]:
    """Build the networks ``learner`` needs, for ``model_cfg``'s architecture.

    ``model_cfg`` is the loaded `model` section; its ``architecture`` field picks the
    builder. ``state_space`` not None means asymmetric actor-critic: the critic consumes
    the state vector while the actor keeps the policy observation. ``controller_cfg`` is the
    loaded `controller` section when there is one: it is where the MATCH pairs and the
    selection axis names come from (see :func:`actor_kwargs`). ``num_critics`` is
    ``sac.num_critics`` for SAC (the Q ensemble's size) and must stay 2 for PPO.
    """
    key = (model_cfg.architecture, learner)
    if key not in MODEL_BUILDERS:
        raise ValueError(
            f"no model builder for architecture '{model_cfg.architecture}' and learner "
            f"'{learner}'; known pairs: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[key](
        model_cfg,
        observation_space,
        state_space,
        action_space,
        num_agents,
        device,
        actor_kwargs(model_cfg, controller_cfg),
        num_critics=num_critics,
    )

"""Model configuration: the `model` section.

Fields are the ones the ported networks read. OmegaConf-supported types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

__all__ = ["ActorCfg", "CriticCfg", "ModelCfg"]


@dataclass
class ActorCfg:
    """Policy network. ``*_action_dims`` index the env-facing action vector."""

    actor_n: int = 2
    actor_latent: int = 512
    act_init_std: float = 0.60653066
    second_act_init_std: Optional[float] = None
    second_act_init_std_dims: Optional[List[int]] = None
    last_layer_scale: float = 1.0
    clip_log_std: bool = True
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    #: how skrl reduces the log-probability density over action dims:
    #: "sum", "mean", "prod" or "none" (checked in ModelCfg.validate; OmegaConf 2.3 cannot
    #: type-check typing.Literal in a structured config)
    reduction: str = "sum"
    use_state_dependent_std: bool = False
    #: action dims drawn from a Bernoulli instead of the squashed Gaussian (e.g. a gripper)
    bernoulli_action_dims: Optional[List[int]] = None
    #: action dims the policy never produces; emitted as 0
    force_zero_action_dims: Optional[List[int]] = None
    #: action dims whose output weights are scaled by ``last_layer_scale`` (default: all)
    scale_down_action_dims: Optional[List[int]] = None


@dataclass
class CriticCfg:
    """Critic network."""

    critic_n: int = 2
    critic_latent: int = 512
    critic_output_init_mean: float = 0.0
    clip_actions: bool = False


@dataclass
class ModelCfg:
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)

    REDUCTIONS = ("sum", "mean", "prod", "none")

    def validate(self, cfg: Any) -> None:
        if self.actor.reduction not in self.REDUCTIONS:
            raise ValueError(
                f"model.actor.reduction must be one of {self.REDUCTIONS}, got "
                f"{self.actor.reduction!r}"
            )
        if self.actor.min_log_std >= self.actor.max_log_std:
            raise ValueError(
                f"model.actor.min_log_std ({self.actor.min_log_std}) must be below "
                f"max_log_std ({self.actor.max_log_std})"
            )

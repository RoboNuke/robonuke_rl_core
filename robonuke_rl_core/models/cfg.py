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
    """Critic network. ``n_atoms`` / ``v_min`` / ``v_max`` are FlashSAC's categorical support."""

    critic_n: int = 2
    critic_latent: int = 512
    critic_output_init_mean: float = 0.0
    clip_actions: bool = False
    n_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0


@dataclass
class ModelCfg:
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)

    def validate(self, cfg: Any) -> None:
        if self.actor.min_log_std >= self.actor.max_log_std:
            raise ValueError(
                f"model.actor.min_log_std ({self.actor.min_log_std}) must be below "
                f"max_log_std ({self.actor.max_log_std})"
            )
        if self.critic.v_min >= self.critic.v_max:
            raise ValueError(
                f"model.critic.v_min ({self.critic.v_min}) must be below v_max ({self.critic.v_max})"
            )
        if self.critic.n_atoms < 2:
            raise ValueError(f"model.critic.n_atoms must be >= 2, got {self.critic.n_atoms}")

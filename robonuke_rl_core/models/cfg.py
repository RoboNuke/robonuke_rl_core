"""Model configuration: the `model` section.

The YAML keys are always ``model.actor.*`` and ``model.critic.*``; **which** dataclasses sit
behind them is picked by ``model.architecture`` (last layer that sets it wins, default
``"simba"``) — the same pattern ``task.name`` uses to pick the env cfg schema. Struct mode
then rejects another architecture's fields, so a SimBa option under a future architecture
fails loudly instead of being silently accepted.

The only architecture shipped today is **SimBa** (Lee et al., 2025,
https://arxiv.org/abs/2410.09754) — residual MLP blocks with LayerNorm, scaled for RL
(``models/simba.py``). A new architecture registers its own ``<Arch>ModelCfg`` with
:func:`register_architecture`; it never mixes extra fields into another architecture's
groups. OmegaConf-supported types only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "SimbaActorCfg",
    "SimbaCriticCfg",
    "SimbaModelCfg",
    "MODEL_ARCHITECTURES",
    "register_architecture",
    "model_cfg_class",
]


@dataclass
class SimbaActorCfg:
    """SimBa policy network. ``*_action_dims`` index the env-facing action vector."""

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
    #: "sum", "mean", "prod" or "none" (checked in SimbaModelCfg.validate; OmegaConf 2.3
    #: cannot type-check typing.Literal in a structured config)
    reduction: str = "sum"
    use_state_dependent_std: bool = False
    #: action dims drawn from a Bernoulli instead of the squashed Gaussian (e.g. a gripper)
    bernoulli_action_dims: Optional[List[int]] = None
    #: action dims the policy never produces; emitted as 0
    force_zero_action_dims: Optional[List[int]] = None
    #: action dims whose output weights are scaled by ``last_layer_scale`` (default: all)
    scale_down_action_dims: Optional[List[int]] = None


@dataclass
class SimbaCriticCfg:
    """SimBa critic network."""

    critic_n: int = 2
    critic_latent: int = 512
    critic_output_init_mean: float = 0.0
    clip_actions: bool = False


@dataclass
class SimbaModelCfg:
    """The `model` section when ``architecture`` is ``simba`` (the default)."""

    architecture: str = "simba"
    actor: SimbaActorCfg = field(default_factory=SimbaActorCfg)
    critic: SimbaCriticCfg = field(default_factory=SimbaCriticCfg)

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


#: registered architectures: ``model.architecture`` value -> the section dataclass
MODEL_ARCHITECTURES: Dict[str, type] = {}


def register_architecture(name: str, cls: type) -> None:
    """Register ``cls`` as the `model` section for ``model.architecture: <name>``.

    ``cls`` must be a dataclass with an ``architecture`` field whose default is ``name``
    (the field is what experiment files set, so the two must agree).
    """
    if name in MODEL_ARCHITECTURES:
        taken = MODEL_ARCHITECTURES[name]
        raise ValueError(
            f"model architecture '{name}' is already registered as "
            f"{taken.__module__}:{taken.__qualname__}"
        )
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"architecture '{name}': {cls!r} is not a @dataclass")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    if "architecture" not in fields or fields["architecture"].default != name:
        raise ValueError(
            f"architecture '{name}': {cls.__qualname__} must have an 'architecture' field "
            f"whose default is {name!r}"
        )
    MODEL_ARCHITECTURES[name] = cls


def model_cfg_class(architecture: str) -> type:
    """The `model` section dataclass for ``architecture``; raises listing what exists."""
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"unknown model.architecture {architecture!r}; registered: "
            f"{sorted(MODEL_ARCHITECTURES)}"
        )
    return MODEL_ARCHITECTURES[architecture]


register_architecture("simba", SimbaModelCfg)

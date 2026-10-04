"""Auxiliary losses: extra terms a learner adds to its policy or critic loss."""

from .cfg import LossesCfg, LossTermCfg
from .losses import (
    LOSSES,
    TARGETS,
    AuxLoss,
    LossContext,
    build_aux_losses,
    register_loss,
)

__all__ = [
    "LOSSES",
    "TARGETS",
    "AuxLoss",
    "LossContext",
    "LossTermCfg",
    "LossesCfg",
    "build_aux_losses",
    "register_loss",
]

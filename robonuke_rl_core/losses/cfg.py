"""Losses configuration: the `losses` section.

A list of terms, so adding a loss needs no new config fields:

    losses:
      terms:
        - name: action_l2
          target: policy
          weight: 0.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from omegaconf import MISSING

__all__ = ["LossTermCfg", "LossesCfg"]


@dataclass
class LossTermCfg:
    """One auxiliary loss term: which loss, which optimizer it feeds, and how strongly."""

    #: a registered loss name (see losses.LOSSES)
    name: str = MISSING
    #: "policy" or "critic"; the loss must support it
    target: str = MISSING
    weight: float = MISSING
    #: constructor arguments for the loss class
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossesCfg:
    terms: List[LossTermCfg] = field(default_factory=list)

    def validate(self, cfg: Any) -> None:
        from .losses import TARGETS

        for index, term in enumerate(self.terms):
            if term.target not in TARGETS:
                raise ValueError(
                    f"losses.terms[{index}].target must be one of {TARGETS}, got {term.target!r}"
                )
            if not term.name:
                raise ValueError(f"losses.terms[{index}].name is empty")

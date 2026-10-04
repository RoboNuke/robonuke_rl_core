"""Auxiliary losses: extra terms added to a learner's policy or critic loss.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/losses.py``, with the
config turned into a list of terms, so adding a loss needs no new config fields.

A loss returns **one raw value per agent**, shape ``(num_agents,)``; never average across
agents inside ``compute``. :func:`build_aux_losses` reduces it with a fixed ``1/num_agents``
weight, so each agent's gradient depends only on its own rows.

The learners know nothing about this module beyond the ``LossContext`` they build and the
callable they put in their ``aux_loss`` hook list.

The package ships **no** built-in loss: a penalty on action magnitude belongs in the env's
reward, not in a policy-side term. Projects register their own with ``@register_loss``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:  # the learners import from here, not the other way round
    from ..learners.base import LearnerBase
    from .cfg import LossesCfg

__all__ = [
    "TARGETS",
    "LossContext",
    "AuxLoss",
    "LOSSES",
    "register_loss",
    "build_aux_losses",
]

#: the optimizers an auxiliary loss may feed
TARGETS = ("policy", "critic")


@dataclasses.dataclass
class LossContext:
    """Read-only bundle of what a loss may need, filled in at the call site.

    A learner builds this twice per update — once for the critic block and once for the
    policy block — so only the fields relevant to ``target`` are set. A loss should read
    only the fields valid for the targets it declares.
    """

    learner: "LearnerBase"
    target: str
    step: int = 0

    #: the sampled minibatch (observations, actions, rewards, ... as the learner stores them)
    sampled: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    # ---- policy-block fields (set when target == "policy") ----
    actions: Optional[torch.Tensor] = None
    log_prob: Optional[torch.Tensor] = None
    policy_outputs: Optional[Dict[str, torch.Tensor]] = None
    inputs: Optional[Dict[str, torch.Tensor]] = None

    # ---- critic-block fields (set when target == "critic") ----
    critic_1_values: Optional[torch.Tensor] = None
    critic_2_values: Optional[torch.Tensor] = None
    target_values: Optional[torch.Tensor] = None
    critic_inputs: Optional[Dict[str, torch.Tensor]] = None


class AuxLoss:
    """Base class for an auxiliary loss. Constructed with the term's ``kwargs``."""

    name: str = ""
    supported_targets: Tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} takes no kwargs, got {sorted(kwargs)}; remove them from "
                f"the losses.terms entry for '{self.name}'"
            )

    def compute(self, ctx: LossContext) -> torch.Tensor:
        """Return the raw, unweighted loss per agent as a ``(num_agents,)`` tensor.

        Reshape a flat ``(num_agents * rows, ...)`` batch tensor with
        ``.view(ctx.learner.num_agents, -1)`` before reducing, so agent ``i``'s value comes
        from agent ``i``'s rows only.
        """
        raise NotImplementedError


#: registered losses, keyed by ``AuxLoss.name``
LOSSES: Dict[str, type] = {}


def register_loss(cls: type) -> type:
    """Class decorator: register an :class:`AuxLoss` subclass under its ``name``."""
    if not issubclass(cls, AuxLoss):
        raise TypeError(f"{cls.__name__} must subclass AuxLoss")
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    if cls.name in LOSSES:
        taken = LOSSES[cls.name]
        raise ValueError(
            f"loss name {cls.name!r} is already registered as {taken.__module__}:"
            f"{taken.__qualname__}; refusing to replace it with {cls.__qualname__}"
        )
    unknown = set(cls.supported_targets) - set(TARGETS)
    if unknown or not cls.supported_targets:
        raise ValueError(
            f"{cls.__name__}.supported_targets must be a non-empty subset of {TARGETS}, got "
            f"{cls.supported_targets!r}"
        )
    LOSSES[cls.name] = cls
    return cls


def build_aux_losses(losses_cfg: "LossesCfg", num_agents: int) -> Optional[Callable]:
    """Build the callable the learners put in ``aux_loss``, or None when no term is configured.

    The callable returns ``Σ weight * raw.mean()`` for the context's target — ``raw.mean()``
    is a fixed ``1/num_agents`` per agent, so one agent's value never scales another's
    gradient — and emits each term's per-agent raw values as ``loss/<name>_<target>``.
    """
    terms = list(losses_cfg.terms)
    if not terms:
        return None

    built = []
    for index, term in enumerate(terms):
        if term.name not in LOSSES:
            raise ValueError(
                f"losses.terms[{index}]: unknown loss {term.name!r}; registered: "
                f"{sorted(LOSSES) or '<none>'}. Projects register their own losses with "
                "@register_loss before loading the config."
            )
        loss_cls = LOSSES[term.name]
        if term.target not in loss_cls.supported_targets:
            raise ValueError(
                f"losses.terms[{index}]: loss {term.name!r} does not support target "
                f"{term.target!r} (supported: {loss_cls.supported_targets})"
            )
        built.append((term, loss_cls(**dict(term.kwargs))))

    checked: set = set()

    def aux_loss(ctx: LossContext) -> Optional[torch.Tensor]:
        total = None
        raw_values: Dict[str, torch.Tensor] = {}
        for term, loss in built:
            if term.target != ctx.target:
                continue
            raw = loss.compute(ctx)
            if term.name not in checked:
                if not torch.is_tensor(raw) or raw.shape != (num_agents,):
                    raise TypeError(
                        f"loss {term.name!r}.compute() must return a tensor of shape "
                        f"({num_agents},), one raw value per agent, got "
                        f"{tuple(raw.shape) if torch.is_tensor(raw) else type(raw).__name__}"
                    )
                checked.add(term.name)
            total = term.weight * raw.mean() if total is None else total + term.weight * raw.mean()
            raw_values[f"loss/{term.name}_{term.target}"] = raw.detach()
        if raw_values:
            ctx.learner.emit_per_agent(raw_values, ctx.step)
        return total

    return aux_loss

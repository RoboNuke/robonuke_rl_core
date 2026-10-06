"""Auxiliary losses: extra terms added to a learner's policy or critic loss.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``learning/losses.py``, with the
config turned into a list of terms, so adding a loss needs no new config fields.

A loss returns **one raw value per agent**, shape ``(num_agents,)``; never average across
agents inside ``compute``. :func:`build_aux_losses` reduces it with a fixed ``1/num_agents``
weight, so each agent's gradient depends only on its own rows.

The learners know nothing about this module beyond the ``LossContext`` they build and the
callable they put in their ``aux_loss`` hook list.

Projects register their own losses with ``@register_loss``. The package ships exactly one,
:class:`SupervisedSelectionLoss`, and the bar it had to clear is worth stating: a term that
could be written as a reward (a penalty on action magnitude, say) belongs in the env's
reward, not here. This one cannot — it supervises a *head of the policy network* against a
per-transition fact the env knows, so it has to reach inside the model's outputs.

A loss may also declare :meth:`AuxLoss.required_memory_keys`: per-transition tensors it
needs from the replay batch. :func:`build_aux_losses` merges them onto the returned hook as
``memory_keys``, the learner creates them in ``_create_memory_tensors`` and fills them from
``infos`` in ``record_transition``, and they arrive back in ``ctx.sampled``.
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
    "SupervisedSelectionLoss",
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

    def required_memory_keys(self) -> Dict[str, int]:
        """Per-transition tensors this loss needs from the replay batch: name -> width.

        The learner creates each as a float tensor of that width, fills it from
        ``infos[name]`` every step, and the sampled batch carries it in ``ctx.sampled``. An
        empty dict (the default) means the loss works from what the learner already stores.
        """
        return {}

    def compute(self, ctx: LossContext) -> torch.Tensor:
        """Return the raw, unweighted loss per agent as a ``(num_agents,)`` tensor.

        Reshape a flat ``(num_agents * rows, ...)`` batch tensor with
        ``.view(ctx.learner.num_agents, -1)`` before reducing, so agent ``i``'s value comes
        from agent ``i``'s rows only.
        """
        raise NotImplementedError

    def sampled(self, ctx: LossContext, key: str) -> torch.Tensor:
        """``ctx.sampled[key]``, with the error that names how to make it appear."""
        value = ctx.sampled.get(key)
        if value is None:
            raise RuntimeError(
                f"loss {self.name!r} needs '{key}' in the sampled batch but the learner did "
                f"not store it. It is declared in {type(self).__name__}."
                "required_memory_keys, so the learner creates it only if the aux-loss hook "
                "is in learner.aux_loss before the trainer calls init(), and only if the "
                f"env publishes infos['{key}'] every step."
            )
        return value


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


def _merge_memory_keys(built) -> Dict[str, int]:
    """Union of every term's ``required_memory_keys``; two widths for one name is an error."""
    merged: Dict[str, int] = {}
    for term, loss in built:
        for key, width in loss.required_memory_keys().items():
            if key in merged and merged[key] != int(width):
                raise ValueError(
                    f"two losses need memory key '{key}' at different widths: "
                    f"{merged[key]} and {int(width)}. One transition tensor cannot be both."
                )
            merged[key] = int(width)
    return merged


def build_aux_losses(losses_cfg: "LossesCfg", num_agents: int) -> Optional[Callable]:
    """Build the callable the learners put in ``aux_loss``, or None when no term is configured.

    The callable returns ``Σ weight * raw.mean()`` for the context's target — ``raw.mean()``
    is a fixed ``1/num_agents`` per agent, so one agent's value never scales another's
    gradient — and emits each term's per-agent raw values as ``loss/<name>_<target>``.

    It also carries ``memory_keys``: the union of the terms' ``required_memory_keys``, which
    the learner reads in ``_create_memory_tensors``. Append the hook to ``learner.aux_loss``
    **before** the trainer calls ``init()``, or those tensors are never created.
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

    aux_loss.memory_keys = _merge_memory_keys(built)
    return aux_loss


# --------------------------------------------------------------------------- built-in losses
@register_loss
class SupervisedSelectionLoss(AuxLoss):
    """Teach the hybrid selection head to pick force control exactly in contact.

    Binary cross entropy between the actor's per-axis probability of **force** control
    (``policy_outputs["selection_prob"]`` — the selection bit is 1 for force, see
    ``models/simba.SELECTION_BIT_IS_FORCE``) and whether that axis is in contact
    (``infos["in_contact"]``, published by the contact-sensor wrapper). Force control on an
    axis that is touching nothing has no force to regulate, and impedance control on an axis
    that is pressing is what breaks a fragile peg, so the target is the contact flag itself.

    It is a *supervision* term, not a reward: it names which head output should be what,
    which a scalar reward cannot express. The policy is still free to disagree — the weight
    decides how strongly — and it is the only reason this one ships with the package.

    Per agent, the value is the mean BCE over that agent's rows and axes; the fixed
    ``1/num_agents`` weight is applied by :func:`build_aux_losses`, never here.
    """

    name = "supervised_selection"
    supported_targets = ("policy",)
    #: the probability is clamped to this before the log, so a saturated head is finite
    EPS = 1.0e-7
    #: the per-transition contact flags, one per selection axis
    MEMORY_KEY = "in_contact"

    def __init__(self, *, num_axes: int) -> None:
        """``num_axes`` is the number of selection dims — the width of the contact flags."""
        if int(num_axes) < 1:
            raise ValueError(
                f"supervised_selection: num_axes must be >= 1, got {num_axes}. It is the "
                "number of the controller's force-eligible axes (sum of controller.force_axes)."
            )
        self.num_axes = int(num_axes)

    def required_memory_keys(self) -> Dict[str, int]:
        return {self.MEMORY_KEY: self.num_axes}

    def compute(self, ctx: LossContext) -> torch.Tensor:
        outputs = ctx.policy_outputs or {}
        probability = outputs.get("selection_prob")
        if probability is None:
            raise RuntimeError(
                "loss 'supervised_selection' needs the actor's selection_prob, which only a "
                "policy with selection dims produces. Set model.actor.bernoulli_action_dims "
                "to the controller's selection block (controller.use_pose and "
                "controller.use_force both true)."
            )
        target = self.sampled(ctx, self.MEMORY_KEY)
        if probability.shape != target.shape:
            raise ValueError(
                f"supervised_selection: the actor produced {tuple(probability.shape)} "
                f"selection probabilities but infos['{self.MEMORY_KEY}'] gave "
                f"{tuple(target.shape)}. num_axes={self.num_axes} must be the number of "
                "selection dims, and the contact wrapper must publish one flag per one."
            )
        probability = probability.clamp(self.EPS, 1.0 - self.EPS)
        target = target.to(probability.dtype)
        per_row = -(
            target * torch.log(probability) + (1.0 - target) * torch.log(1.0 - probability)
        )
        # per agent: the mean over ITS rows and axes. A mean over the whole batch would let
        # one agent's contacts scale another agent's gradient.
        return per_row.view(ctx.learner.num_agents, -1).mean(dim=-1)

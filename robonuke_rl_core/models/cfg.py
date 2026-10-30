"""Model configuration: the `model` section.

The YAML keys are always ``model.actor.*`` and ``model.critic.*``; **which** dataclasses sit
behind them is picked by ``model.architecture`` (last layer that sets it wins, default
``"simba"``) — the same pattern ``task.name`` uses to pick the env cfg schema. Struct mode
then rejects another architecture's fields, so a SimBa option under a future architecture
fails loudly instead of being silently accepted.

Two architectures ship: **SimBa** (Lee et al., 2025, https://arxiv.org/abs/2410.09754) —
residual MLP blocks with LayerNorm, scaled for RL (``models/simba.py``), the default — and a
plain **MLP** (``models/mlp.py``: Linear -> activation stacks, the nets ResiP and Policy
Decorator train). A new architecture registers its own ``<Arch>ModelCfg`` with
:func:`register_architecture`; it never mixes extra fields into another architecture's
groups. OmegaConf-supported types only.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "SimbaActorCfg",
    "SimbaCriticCfg",
    "SimbaModelCfg",
    "MlpActorCfg",
    "MlpCriticCfg",
    "MlpModelCfg",
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
    #: how the selection dims and the continuous dims form one joint distribution:
    #: "product" (independent: one density times the Bernoullis) or "match" (conditional:
    #: each selection dim picks which of its two continuous components is live). Checked in
    #: SimbaModelCfg.validate; "match" needs the controller's (pose, force) pairs, which the
    #: model factory derives from the action layout.
    selection_distribution: str = "product"
    #: added to the selection logits' output bias at init, so training starts biased toward
    #: one branch: sigmoid(-2.2) ~= 0.1, i.e. ~90% position at the first step
    selection_init_bias: float = 0.0


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

        from .simba import SELECTION_DISTRIBUTIONS

        if self.actor.selection_distribution not in SELECTION_DISTRIBUTIONS:
            raise ValueError(
                "model.actor.selection_distribution must be one of "
                f"{SELECTION_DISTRIBUTIONS}, got {self.actor.selection_distribution!r}"
            )
        selection = self.actor.bernoulli_action_dims or []
        if self.actor.selection_distribution == "match" and not selection:
            raise ValueError(
                "model.actor.selection_distribution: 'match' conditions the continuous "
                "density on the selection dims, so model.actor.bernoulli_action_dims must "
                "name them; got none"
            )
        if self.actor.selection_distribution == "match":
            self._check_match_has_a_controller(cfg)
        if self.actor.selection_init_bias and not selection:
            raise ValueError(
                "model.actor.selection_init_bias "
                f"({self.actor.selection_init_bias}) biases the selection logits, but "
                "model.actor.bernoulli_action_dims names no selection dims"
            )

    def _check_match_has_a_controller(self, cfg: Any) -> None:
        """``match`` gates (pose, force) pairs, and only the controller can say which.

        The pairs are derived from ``controller.force_axes`` and the action layout (see
        ``models/factory.actor_kwargs``), so ``match`` without a controller that has a
        selection block has nothing to condition on. The actor would raise when it is built;
        this says the same thing at config load, naming the field to change.
        """
        controller = getattr(cfg, "controller", None) if cfg is not None else None
        if controller is None:
            return  # no controller section registered: nothing to check against
        from ..envs.interface import ActionLayout

        if not getattr(controller, "enabled", False) or not ActionLayout(controller).selection_dim:
            raise ValueError(
                "model.actor.selection_distribution: 'match' gates each selection dim's "
                "(pose, force) pair, which is derived from the controller's action layout. "
                "It needs controller.enabled, controller.use_pose and controller.use_force "
                "all true (that is the only case with a selection block)."
            )


@dataclass
class MlpActorCfg:
    """Plain MLP policy (``models/mlp.py``): Linear -> activation stacks and a Gaussian head."""

    #: width of every hidden layer
    hidden_size: int = 256
    #: hidden layers (ResiP: 2 x 512 SiLU; Policy Decorator: 3 x 256 ReLU)
    num_layers: int = 2
    #: a ``torch.nn`` activation class name: ReLU, SiLU, Tanh, ELU, GELU, ...
    activation: str = "ReLU"
    #: hidden-layer init: "orthogonal" (gain sqrt(2)) or "kaiming" (fan-in normal, ResiP) -- the
    #: output layers orthogonal -- or "xavier" (SERL: xavier-uniform on every layer); biases zero
    init: str = "orthogonal"
    #: the gain of the mean (and state-dependent log std) head's init (ResiP and Policy Decorator:
    #: 0.01, a near-zero initial action; SERL: 1.0)
    output_init_scale: float = 0.01
    #: the log std every dim starts at (ResiP's residual: -1.0)
    init_log_std: float = 0.0
    #: false freezes the state-independent log std at ``init_log_std`` (ResiP): a buffer, not
    #: a parameter, so no optimizer ever touches it. Needs ``use_state_dependent_std: false``
    learn_std: bool = True
    #: a second head predicts the log std from the state (classic SAC) instead of one learned
    #: value per dim
    use_state_dependent_std: bool = False
    #: how the log std is kept in [min_log_std, max_log_std]: "clip" (hard clamp, SimBa's
    #: ``clip_log_std``), "tanh" (soft: ``min + 0.5 (max - min) (tanh(x) + 1)``, Policy
    #: Decorator) or "none"
    log_std_bound: str = "clip"
    min_log_std: float = -20.0
    max_log_std: float = 2.0
    #: tanh-squash the Gaussian (SAC; actions in [-1, 1]). false is a plain Gaussian (ResiP's
    #: residual): PPO, or SAC with ``sac.unsquashed_actor``
    squash: bool = True
    #: a LayerNorm between every hidden layer's Linear and its activation (SERL's ``use_layer_norm``)
    layer_norm: bool = False


@dataclass
class MlpCriticCfg:
    """Plain MLP critic(s): Q(s, a) for SAC, V(s) for PPO."""

    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "ReLU"
    #: "orthogonal", "kaiming" or "xavier", as the actor's
    init: str = "orthogonal"
    #: the gain of the value head's init (ResiP: 1.0)
    output_init_scale: float = 1.0
    #: a LayerNorm between every hidden layer's Linear and its activation (SERL's ``use_layer_norm``)
    layer_norm: bool = False


@dataclass
class MlpModelCfg:
    """The `model` section when ``architecture`` is ``mlp``."""

    architecture: str = "mlp"
    actor: MlpActorCfg = field(default_factory=MlpActorCfg)
    critic: MlpCriticCfg = field(default_factory=MlpCriticCfg)

    def validate(self, cfg: Any) -> None:
        from .mlp import check_mlp_actor_fields, check_mlp_critic_fields

        check_mlp_actor_fields("model.actor", **dataclasses.asdict(self.actor))
        critic = self.critic
        check_mlp_critic_fields(
            "model.critic", critic.hidden_size, critic.num_layers, critic.activation, critic.init,
            critic.output_init_scale,
        )
        trainer = getattr(cfg, "trainer", None) if cfg is not None else None
        sac = getattr(cfg, "sac", None) if cfg is not None else None
        unsquashed_ok = bool(getattr(sac, "unsquashed_actor", False))
        if trainer is not None and getattr(trainer, "learner", None) == "sac" and not self.actor.squash and not unsquashed_ok:
            raise ValueError(
                "model.actor.squash: false is a plain Gaussian; SAC's actor is tanh-squashed (its "
                "actions and log-probabilities assume [-1, 1]), so trainer.learner: sac needs "
                "model.actor.squash: true, or sac.unsquashed_actor: true to train the plain Gaussian"
            )
        if unsquashed_ok and self.actor.squash:
            raise ValueError("sac.unsquashed_actor: true trains a plain Gaussian; set model.actor.squash: false")


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


@dataclass
class FlashSACActorCfg:
    """FlashSAC's actor (``models/flashsac.py``; arXiv:2604.04539 Table 9): 2 blocks of hidden 128."""

    hidden_size: int = 128
    num_blocks: int = 2
    #: the inverted residual block's expansion of the hidden width
    expansion: int = 4
    #: the log std's tanh bound, ``min + 0.5 (max - min) (1 + tanh(raw))``
    min_log_std: float = -10.0
    max_log_std: float = 2.0
    #: the mean head's orthogonal init gain. 1.0 is FlashSAC's (unit rows, kept so by weight normalization);
    #: below 1 (a residual's near-zero start, Policy Decorator's 0.01) the head starts that small and is left
    #: out of weight normalization, which would set its rows back to unit length
    mean_init_scale: float = 1.0
    #: the std every state starts at (None: FlashSAC's own init, the log std head's unit rows, so the initial std
    #: varies by state over most of the bound). Set, the state-dependent log std head's weights start at gain 0.01
    #: (left out of weight normalization) and its bias at the value the tanh bound maps to ``log(init_std)``; the
    #: head then learns its state dependence from there
    init_std: Optional[float] = None
    #: a constant std for every state and dim, no std head (ResiP's frozen exploration, e.g. exp(-1) = 0.368). None:
    #: the state-dependent head. Not with ``init_std``
    fixed_std: Optional[float] = None


@dataclass
class FlashSACCriticCfg:
    """FlashSAC's critic (Table 9): 2 blocks of hidden 256; its output width is the learner's
    (``sac.critic_distribution``)."""

    hidden_size: int = 256
    num_blocks: int = 2
    expansion: int = 4


@dataclass
class FlashSACModelCfg:
    """The `model` section when ``architecture`` is ``flashsac``: FlashSAC's BatchNorm inverted-residual
    network. SAC only."""

    architecture: str = "flashsac"
    actor: FlashSACActorCfg = field(default_factory=FlashSACActorCfg)
    critic: FlashSACCriticCfg = field(default_factory=FlashSACCriticCfg)

    def validate(self, cfg: Any) -> None:
        for name, section in (("actor", self.actor), ("critic", self.critic)):
            if section.hidden_size < 1 or section.num_blocks < 0 or section.expansion < 1:
                raise ValueError(
                    f"model.{name}: hidden_size >= 1, num_blocks >= 0 and expansion >= 1, got "
                    f"{section.hidden_size}, {section.num_blocks}, {section.expansion}"
                )
        if self.actor.fixed_std is not None:
            if self.actor.init_std is not None:
                raise ValueError("model.actor.fixed_std freezes the std; init_std starts a learned one: set one")
            if self.actor.fixed_std <= 0:
                raise ValueError(f"model.actor.fixed_std must be > 0, got {self.actor.fixed_std}")
        if self.actor.init_std is not None and not (
            math.exp(self.actor.min_log_std) < self.actor.init_std < math.exp(self.actor.max_log_std)
        ):
            raise ValueError(
                f"model.actor.init_std ({self.actor.init_std}) must lie strictly inside the log std bound "
                f"[exp({self.actor.min_log_std}), exp({self.actor.max_log_std})]"
            )
        if self.actor.mean_init_scale <= 0:
            raise ValueError(f"model.actor.mean_init_scale must be > 0, got {self.actor.mean_init_scale}")
        if not self.actor.min_log_std < self.actor.max_log_std:
            raise ValueError(f"model.actor.min_log_std ({self.actor.min_log_std}) must be below max_log_std ({self.actor.max_log_std})")
        trainer = getattr(cfg, "trainer", None) if cfg is not None else None
        if trainer is not None and getattr(trainer, "learner", None) not in (None, "sac"):
            raise ValueError(f"model.architecture: flashsac is a SAC network; trainer.learner is {trainer.learner!r}")


register_architecture("simba", SimbaModelCfg)
register_architecture("mlp", MlpModelCfg)
register_architecture("flashsac", FlashSACModelCfg)

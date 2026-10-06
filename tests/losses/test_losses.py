"""Auxiliary losses: the weighted total, per-agent reporting, independence, and errors."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
import torch
from omegaconf import OmegaConf

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import dump, load_config, load_from_run
from robonuke_rl_core.losses import (
    LOSSES,
    AuxLoss,
    LossContext,
    LossesCfg,
    LossTermCfg,
    build_aux_losses,
    register_loss,
)

NUM_AGENTS = 3
ROWS = 4
ACT_DIM = 2


class FakeLearner:
    """What a LossContext exposes to a loss: the agent count, a device, and logging."""

    def __init__(self, num_agents: int = NUM_AGENTS):
        self.num_agents = num_agents
        self.device = torch.device("cpu")
        self.logged: list[tuple[int, dict, int]] = []

    def emit_per_agent(self, metrics: dict, step: int) -> None:
        for agent in range(self.num_agents):
            self.logged.append((agent, {k: v[agent] for k, v in metrics.items()}, step))


class ActionSquaredLoss(AuxLoss):
    """A project-style loss: mean squared action magnitude, one value per agent.

    An action penalty belongs in the env's reward, so the package does not ship this one;
    the tests register it themselves, which is also how a project does it.
    """

    name = "action_squared"
    supported_targets = ("policy",)

    def compute(self, ctx):
        if ctx.actions is None:
            raise ValueError("action_squared needs ctx.actions, which the policy block sets")
        return ctx.actions.pow(2).view(ctx.learner.num_agents, -1).mean(dim=1)


@pytest.fixture(autouse=True)
def registry():
    """Register the test loss, and undo anything a test registers."""
    known = dict(LOSSES)
    register_loss(ActionSquaredLoss)
    yield
    LOSSES.clear()
    LOSSES.update(known)


def policy_ctx(learner, actions: torch.Tensor, step: int = 3) -> LossContext:
    return LossContext(learner=learner, target="policy", step=step, actions=actions)


# ------------------------------------------------------------------ 7. weighted total
def test_the_total_is_the_sum_of_weight_times_mean_raw():
    learner = FakeLearner()
    actions = torch.randn(NUM_AGENTS * ROWS, ACT_DIM)
    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="action_squared", target="policy", weight=0.25)]),
        NUM_AGENTS,
    )
    ctx = policy_ctx(learner, actions)
    total = aux(ctx)

    raw = ActionSquaredLoss().compute(ctx)
    assert raw.shape == (NUM_AGENTS,)
    assert float(total) == pytest.approx(float(0.25 * raw.mean()))

    # the per-agent raw values were reported, unweighted
    assert [agent for agent, _, _ in learner.logged] == list(range(NUM_AGENTS))
    for agent, metrics, step in learner.logged:
        assert step == 3
        assert float(metrics["loss/action_squared_policy"]) == pytest.approx(float(raw[agent]))


def test_terms_for_another_target_do_not_contribute():
    learner = FakeLearner()
    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="action_squared", target="policy", weight=1.0)]),
        NUM_AGENTS,
    )
    critic_ctx = LossContext(learner=learner, target="critic", sampled={})
    assert aux(critic_ctx) is None
    assert learner.logged == []


def test_no_terms_means_no_hook():
    assert build_aux_losses(LossesCfg(), NUM_AGENTS) is None


def test_several_terms_add_up():
    @register_loss
    class ConstantLoss(AuxLoss):
        name = "constant"
        supported_targets = ("policy", "critic")

        def __init__(self, value: float = 1.0):
            self.value = float(value)

        def compute(self, ctx):
            return torch.full((ctx.learner.num_agents,), self.value)

    learner = FakeLearner()
    aux = build_aux_losses(
        LossesCfg(
            terms=[
                LossTermCfg(name="constant", target="policy", weight=2.0, kwargs={"value": 3.0}),
                LossTermCfg(name="constant", target="policy", weight=0.5, kwargs={"value": 1.0}),
            ]
        ),
        NUM_AGENTS,
    )
    total = aux(policy_ctx(learner, torch.zeros(NUM_AGENTS * ROWS, ACT_DIM)))
    assert float(total) == pytest.approx(2.0 * 3.0 + 0.5 * 1.0)


def test_kwargs_reach_the_loss_and_a_bad_one_raises():
    with pytest.raises(TypeError) as err:
        build_aux_losses(
            LossesCfg(
                terms=[
                    LossTermCfg(name="action_squared", target="policy", weight=1.0, kwargs={"x": 1})
                ]
            ),
            NUM_AGENTS,
        )
    assert "action_squared" in str(err.value)


# ------------------------------------------------------------------ 8. independence
def test_one_agents_actions_do_not_change_another_agents_raw_value():
    learner = FakeLearner()
    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="action_squared", target="policy", weight=1.0)]),
        NUM_AGENTS,
    )
    torch.manual_seed(0)
    actions = torch.randn(NUM_AGENTS * ROWS, ACT_DIM)
    extreme = actions.clone()
    extreme[ROWS : 2 * ROWS] *= 1.0e3  # agent 1 only

    aux(policy_ctx(learner, actions))
    plain = {agent: metrics["loss/action_squared_policy"] for agent, metrics, _ in learner.logged}
    learner.logged.clear()
    aux(policy_ctx(learner, extreme))
    changed = {agent: metrics["loss/action_squared_policy"] for agent, metrics, _ in learner.logged}

    for agent in (0, 2):
        assert torch.equal(plain[agent], changed[agent])
    assert not torch.equal(plain[1], changed[1])


def test_the_gradient_of_one_agents_slice_comes_only_from_its_rows():
    learner = FakeLearner()
    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="action_squared", target="policy", weight=1.0)]),
        NUM_AGENTS,
    )

    def grads(scale_agent_1: float) -> torch.Tensor:
        actions = torch.ones(NUM_AGENTS * ROWS, ACT_DIM, requires_grad=True)
        scaled = actions.clone()
        scaled[ROWS : 2 * ROWS] = scaled[ROWS : 2 * ROWS] * scale_agent_1
        aux(policy_ctx(learner, scaled)).backward()
        return actions.grad.clone()

    plain, changed = grads(1.0), grads(1000.0)
    for agent in (0, 2):
        rows = slice(agent * ROWS, (agent + 1) * ROWS)
        assert torch.equal(plain[rows], changed[rows])


# ------------------------------------------------------------------ 9. errors
def test_an_unknown_loss_name_raises():
    with pytest.raises(ValueError) as err:
        build_aux_losses(
            LossesCfg(terms=[LossTermCfg(name="nope", target="policy", weight=1.0)]), NUM_AGENTS
        )
    message = str(err.value)
    assert "nope" in message and "action_squared" in message


def test_an_unsupported_target_raises():
    with pytest.raises(ValueError) as err:
        build_aux_losses(
            LossesCfg(terms=[LossTermCfg(name="action_squared", target="critic", weight=1.0)]),
            NUM_AGENTS,
        )
    message = str(err.value)
    assert "action_squared" in message and "critic" in message


def test_a_raw_value_of_the_wrong_shape_raises():
    @register_loss
    class BadShapeLoss(AuxLoss):
        name = "bad_shape"
        supported_targets = ("policy",)

        def compute(self, ctx):
            return torch.zeros(ctx.learner.num_agents + 1)  # not one value per agent

    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="bad_shape", target="policy", weight=1.0)]),
        NUM_AGENTS,
    )
    with pytest.raises(TypeError) as err:
        aux(policy_ctx(FakeLearner(), torch.zeros(NUM_AGENTS * ROWS, ACT_DIM)))
    assert "bad_shape" in str(err.value)
    assert f"({NUM_AGENTS},)" in str(err.value)


def test_a_scalar_raw_value_raises():
    @register_loss
    class ScalarLoss(AuxLoss):
        name = "scalar"
        supported_targets = ("policy",)

        def compute(self, ctx):
            return ctx.actions.pow(2).mean()  # averaged across agents: not allowed

    aux = build_aux_losses(
        LossesCfg(terms=[LossTermCfg(name="scalar", target="policy", weight=1.0)]), NUM_AGENTS
    )
    with pytest.raises(TypeError):
        aux(policy_ctx(FakeLearner(), torch.zeros(NUM_AGENTS * ROWS, ACT_DIM)))


def test_duplicate_registration_raises():
    with pytest.raises(ValueError) as err:

        @register_loss
        class Duplicate(AuxLoss):
            name = "action_squared"
            supported_targets = ("policy",)

            def compute(self, ctx):
                return torch.zeros(ctx.learner.num_agents)

    assert "action_squared" in str(err.value)


def test_a_loss_must_declare_a_name_and_valid_targets():
    with pytest.raises(ValueError):

        @register_loss
        class NoName(AuxLoss):
            supported_targets = ("policy",)

            def compute(self, ctx):
                return torch.zeros(ctx.learner.num_agents)

    with pytest.raises(ValueError):

        @register_loss
        class BadTarget(AuxLoss):
            name = "bad_target"
            supported_targets = ("value",)

            def compute(self, ctx):
                return torch.zeros(ctx.learner.num_agents)

    with pytest.raises(TypeError):

        @register_loss
        class NotALoss:
            name = "not_a_loss"
            supported_targets = ("policy",)


def test_the_only_built_in_loss_is_the_selection_supervision():
    """The bar for shipping a loss: it must read a model output, not an env quantity.

    An action-magnitude penalty is expressible as a reward, so it stays in the env (the
    fixture's ``action_squared`` is a stand-in for a project's own). The supervised
    selection loss is not: it names what a head of the network should output.
    """
    known = dict(LOSSES)
    known.pop("action_squared")  # the fixture's own
    assert sorted(known) == ["supervised_selection"]


def test_a_loss_reading_a_field_its_target_does_not_set_raises():
    with pytest.raises(ValueError) as err:
        ActionSquaredLoss().compute(LossContext(learner=FakeLearner(), target="policy"))
    assert "ctx.actions" in str(err.value)


# ------------------------------------------------------------------ 10. config
@dataclass
class FakeScene:
    num_envs: int = 8


@dataclass
class FakeEnvCfg:
    decimation: int = 8
    seed: Optional[int] = None
    scene: FakeScene = field(default_factory=FakeScene)


@pytest.fixture
def fake_task(monkeypatch):
    monkeypatch.setattr(cfgmod, "load_task_cfg", lambda name: FakeEnvCfg())
    monkeypatch.setattr(
        cfgmod, "task_cfg_to_dict", lambda cfg: dataclasses.asdict(copy.deepcopy(cfg))
    )

    def apply(env_cfg, data):
        for key, value in data.items():
            current = getattr(env_cfg, key)
            if dataclasses.is_dataclass(current):
                apply(current, value)
            else:
                setattr(env_cfg, key, value)
        return env_cfg

    monkeypatch.setattr(cfgmod, "apply_task_cfg", apply)


BASE = """
task:
  name: Fake-Task-v0
experiment:
  seed: 1
  num_agents: 2
wandb:
  entity: hur
  project: p
  group: g
trainer:
  learner: sac
  total_timesteps: 100
"""


def test_terms_load_from_yaml_and_the_cli(tmp_path, fake_task):
    path = tmp_path / "exp.yaml"
    path.write_text(
        BASE
        + """
losses:
  terms:
    - name: action_squared
      target: policy
      weight: 0.1
"""
    )
    cfg = load_config(path)
    assert len(cfg.losses.terms) == 1
    term = cfg.losses.terms[0]
    assert (term.name, term.target, term.weight, term.kwargs) == ("action_squared", "policy", 0.1, {})
    assert build_aux_losses(cfg.losses, cfg.experiment.num_agents) is not None

    # and the whole list can come from the CLI
    cli = load_config(
        tmp_path / "exp.yaml",
        ['losses.terms=[{name: action_squared, target: policy, weight: 0.5}]'],
    )
    assert len(cli.losses.terms) == 1
    assert cli.losses.terms[0].weight == 0.5

    # it round trips through the dump
    first = dump(cfg, tmp_path / "run_a", cfg.task_cfg)
    second = dump(load_from_run(tmp_path / "run_a"), tmp_path / "run_b", cfg.task_cfg)
    a = OmegaConf.to_container(OmegaConf.load(first))
    b = OmegaConf.to_container(OmegaConf.load(second))
    a.pop("meta"), b.pop("meta")
    assert a == b
    assert a["losses"]["terms"][0]["name"] == "action_squared"


def test_a_bad_target_in_the_config_raises(tmp_path, fake_task):
    path = tmp_path / "exp.yaml"
    path.write_text(BASE)
    with pytest.raises(ValueError) as err:
        load_config(path, ['losses.terms=[{name: action_squared, target: value, weight: 0.1}]'])
    assert "target" in str(err.value)


def test_an_unknown_field_in_a_term_raises(tmp_path, fake_task):
    path = tmp_path / "exp.yaml"
    path.write_text(BASE)
    with pytest.raises(ValueError) as err:
        load_config(
            path, ['losses.terms=[{name: action_squared, target: policy, weight: 0.1, scale: 2}]']
        )
    assert "scale" in str(err.value)


# ------------------------------------------------------------------ the selection loss
#  BCE between the actor's probability of FORCE control and whether the axis is in contact.
AXES = 3


def selection_ctx(learner, probability, contact, step: int = 5) -> LossContext:
    return LossContext(
        learner=learner,
        target="policy",
        step=step,
        policy_outputs={"selection_prob": probability},
        sampled={"in_contact": contact},
    )


def selection_loss(**kwargs):
    from robonuke_rl_core.losses.losses import SupervisedSelectionLoss

    return SupervisedSelectionLoss(**{"num_axes": AXES, **kwargs})


def test_the_selection_loss_is_the_bce_against_the_contact_flags():
    learner = FakeLearner()
    torch.manual_seed(0)
    probability = torch.rand(NUM_AGENTS * ROWS, AXES)
    contact = (torch.rand(NUM_AGENTS * ROWS, AXES) > 0.5).float()

    value = selection_loss().compute(selection_ctx(learner, probability, contact))
    expected = torch.nn.functional.binary_cross_entropy(
        probability, contact, reduction="none"
    ).view(NUM_AGENTS, -1).mean(dim=-1)
    assert value.shape == (NUM_AGENTS,)
    assert torch.allclose(value, expected, atol=1e-6)


def test_a_perfect_prediction_costs_almost_nothing_and_a_wrong_one_costs_a_lot():
    learner = FakeLearner()
    contact = torch.tensor([[1.0, 0.0, 1.0]]).expand(NUM_AGENTS * ROWS, AXES).contiguous()
    right = selection_loss().compute(selection_ctx(learner, contact.clone(), contact))
    wrong = selection_loss().compute(selection_ctx(learner, 1.0 - contact, contact))
    # clamped at 1e-7, so neither is infinite and the ordering is the whole point
    assert float(right.max()) < 1.0e-6
    assert float(wrong.min()) > 10.0
    assert torch.isfinite(wrong).all()


def test_the_selection_loss_is_per_agent():
    """Agent 1's contacts must not move agent 0's or agent 2's value."""
    learner = FakeLearner()
    torch.manual_seed(1)
    probability = torch.rand(NUM_AGENTS * ROWS, AXES)
    contact = (torch.rand(NUM_AGENTS * ROWS, AXES) > 0.5).float()
    before = selection_loss().compute(selection_ctx(learner, probability, contact))

    extreme = contact.clone()
    extreme[ROWS : 2 * ROWS] = 1.0 - extreme[ROWS : 2 * ROWS]
    after = selection_loss().compute(selection_ctx(learner, probability, extreme))
    assert torch.equal(before[[0, 2]], after[[0, 2]])
    assert not torch.equal(before[1], after[1])


def test_the_gradient_reaches_the_probability_and_only_the_right_rows():
    learner = FakeLearner()
    probability = torch.full((NUM_AGENTS * ROWS, AXES), 0.5, requires_grad=True)
    contact = torch.zeros(NUM_AGENTS * ROWS, AXES)
    contact[0, 0] = 1.0
    selection_loss().compute(selection_ctx(learner, probability, contact)).sum().backward()
    # pushing toward force where there is contact, away from it elsewhere
    assert float(probability.grad[0, 0]) < 0.0
    assert float(probability.grad[0, 1]) > 0.0


def test_the_selection_loss_declares_what_it_needs_from_the_memory():
    assert selection_loss().required_memory_keys() == {"in_contact": AXES}
    assert selection_loss(num_axes=6).required_memory_keys() == {"in_contact": 6}
    with pytest.raises(ValueError, match="num_axes"):
        selection_loss(num_axes=0)


def test_build_aux_losses_carries_the_memory_keys_to_the_learner():
    aux = build_aux_losses(
        LossesCfg(
            terms=[
                LossTermCfg(
                    name="supervised_selection",
                    target="policy",
                    weight=1.0,
                    kwargs={"num_axes": AXES},
                )
            ]
        ),
        NUM_AGENTS,
    )
    assert aux.memory_keys == {"in_contact": AXES}
    # and a config with no term at all still asks for nothing
    assert build_aux_losses(LossesCfg(terms=[]), NUM_AGENTS) is None


def test_a_missing_contact_tensor_says_how_to_make_it_appear():
    learner = FakeLearner()
    probability = torch.rand(NUM_AGENTS * ROWS, AXES)
    ctx = LossContext(
        learner=learner, target="policy", policy_outputs={"selection_prob": probability}
    )
    with pytest.raises(RuntimeError) as err:
        selection_loss().compute(ctx)
    assert "in_contact" in str(err.value) and "required_memory_keys" in str(err.value)


def test_a_policy_without_a_selection_head_says_so():
    learner = FakeLearner()
    ctx = LossContext(
        learner=learner,
        target="policy",
        policy_outputs={"selection_prob": None},
        sampled={"in_contact": torch.zeros(NUM_AGENTS * ROWS, AXES)},
    )
    with pytest.raises(RuntimeError, match="bernoulli_action_dims"):
        selection_loss().compute(ctx)


def test_a_width_mismatch_names_both_widths():
    learner = FakeLearner()
    probability = torch.rand(NUM_AGENTS * ROWS, AXES)
    contact = torch.zeros(NUM_AGENTS * ROWS, 2)
    with pytest.raises(ValueError, match="num_axes"):
        selection_loss().compute(selection_ctx(learner, probability, contact))

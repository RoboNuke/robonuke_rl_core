"""The plain MLP architecture (``model.architecture: mlp``, ``models/mlp.py``).

Shapes for both learners, the two papers' nets it has to express (ResiP's fixed-std plain
Gaussian residual, Policy Decorator's soft-bounded SAC actor), the init choices, the three
log-std bounds, and the rules the fields obey. The stacked-parameter, independence and
checkpoint-slicing invariants run in ``test_factory.py`` for every architecture.
"""

from __future__ import annotations

import math

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.models.cfg import MlpActorCfg, MlpCriticCfg, MlpModelCfg
from robonuke_rl_core.models.factory import build_models
from robonuke_rl_core.models.mlp import (
    MlpActor,
    MlpActorNet,
    activation_module,
    check_mlp_actor_fields,
)
from robonuke_rl_core.models.simba import squash_log_prob_correction, safe_atanh
from robonuke_rl_core.optim import BlockAdamW

NUM_AGENTS = 2
OBS_DIM = 5
ACT_DIM = 3
ROWS = 4


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def cfg(**actor) -> MlpModelCfg:
    return MlpModelCfg(actor=MlpActorCfg(hidden_size=16, num_layers=2, **actor), critic=MlpCriticCfg(hidden_size=16, num_layers=2))


def build(learner: str, model_cfg: MlpModelCfg, num_critics: int = 2, asymmetric: bool = False) -> dict:
    torch.manual_seed(0)
    return build_models(
        learner, model_cfg, box(OBS_DIM), box(OBS_DIM + 2) if asymmetric else None, box(ACT_DIM),
        NUM_AGENTS, "cpu", num_critics=num_critics,
    )


def plain_actor(**fields) -> MlpActorNet:
    defaults = dict(
        obs_dim=OBS_DIM, action_dim=ACT_DIM, hidden_size=16, num_layers=2, activation="ReLU",
        init="orthogonal", output_init_scale=0.01, init_log_std=0.0, learn_std=True,
        use_state_dependent_std=False, log_std_bound="clip", min_log_std=-20.0, max_log_std=2.0,
    )
    torch.manual_seed(0)
    return MlpActorNet(**{**defaults, **fields})


# ------------------------------------------------------------------ shapes, both learners
def test_sac_models_have_the_learners_shapes():
    models = build("sac", cfg(), num_critics=3)
    assert set(models) == {"policy", "critics", "target_critics"}
    assert models["critics"].members == 3
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions, outputs = models["policy"].act({"observations": observations}, role="policy")
    assert actions.shape == (NUM_AGENTS * ROWS, ACT_DIM)
    assert outputs["log_prob"].shape == (NUM_AGENTS * ROWS, 1)
    assert outputs["log_std"].shape == (NUM_AGENTS * ROWS, ACT_DIM)
    assert outputs["mean_actions"].shape == (NUM_AGENTS * ROWS, ACT_DIM)
    assert torch.all(actions.abs() <= 1.0)  # squashed
    q, _ = models["critics"].act({"observations": observations, "taken_actions": actions}, role="critics")
    assert q.shape == (3, NUM_AGENTS * ROWS, 1)
    assert models["policy"].get_entropy(role="policy").shape == (NUM_AGENTS * ROWS, ACT_DIM)
    for param in models["critics"].parameters():
        assert param.shape[:2] == (NUM_AGENTS, 3)


def test_ppo_models_have_the_learners_shapes():
    models = build("ppo", cfg(), asymmetric=True)
    assert set(models) == {"policy", "value"}
    assert models["value"].num_observations == OBS_DIM + 2  # the asymmetric critic takes the state
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    taken = torch.rand(NUM_AGENTS * ROWS, ACT_DIM) * 2 - 1
    actions, outputs = models["policy"].act({"observations": observations, "taken_actions": taken}, role="policy")
    assert torch.allclose(actions, taken, atol=1e-6)  # squashed: tanh(atanh(a))
    assert outputs["log_prob"].shape == (NUM_AGENTS * ROWS, 1)
    value, _ = models["value"].act({"observations": torch.randn(NUM_AGENTS * ROWS, OBS_DIM + 2)}, role="value")
    assert value.shape == (NUM_AGENTS * ROWS, 1)


# ------------------------------------------------------------------ ResiP's residual net
def test_resips_residual_net_is_expressible():
    """512 x 2 SiLU actor and critic, kaiming hidden layers, an orthogonal mean head with gain
    0.01, a plain Gaussian with a fixed state-independent log std of -1.0."""
    resip = MlpModelCfg(
        actor=MlpActorCfg(
            hidden_size=512, num_layers=2, activation="SiLU", init="kaiming", output_init_scale=0.01,
            init_log_std=-1.0, learn_std=False, squash=False,
        ),
        critic=MlpCriticCfg(hidden_size=512, num_layers=2, activation="SiLU", init="kaiming", output_init_scale=1.0),
    )
    models = build("ppo", resip)
    policy = models["policy"]
    net = policy.net._meta  # the plain module's structure
    linears = [m for m in net.trunk if isinstance(m, nn.Linear)]
    assert len(linears) == 2 and all(m.out_features == 512 for m in linears)
    assert sum(isinstance(m, nn.SiLU) for m in net.trunk) == 2
    assert isinstance(models["value"].net._meta.trunk[1], nn.SiLU)

    # the fixed log std is a buffer: stacked, checkpointed, invisible to the optimizer
    assert "log_std" not in {name for name, _ in net.named_parameters()}
    assert "log_std" in {name for name, _ in net.named_buffers()}
    assert "log_std" in policy.agent_state_dict(0)
    assert torch.equal(policy.agent_state_dict(0)["log_std"], torch.full((ACT_DIM,), -1.0))
    BlockAdamW(policy.parameters(), NUM_AGENTS, lr=1e-3)  # every parameter has the agent dim
    assert all("log_std" not in name for name, _ in policy.named_parameters())

    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    taken = torch.randn(NUM_AGENTS * ROWS, ACT_DIM) * 3.0  # a plain Gaussian is unbounded
    actions, outputs = policy.act({"observations": observations, "taken_actions": taken}, role="policy")
    assert torch.equal(actions, taken)
    assert torch.all(outputs["log_std"] == -1.0)
    mean = outputs["mean_actions"]
    expected = torch.distributions.Normal(mean, math.exp(-1.0)).log_prob(taken).sum(-1, keepdim=True)
    assert torch.allclose(outputs["log_prob"], expected, atol=1e-5)
    # sampling draws from the same Gaussian, unsquashed: with std e^-1 the mean is near zero
    # (gain 0.01 head) and samples spread around it
    sampled, _ = policy.act({"observations": observations}, role="policy")
    assert float(sampled.std()) > 0.2


def test_kaiming_hidden_layers_and_orthogonal_output_heads():
    actor = plain_actor(init="kaiming", hidden_size=512, obs_dim=256, output_init_scale=0.01)
    first = actor.trunk[0]
    assert first.weight.std().item() == pytest.approx(math.sqrt(2.0 / 256), rel=0.1)  # fan-in normal
    assert torch.all(first.bias == 0)
    assert torch.allclose(torch.linalg.svdvals(actor.mean.weight), torch.full((ACT_DIM,), 0.01), atol=1e-6)
    assert torch.all(actor.mean.bias == 0)

    actor = plain_actor(init="orthogonal", hidden_size=256, obs_dim=512)
    weight = actor.trunk[0].weight  # (256, 512): orthonormal rows times sqrt(2)
    assert torch.allclose(weight @ weight.T, 2.0 * torch.eye(256), atol=1e-4)


# ------------------------------------------------------------------ Policy Decorator's SAC actor
def test_policy_decorators_soft_log_std_bound_is_expressible_and_respected():
    """3 x 256 ReLU, a state-dependent log std mapped through
    ``min + 0.5 (max - min) (tanh(x) + 1)`` on [-20, 2] instead of a clamp."""
    decorator = MlpModelCfg(
        actor=MlpActorCfg(hidden_size=256, num_layers=3, activation="ReLU", use_state_dependent_std=True, log_std_bound="tanh"),
        critic=MlpCriticCfg(hidden_size=256, num_layers=3, activation="ReLU"),
    )
    models = build("sac", decorator)
    policy = models["policy"]
    net = policy.net._meta
    assert len([m for m in net.trunk if isinstance(m, nn.Linear)]) == 3
    assert isinstance(net.log_std_head, nn.Linear)

    # huge inputs: the raw head output is far outside the range, the bound keeps it inside
    huge = torch.randn(NUM_AGENTS * ROWS, OBS_DIM) * 1.0e6
    _, outputs = policy.act({"observations": huge}, role="policy")
    log_std = outputs["log_std"]
    # within the limits (float32 tanh saturates to exactly +-1, so the bounds themselves appear)
    assert torch.all(log_std >= -20.0) and torch.all(log_std <= 2.0)
    assert float(log_std.min()) < -10.0 or float(log_std.max()) > 1.0  # the bound was active

    # the exact map, against the plain net's raw head output
    plain = plain_actor(use_state_dependent_std=True, log_std_bound="tanh")
    observations = torch.randn(ROWS, OBS_DIM) * 50.0
    hidden = plain.trunk(observations)
    raw = plain.log_std_head(hidden)
    _, bounded = plain(observations)
    assert torch.allclose(bounded, -20.0 + 0.5 * 22.0 * (torch.tanh(raw) + 1.0), atol=1e-6)
    # and at a zero input (hidden all zero: zero biases) the std is exactly the configured one
    _, at_zero = plain(torch.zeros(ROWS, OBS_DIM))
    assert torch.allclose(at_zero, torch.zeros(ROWS, ACT_DIM), atol=1e-4)  # float32 atanh/tanh round trip


def test_the_hard_clip_and_the_unbounded_log_std():
    clipped = plain_actor(log_std_bound="clip", min_log_std=-1.0, max_log_std=1.0)
    with torch.no_grad():
        clipped.log_std.fill_(5.0)
    _, log_std = clipped(torch.randn(ROWS, OBS_DIM))
    assert torch.all(log_std == 1.0)
    free = plain_actor(log_std_bound="none")
    with torch.no_grad():
        free.log_std.fill_(5.0)
    _, log_std = free(torch.randn(ROWS, OBS_DIM))
    assert torch.all(log_std == 5.0)
    # the tanh bound starts at the configured std too (the raw value is the bound's inverse)
    soft = plain_actor(log_std_bound="tanh", init_log_std=-1.0, learn_std=True)
    _, log_std = soft(torch.randn(ROWS, OBS_DIM))
    assert torch.allclose(log_std, torch.full((ROWS, ACT_DIM), -1.0), atol=1e-4)


def test_squashed_log_prob_is_the_tanh_corrected_gaussian():
    policy = build("ppo", cfg(squash=True))["policy"]
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    taken = torch.rand(NUM_AGENTS * ROWS, ACT_DIM) * 1.8 - 0.9
    _, outputs = policy.act({"observations": observations, "taken_actions": taken}, role="policy")
    mean, _ = policy.compute({"observations": observations}, role="policy")
    u = safe_atanh(taken)
    expected = torch.distributions.Normal(mean, outputs["log_std"].exp()).log_prob(u).sum(-1, keepdim=True)
    expected = expected - squash_log_prob_correction(u).unsqueeze(-1)
    assert torch.allclose(outputs["log_prob"], expected, atol=1e-5)
    assert torch.allclose(outputs["mean_actions"], torch.tanh(mean))


# ------------------------------------------------------------------ the rules
def test_activations_are_torch_nn_class_names():
    assert isinstance(activation_module("SiLU"), nn.SiLU)
    assert isinstance(activation_module("Tanh"), nn.Tanh)
    for bad in ("Swish", "relu", "torch.nn.ReLU"):
        with pytest.raises(ValueError, match="activation"):
            activation_module(bad)


def valid_fields(**overrides) -> dict:
    fields = dict(
        hidden_size=8, num_layers=1, activation="ReLU", init="orthogonal", output_init_scale=0.01,
        init_log_std=0.0, learn_std=True, use_state_dependent_std=False, log_std_bound="clip",
        min_log_std=-20.0, max_log_std=2.0, squash=True,
    )
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"learn_std": False, "use_state_dependent_std": True}, "learn_std"),
        ({"log_std_bound": "sigmoid"}, "log_std_bound"),
        ({"log_std_bound": "tanh", "init_log_std": 2.0}, "init_log_std"),  # tanh never reaches a bound
        ({"init_log_std": 3.0}, "init_log_std"),  # outside the clip range
        ({"min_log_std": 2.0, "max_log_std": 2.0}, "min_log_std"),
        ({"num_layers": 0}, "num_layers"),
        ({"hidden_size": 0}, "hidden_size"),
        ({"init": "xavier"}, "init"),
        ({"output_init_scale": 0.0}, "output_init_scale"),
        ({"activation": "Swish"}, "activation"),
    ],
)
def test_each_actor_rule_rejects_a_bad_value(overrides, match):
    with pytest.raises(ValueError, match=match):
        check_mlp_actor_fields("model.actor", **valid_fields(**overrides))


def test_edge_values_the_rules_allow():
    check_mlp_actor_fields("model.actor", **valid_fields(init_log_std=2.0))  # on the clip edge
    check_mlp_actor_fields("model.actor", **valid_fields(log_std_bound="none", init_log_std=7.0, min_log_std=5.0, max_log_std=1.0))
    check_mlp_actor_fields("model.actor", **valid_fields(learn_std=False))  # ResiP's frozen std


def test_the_actor_refuses_selection_dims_and_points_at_simba():
    with pytest.raises(ValueError, match="simba"):
        MlpActor(box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=1, pos_component_dims=[0], force_component_dims=[1], selection_names=["x"])

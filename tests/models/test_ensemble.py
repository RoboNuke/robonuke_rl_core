"""VmapEnsemble: a slot must behave exactly like the plain module it was built from."""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.models.ensemble import VmapEnsemble
from robonuke_rl_core.models.simba import (
    EnsembleActor,
    EnsembleQCritic,
    SimbaActorNet,
    SimbaQCriticNet,
)

NUM_AGENTS = 3
OBS_DIM, ACT_DIM, ROWS = 5, 2, 4


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


class Tiny(nn.Module):
    """A plain module with a weight, a bias and a buffer."""

    def __init__(self, in_dim: int = OBS_DIM, out_dim: int = ACT_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.register_buffer("offset", torch.zeros(out_dim))

    def forward(self, x):
        return self.fc(x) * self.scale + self.offset


# ------------------------------------------------------------------ the core equivalence
def test_each_slot_matches_the_plain_module_it_holds():
    torch.manual_seed(0)
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    x = torch.randn(NUM_AGENTS, ROWS, OBS_DIM)

    out = ensemble(x)
    assert out.shape == (NUM_AGENTS, ROWS, ACT_DIM)

    for agent in range(NUM_AGENTS):
        plain = Tiny()
        plain.load_state_dict(ensemble.agent_state_dict(agent))
        with torch.no_grad():
            expected = plain(x[agent])
        assert torch.allclose(out[agent], expected, atol=1e-6), f"agent {agent}"


def test_gradients_match_a_loop_over_plain_modules():
    torch.manual_seed(1)
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    x = torch.randn(NUM_AGENTS, ROWS, OBS_DIM)

    plains = []
    for agent in range(NUM_AGENTS):
        plain = Tiny()
        plain.load_state_dict(ensemble.agent_state_dict(agent))
        plains.append(plain)

    ensemble(x).square().mean().backward()
    torch.stack([plain(x[agent]) for agent, plain in enumerate(plains)]).square().mean().backward()

    for name in ensemble._param_names:
        stacked = getattr(ensemble, ensemble._mangle(name))
        assert stacked.grad is not None, name
        for agent, plain in enumerate(plains):
            expected = dict(plain.named_parameters())[name].grad
            assert torch.allclose(stacked.grad[agent], expected, atol=1e-7), f"{name}, {agent}"


def test_per_agent_inits_are_distinct():
    torch.manual_seed(2)
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    weight = getattr(ensemble, ensemble._mangle("fc.weight"))
    for agent in range(1, NUM_AGENTS):
        assert not torch.equal(weight[0], weight[agent])


def test_a_slot_round_trips_through_a_plain_module():
    torch.manual_seed(3)
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    plain = Tiny()  # a fresh, different init
    before = getattr(ensemble, ensemble._mangle("fc.weight"))[0].clone()
    ensemble.load_agent_state_dict(1, plain.state_dict())
    for name, tensor in plain.state_dict().items():
        stacked = getattr(ensemble, ensemble._mangle(name))
        assert torch.equal(stacked[1], tensor), name
    # the other slots are untouched (checked on a parameter whose init really differs)
    assert torch.equal(getattr(ensemble, ensemble._mangle("fc.weight"))[0], before)


def test_state_dict_mismatches_raise():
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    with pytest.raises(KeyError):
        ensemble.load_agent_state_dict(0, {"fc.weight": torch.zeros(ACT_DIM, OBS_DIM)})
    bad = dict(Tiny().state_dict())
    bad["fc.weight"] = torch.zeros(ACT_DIM + 1, OBS_DIM)
    with pytest.raises(ValueError):
        ensemble.load_agent_state_dict(0, bad)
    with pytest.raises(ValueError):
        ensemble.agent_state_dict(NUM_AGENTS)


def test_the_meta_template_is_not_a_parameter():
    """A registered meta module would leak storage-free parameters into the optimizer."""
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    for name, param in ensemble.named_parameters():
        assert param.shape[0] == NUM_AGENTS, name
        assert not param.is_meta, name
    assert all("_meta" not in name for name in ensemble.state_dict())


def test_flat_rows_must_divide_by_the_agent_count():
    ensemble = VmapEnsemble(Tiny, NUM_AGENTS)
    with pytest.raises(ValueError) as err:
        ensemble.forward_flat(torch.zeros(NUM_AGENTS * ROWS + 1, OBS_DIM))
    assert "num_agents" in str(err.value)


# ------------------------------------------------------------------ the real models
def test_actor_slot_matches_the_plain_actor_net():
    torch.manual_seed(4)
    actor = EnsembleActor(box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=NUM_AGENTS,
                          actor_n=1, actor_latent=8)
    plain = SimbaActorNet(
        obs_dim=OBS_DIM,
        policy_out_dim=ACT_DIM,
        num_continuous=ACT_DIM,
        hidden_dim=8,
        num_blocks=1,
        use_state_dependent_std=False,
        log_std_init=torch.zeros(ACT_DIM),
        last_layer_scale=1.0,
        scale_rows=None,
    )
    plain.load_state_dict(actor.net.agent_state_dict(2))

    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    with torch.no_grad():
        raw, outputs = actor.compute({"observations": observations}, "policy")
        expected_mean, expected_log_std = plain(observations[2 * ROWS :])
    assert torch.allclose(raw[2 * ROWS :], expected_mean, atol=1e-6)
    assert torch.allclose(outputs["log_std"][2 * ROWS :], expected_log_std, atol=1e-6)


def test_critic_slot_matches_the_plain_critic_net():
    torch.manual_seed(5)
    critic = EnsembleQCritic(box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=NUM_AGENTS,
                             critic_n=1, critic_latent=8)
    plain = SimbaQCriticNet(OBS_DIM, ACT_DIM, 8, 1, 0.0)
    plain.load_state_dict(critic.net.agent_state_dict(1))

    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions = torch.rand(NUM_AGENTS * ROWS, ACT_DIM) * 2 - 1
    with torch.no_grad():
        values, _ = critic.act({"observations": observations, "taken_actions": actions}, role="critic")
        expected = plain(observations[ROWS : 2 * ROWS], actions[ROWS : 2 * ROWS])
    assert torch.allclose(values[ROWS : 2 * ROWS], expected, atol=1e-6)


# ------------------------------------------------------------------ the actor's options
@pytest.mark.parametrize(
    "kwargs, expectation",
    [
        ({"use_state_dependent_std": True}, "state dependent std"),
        ({"bernoulli_action_dims": [1]}, "a Bernoulli dim"),
        ({"force_zero_action_dims": [1]}, "a force-zero dim"),
        ({"last_layer_scale": 0.01, "scale_down_action_dims": [0]}, "a scaled dim"),
        ({"second_act_init_std": 0.1, "second_act_init_std_dims": [0]}, "a second init std"),
    ],
)
def test_the_actor_options_build_and_act(kwargs, expectation):
    torch.manual_seed(6)
    actor = EnsembleActor(
        box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=NUM_AGENTS, actor_n=1, actor_latent=8, **kwargs
    )
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions, outputs = actor.act({"observations": observations}, role="policy")
    assert actions.shape == (NUM_AGENTS * ROWS, ACT_DIM), expectation
    assert outputs["log_prob"].shape == (NUM_AGENTS * ROWS, 1)
    assert torch.isfinite(actions).all() and torch.isfinite(outputs["log_prob"]).all()

    if "force_zero_action_dims" in kwargs:
        assert float(actions[:, 1].abs().max()) == 0.0
    if "bernoulli_action_dims" in kwargs:
        assert set(actions[:, 1].unique().tolist()) <= {-1.0, 1.0}

    # a replayed action gives a finite log_prob too (the atanh path)
    _, replay = actor.act({"observations": observations, "taken_actions": actions}, role="policy")
    assert torch.isfinite(replay["log_prob"]).all()


def test_overlapping_bernoulli_and_force_zero_dims_raise():
    with pytest.raises(ValueError) as err:
        EnsembleActor(
            box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=1,
            bernoulli_action_dims=[0], force_zero_action_dims=[0],
        )
    assert "disjoint" in str(err.value)

    with pytest.raises(ValueError):
        EnsembleActor(box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=1, bernoulli_action_dims=[99])

    with pytest.raises(ValueError) as err:
        EnsembleActor(
            box(OBS_DIM), box(ACT_DIM), "cpu", num_agents=1, second_act_init_std_dims=[0]
        )
    assert "second_act_init_std" in str(err.value)

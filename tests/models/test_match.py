"""MATCH: the conditional hybrid distribution over (pose, force) pairs.

Two styles share one actor. ``product`` treats the selection bits and the continuous dims as
independent — one squashed-Gaussian density times the Bernoullis — and is what every run so
far used, so the first test is a **regression gate**: its log-probability must still be the
exact float it has always been. ``match`` (Hou et al., hybrid action spaces) conditions the
continuous density on the selection: each selection bit names a (pose, force) pair and only
the selected member of that pair is a random variable, because the other one is a controller
input the env ignores on that axis.

**Which member the bit selects.** A bit of 1 is ``S = 1``, which is **force** control
(``S`` multiplies the force branch in the torque law), so bit 1 makes the *force* component
live. That one convention runs from here to the controller, so
:func:`test_the_gate_agrees_with_the_controller_s_selection_matrix` pins the actor against
``ActionInterface`` itself rather than against a hand-written expectation.

The toy layout used throughout, 7 actions:

====  ==========================================
dim   role
====  ==========================================
0, 1  pose components of pairs 0 and 1
2, 3  force components of pairs 0 and 1
4     a free continuous dim (always live)
5, 6  the selection bits for pairs 0 and 1
====  ==========================================
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Normal

from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaModelCfg
from robonuke_rl_core.models.simba import (
    SELECTION_DISTRIBUTIONS,
    EnsembleActor,
    safe_atanh,
    squash_correction_per_dim,
)

NUM_AGENTS = 3
OBS_DIM = 4
ROWS = 6  # per agent
ACT_DIM = 7
SELECTION = [5, 6]
POSE_PAIR = [0, 1]
FORCE_PAIR = [2, 3]
FREE = 4


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def actor(style: str, *, pairs: bool = True, seed: int = 0, **kwargs) -> EnsembleActor:
    torch.manual_seed(seed)
    return EnsembleActor(
        observation_space=box(OBS_DIM),
        action_space=box(ACT_DIM),
        device="cpu",
        num_agents=NUM_AGENTS,
        actor_n=1,
        actor_latent=8,
        bernoulli_action_dims=list(SELECTION),
        selection_distribution=style,
        pos_component_dims=list(POSE_PAIR) if pairs else None,
        force_component_dims=list(FORCE_PAIR) if pairs else None,
        **kwargs,
    )


def output_bias(model: EnsembleActor) -> torch.Tensor:
    """The stacked output-layer bias: ``(num_agents, policy_out_dim)``."""
    return model.net._stacked()[0]["trunk.fc_out.bias"]


def observations(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(NUM_AGENTS * ROWS, OBS_DIM)


def taken(model: EnsembleActor, obs: torch.Tensor) -> torch.Tensor:
    """A concrete action to score, so nothing in a test depends on sampling."""
    actions, _ = model.act({"observations": obs}, role="policy")
    return actions.detach()


def per_dim_reference(model: EnsembleActor, obs: torch.Tensor, actions: torch.Tensor):
    """The pieces a log-probability is built from, recomputed outside the model."""
    raw, extras = model.compute({"observations": obs}, "policy")
    log_std = torch.clamp(extras["log_std"], model._g_min_log_std, model._g_max_log_std)
    # index_select, not a slice: a slice is a non-contiguous view and sigmoid takes a
    # different kernel path over it, which costs the last bit — and the product path is
    # checked with torch.equal
    mean = raw.index_select(-1, model._cont_out_idx)
    u = safe_atanh(actions.index_select(-1, model._cont_action_idx))
    gaussian = Normal(mean, log_std.exp())
    probability = torch.sigmoid(raw.index_select(-1, model._bern_out_idx))
    bits = ((actions.index_select(-1, model._bern_action_idx) + 1.0) / 2.0).round().clamp(0, 1)
    return gaussian, u, probability, bits


# ------------------------------------------------------------------ 1. the regression gate
def test_product_is_the_same_float_it_has_always_been():
    """One summed Gaussian density plus the Bernoullis — the pre-MATCH expression, exactly."""
    for pairs in (False, True):  # passing the pairs must not touch the product path
        model = actor("product", pairs=pairs)
        obs = observations()
        actions = taken(model, obs)
        _, outputs = model.act({"observations": obs, "taken_actions": actions}, role="policy")

        gaussian, u, probability, bits = per_dim_reference(model, obs, actions)
        expected = (
            gaussian.log_prob(u).sum(dim=-1, keepdim=True)
            - squash_correction_per_dim(u).sum(dim=-1).unsqueeze(-1)
        ) + Bernoulli(probs=probability).log_prob(bits).sum(dim=-1, keepdim=True)
        assert torch.equal(outputs["log_prob"], expected), "the product density changed"


def test_product_entropy_is_still_per_dim():
    model = actor("product")
    obs = observations()
    model.act({"observations": obs}, role="policy")
    entropy = model.get_entropy(role="policy")
    assert entropy.shape == (NUM_AGENTS * ROWS, model.num_continuous + model.num_bernoulli)


# ------------------------------------------------------------------ 2. the match density
def test_match_scores_only_the_selected_member_of_each_pair():
    model = actor("match")
    obs = observations()
    actions = taken(model, obs)
    _, outputs = model.act({"observations": obs, "taken_actions": actions}, role="policy")

    gaussian, u, probability, bits = per_dim_reference(model, obs, actions)
    per_dim = gaussian.log_prob(u) - squash_correction_per_dim(u)
    column = {dim: i for i, dim in enumerate(model.continuous_dims)}
    free = per_dim[:, column[FREE]].unsqueeze(-1)
    gated = torch.zeros_like(free)
    for pair, (pose, force) in enumerate(zip(POSE_PAIR, FORCE_PAIR)):
        live = torch.where(  # bit 1 == force-controlled, so the force component is live
            bits[:, pair] > 0.5, per_dim[:, column[force]], per_dim[:, column[pose]]
        )
        gated = gated + live.unsqueeze(-1)
    expected = free + gated + Bernoulli(probs=probability).log_prob(bits).sum(-1, keepdim=True)
    assert torch.allclose(outputs["log_prob"], expected, atol=1e-6)

    # and it really differs from the product density: product charges for both members
    _, product_outputs = actor("product").act(
        {"observations": obs, "taken_actions": actions}, role="policy"
    )
    assert not torch.allclose(outputs["log_prob"], product_outputs["log_prob"])


def test_flipping_a_selection_bit_swaps_which_component_is_scored():
    """The whole point: setting the bit hands the axis to the force law, so the force
    component's density is the one that counts."""
    model = actor("match")
    obs = observations()
    actions = taken(model, obs)
    low = actions.clone()
    low[:, SELECTION] = -1.0  # bit 0: every axis position-controlled
    high = actions.clone()
    high[:, SELECTION] = 1.0  # bit 1: every axis force-controlled

    gaussian, u, _, _ = per_dim_reference(model, obs, low)
    per_dim = gaussian.log_prob(u) - squash_correction_per_dim(u)
    column = {dim: i for i, dim in enumerate(model.continuous_dims)}

    _, low_out = model.act({"observations": obs, "taken_actions": low}, role="policy")
    _, high_out = model.act({"observations": obs, "taken_actions": high}, role="policy")
    difference = (high_out["log_prob"] - low_out["log_prob"]).squeeze(-1)
    continuous_difference = sum(
        per_dim[:, column[force]] - per_dim[:, column[pose]]
        for pose, force in zip(POSE_PAIR, FORCE_PAIR)
    )
    # the Bernoulli term also changes; subtract it to isolate the gating
    raw, _ = model.compute({"observations": obs}, "policy")
    probability = torch.sigmoid(raw.index_select(-1, model._bern_out_idx))
    bernoulli = Bernoulli(probs=probability)
    ones, zeros = torch.ones_like(probability), torch.zeros_like(probability)
    bernoulli_difference = (
        bernoulli.log_prob(ones).sum(-1) - bernoulli.log_prob(zeros).sum(-1)
    )
    assert torch.allclose(difference - bernoulli_difference, continuous_difference, atol=1e-5)


# ------------------------------------------------------------------ 3. the match entropy
def test_match_entropy_mixes_the_two_components_by_the_selection_probability():
    model = actor("match")
    obs = observations()
    actions = taken(model, obs)
    model.act({"observations": obs, "taken_actions": actions}, role="policy")
    entropy = model.get_entropy(role="policy")
    assert entropy.shape == (NUM_AGENTS * ROWS, 1), "a mixture is not a per-dim quantity"

    gaussian, _, probability, _ = per_dim_reference(model, obs, actions)
    per_dim = gaussian.entropy()
    column = {dim: i for i, dim in enumerate(model.continuous_dims)}
    expected = per_dim[:, column[FREE]].unsqueeze(-1)
    for pair, (pose, force) in enumerate(zip(POSE_PAIR, FORCE_PAIR)):
        p = probability[:, pair]
        expected = expected + (
            (1.0 - p) * per_dim[:, column[pose]] + p * per_dim[:, column[force]]
        ).unsqueeze(-1)
    expected = expected + Bernoulli(probs=probability).entropy().sum(-1, keepdim=True)
    assert torch.allclose(entropy, expected, atol=1e-6)


# ------------------------------------------------------------------ 4. replay
@pytest.mark.parametrize("style", SELECTION_DISTRIBUTIONS)
def test_replaying_an_action_reproduces_its_log_probability(style):
    """What the update does: score an action the policy already took."""
    model = actor(style)
    obs = observations()
    actions, sampled = model.act({"observations": obs}, role="policy")
    _, replayed = model.act(
        {"observations": obs, "taken_actions": actions.detach()}, role="policy"
    )
    assert torch.allclose(sampled["log_prob"], replayed["log_prob"], atol=1e-4)


# ------------------------------------------------------------------ 5. gradients
@pytest.mark.parametrize("style", SELECTION_DISTRIBUTIONS)
def test_the_selection_logits_get_a_gradient(style):
    model = actor(style)
    obs = observations()
    actions = taken(model, obs)
    _, outputs = model.act({"observations": obs, "taken_actions": actions}, role="policy")
    assert outputs["selection_prob"].shape == (NUM_AGENTS * ROWS, len(SELECTION))
    assert outputs["selection_prob"].requires_grad, "the supervised loss needs a gradient here"

    outputs["log_prob"].sum().backward()
    bias = output_bias(model)
    assert bias.grad is not None
    selection_rows = bias.grad[:, model.num_continuous :]
    assert torch.any(selection_rows != 0.0)


# ------------------------------------------------------------------ 6. independence
@pytest.mark.parametrize("style", SELECTION_DISTRIBUTIONS)
def test_one_agent_s_observations_cannot_move_another_s_density(style):
    model = actor(style)
    obs = observations()
    actions = taken(model, obs)
    _, before = model.act({"observations": obs, "taken_actions": actions}, role="policy")

    extreme = obs.clone()
    extreme[ROWS : 2 * ROWS] *= 1.0e3  # agent 1 goes wild
    _, after = model.act({"observations": extreme, "taken_actions": actions}, role="policy")

    kept = list(range(ROWS)) + list(range(2 * ROWS, 3 * ROWS))
    assert torch.equal(before["log_prob"][kept], after["log_prob"][kept])


# ------------------------------------------------------------------ construction rules
def test_match_without_pairs_is_an_error():
    with pytest.raises(ValueError, match="pose, force"):
        actor("match", pairs=False)


def test_a_pair_has_to_be_a_continuous_dim():
    with pytest.raises(ValueError, match="not continuous action dims"):
        torch.manual_seed(0)
        EnsembleActor(
            observation_space=box(OBS_DIM),
            action_space=box(ACT_DIM),
            device="cpu",
            num_agents=1,
            actor_n=1,
            actor_latent=8,
            bernoulli_action_dims=list(SELECTION),
            selection_distribution="match",
            pos_component_dims=[0, 5],  # 5 is a selection bit
            force_component_dims=[2, 3],
        )


def test_a_pair_needs_a_selection_bit_each():
    with pytest.raises(ValueError, match="selection dim each"):
        torch.manual_seed(0)
        EnsembleActor(
            observation_space=box(OBS_DIM),
            action_space=box(ACT_DIM),
            device="cpu",
            num_agents=1,
            actor_n=1,
            actor_latent=8,
            bernoulli_action_dims=[6],
            selection_distribution="match",
            pos_component_dims=[0, 1],
            force_component_dims=[2, 3],
        )


def test_an_unknown_style_is_an_error():
    with pytest.raises(ValueError, match="selection_distribution"):
        actor("mixture")


# ------------------------------------------------------------------ the init bias
def test_the_init_bias_moves_the_selection_probability_and_nothing_else():
    """A bias of -2.2 puts the bit at sigmoid(-2.2) ~ 0.1 before anything is learned.

    The bit is 1 for force, so that is 10% *force*: position-dominant at init, which is what
    an experiment wants. The config carries no sign flip of its own — the number lands on
    the logit exactly as written.
    """
    biased = actor("match", selection_init_bias=-2.2)
    obs = observations()
    biased.act({"observations": obs}, role="policy")
    probability = biased._selection_prob
    assert float(probability.mean()) < 0.25

    neutral = actor("match")
    neutral.act({"observations": obs}, role="policy")
    assert float(neutral._selection_prob.mean()) > float(probability.mean())

    # and only the selection rows moved
    bias = output_bias(biased)
    plain = output_bias(neutral)
    assert torch.allclose(bias[:, : biased.num_continuous], plain[:, : biased.num_continuous])


# ------------------------------------------------------------------ the config rules
def test_the_config_rejects_match_without_selection_dims():
    cfg = SimbaModelCfg(actor=SimbaActorCfg(selection_distribution="match"))
    with pytest.raises(ValueError, match="bernoulli_action_dims"):
        cfg.validate(None)

    cfg.actor.bernoulli_action_dims = list(SELECTION)
    cfg.validate(None)


def test_the_config_rejects_an_init_bias_with_nothing_to_bias():
    cfg = SimbaModelCfg(actor=SimbaActorCfg(selection_init_bias=-2.2))
    with pytest.raises(ValueError, match="selection_init_bias"):
        cfg.validate(None)


def test_the_config_rejects_an_unknown_style():
    cfg = SimbaModelCfg(actor=SimbaActorCfg(selection_distribution="mixture"))
    with pytest.raises(ValueError, match="selection_distribution"):
        cfg.validate(None)


# ------------------------------------------------------------------ against the controller
def test_the_gate_agrees_with_the_controller_s_selection_matrix():
    """The bit means one thing, and the actor and the controller must agree on it.

    ``ActionInterface.split`` turns the same bit into ``S``; ``S = 1`` is force control. So
    the component the density scores must be the force one exactly where ``S`` is 1. This is
    the test that catches an inverted sign convention anywhere in the chain.
    """
    from robonuke_rl_core.envs.cfg import ControllerCfg
    from robonuke_rl_core.envs.interface import ActionInterface

    controller = ControllerCfg(enabled=True, use_pose=True, use_force=True)
    interface = ActionInterface(controller, device="cpu")
    layout = interface.layout
    pose_dims = interface.pos_component_indices
    force_dims = interface.force_component_indices

    torch.manual_seed(3)
    model = EnsembleActor(
        observation_space=box(OBS_DIM),
        action_space=box(layout.action_dim),
        device="cpu",
        num_agents=1,
        actor_n=1,
        actor_latent=8,
        bernoulli_action_dims=interface.selection_indices,
        selection_distribution="match",
        pos_component_dims=pose_dims,
        force_component_dims=force_dims,
    )
    obs = torch.randn(8, OBS_DIM)
    actions, outputs = model.act({"observations": obs}, role="policy")
    _, selection, _, _, _ = interface.split(actions.detach())

    # the selection matrix the controller builds, restricted to the gated axes
    gated_selection = selection[:, layout.force_axes]
    bits = (actions.detach()[:, interface.selection_indices] > 0.0).float()
    assert torch.equal(gated_selection, bits)

    # and the actor scored the force component exactly where S is 1
    column = {dim: i for i, dim in enumerate(model.continuous_dims)}
    gaussian, u, _, _ = per_dim_reference(model, obs, actions.detach())
    per_dim = gaussian.log_prob(u) - squash_correction_per_dim(u)
    live = torch.zeros(obs.shape[0], 1)
    for pair, (pose, force) in enumerate(zip(pose_dims, force_dims)):
        force_is_live = gated_selection[:, pair] > 0.5
        live = live + torch.where(
            force_is_live, per_dim[:, column[force]], per_dim[:, column[pose]]
        ).unsqueeze(-1)
    free = per_dim.index_select(-1, model._free_cont_out).sum(-1, keepdim=True)
    bernoulli = Bernoulli(probs=model._selection_prob).log_prob(model._selection_sample)
    expected = free + live + bernoulli.sum(-1, keepdim=True)
    assert torch.allclose(outputs["log_prob"], expected, atol=1e-6)


def test_the_selection_probability_is_the_probability_of_force():
    """``selection_prob`` answers "how likely is this axis to be force-controlled" directly,
    with no complement anywhere for the loss or the logs to get wrong."""
    model = actor("match", selection_init_bias=-4.0)
    obs = observations()
    _, outputs = model.act({"observations": obs}, role="policy")
    # a strongly negative bias is position-dominant, so a LOW force probability
    assert float(outputs["selection_prob"].mean()) < 0.1

    _, forceful = actor("match", selection_init_bias=4.0).act(
        {"observations": obs}, role="policy"
    )
    assert float(forceful["selection_prob"].mean()) > 0.9


# ------------------------------------------------------------------ the factory wiring
def test_the_factory_derives_the_pairs_from_the_controller():
    """No experiment writes pair indices; the action layout does."""
    from robonuke_rl_core.envs.cfg import ControllerCfg
    from robonuke_rl_core.models.cfg import SimbaCriticCfg
    from robonuke_rl_core.models.factory import actor_kwargs, build_models

    controller = ControllerCfg(enabled=True, use_pose=True, use_force=True)
    model_cfg = SimbaModelCfg(
        actor=SimbaActorCfg(
            actor_n=1,
            actor_latent=8,
            selection_distribution="match",
            bernoulli_action_dims=[7, 8, 9],
        ),
        critic=SimbaCriticCfg(critic_n=1, critic_latent=8),
    )
    kwargs = actor_kwargs(model_cfg, controller)
    assert kwargs["pos_component_dims"] == [0, 1, 2]
    assert kwargs["force_component_dims"] == [10, 11, 12]
    assert kwargs["selection_names"] == ["x", "y", "z"]

    models = build_models(
        "sac", model_cfg, box(OBS_DIM), None, box(19), 2, "cpu", controller
    )
    assert models["policy"].selection_distribution == "match"
    assert models["policy"].selection_names == ["x", "y", "z"]

    # and with no controller there is nothing to derive, so match cannot be built
    with pytest.raises(ValueError, match="pose, force"):
        build_models("sac", model_cfg, box(OBS_DIM), None, box(19), 2, "cpu", None)


def test_a_disabled_controller_contributes_nothing():
    from robonuke_rl_core.envs.cfg import ControllerCfg
    from robonuke_rl_core.models.factory import actor_kwargs

    model_cfg = SimbaModelCfg(actor=SimbaActorCfg(actor_n=1, actor_latent=8))
    for controller in (None, ControllerCfg(), ControllerCfg(enabled=True, use_force=False)):
        kwargs = actor_kwargs(model_cfg, controller)
        assert "pos_component_dims" not in kwargs
        assert "selection_names" not in kwargs

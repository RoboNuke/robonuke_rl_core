"""The Bernoulli action dims, exercised properly: log probs, gradients, statistics, entropy.

These dims carry the hybrid force-position mode switches, so "builds and returns ±1" is not
enough: PPO's ratio needs the replayed log_prob to match the sampling-time one, both learners
need gradients through the Bernoulli head, and the entropy bonus must cover the dims.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from robonuke_rl_core.models.simba import EnsembleActor, SimbaActorNet

OBS_DIM, ROWS = 5, 6
NUM_AGENTS = 2


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def all_bernoulli_actor(num_agents: int = NUM_AGENTS) -> EnsembleActor:
    return EnsembleActor(
        box(OBS_DIM), box(2), "cpu", num_agents=num_agents,
        actor_n=1, actor_latent=8, bernoulli_action_dims=[0, 1],
    )


def mixed_actor(num_agents: int = NUM_AGENTS) -> EnsembleActor:
    """Dims: 0 and 2 continuous, 1 Bernoulli."""
    return EnsembleActor(
        box(OBS_DIM), box(3), "cpu", num_agents=num_agents,
        actor_n=1, actor_latent=8, bernoulli_action_dims=[1],
    )


def plain_logits(actor: EnsembleActor, agent: int, observations: torch.Tensor) -> torch.Tensor:
    """The agent's Bernoulli logits, recomputed through the plain single-agent net."""
    plain = SimbaActorNet(
        obs_dim=OBS_DIM,
        policy_out_dim=actor._policy_out_dim,
        num_continuous=actor.num_continuous,
        hidden_dim=8,
        num_blocks=1,
        use_state_dependent_std=False,
        log_std_init=torch.zeros(actor.num_continuous),
        last_layer_scale=1.0,
        scale_rows=None,
    )
    plain.load_state_dict(actor.net.agent_state_dict(agent))
    with torch.no_grad():
        out, _ = plain(observations)
    return out[..., actor.num_continuous :]


def set_constant_logits(actor: EnsembleActor, logits: list) -> None:
    """Zero the output weights and set the bias, so every row's logits equal ``logits``."""
    with torch.no_grad():
        weight = getattr(actor.net, actor.net._mangle("trunk.fc_out.weight"))
        bias = getattr(actor.net, actor.net._mangle("trunk.fc_out.bias"))
        weight.zero_()
        bias.copy_(torch.tensor(logits).expand_as(bias))


# ------------------------------------------------------------------ log probabilities
def test_replayed_log_prob_matches_the_analytic_bernoulli():
    torch.manual_seed(0)
    actor = all_bernoulli_actor()
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions = torch.randint(0, 2, (NUM_AGENTS * ROWS, 2)).float() * 2.0 - 1.0

    _, outputs = actor.act({"observations": observations, "taken_actions": actions}, role="policy")

    for agent in range(NUM_AGENTS):
        sl = slice(agent * ROWS, (agent + 1) * ROWS)
        logits = plain_logits(actor, agent, observations[sl])
        bits = (actions[sl] + 1.0) / 2.0
        expected = torch.distributions.Bernoulli(logits=logits).log_prob(bits).sum(-1, keepdim=True)
        assert torch.allclose(outputs["log_prob"][sl], expected, atol=1e-6), f"agent {agent}"


@pytest.mark.parametrize("build", [all_bernoulli_actor, mixed_actor])
def test_sampling_and_replaying_the_same_action_agree(build):
    """PPO's first-minibatch invariant: ratio starts at 1, Bernoulli dims included."""
    torch.manual_seed(1)
    actor = build()
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions, sampled = actor.act({"observations": observations}, role="policy")
    _, replayed = actor.act(
        {"observations": observations, "taken_actions": actions}, role="policy"
    )
    assert torch.allclose(sampled["log_prob"], replayed["log_prob"], atol=1e-5)


# ------------------------------------------------------------------ gradients
def test_log_prob_gradients_reach_the_bernoulli_head():
    """The PPO path: d log_prob(stored action) / d logits is nonzero."""
    torch.manual_seed(2)
    actor = all_bernoulli_actor()
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions = torch.randint(0, 2, (NUM_AGENTS * ROWS, 2)).float() * 2.0 - 1.0
    _, outputs = actor.act({"observations": observations, "taken_actions": actions}, role="policy")
    outputs["log_prob"].sum().backward()
    bias = getattr(actor.net, actor.net._mangle("trunk.fc_out.bias"))
    assert bias.grad is not None and bias.grad.abs().sum() > 0


def test_straight_through_gradients_flow_from_the_actions():
    """The SAC path: the sampled ±1 actions carry gradient via the probability."""
    torch.manual_seed(3)
    actor = all_bernoulli_actor()
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions, _ = actor.act({"observations": observations}, role="policy")
    actions.sum().backward()
    bias = getattr(actor.net, actor.net._mangle("trunk.fc_out.bias"))
    assert bias.grad is not None and bias.grad.abs().sum() > 0


# ------------------------------------------------------------------ statistics
def test_sampling_frequency_matches_the_probability():
    torch.manual_seed(4)
    actor = all_bernoulli_actor(num_agents=1)
    set_constant_logits(actor, [2.0, -2.0])
    observations = torch.randn(20_000, OBS_DIM)
    with torch.no_grad():
        actions, _ = actor.act({"observations": observations}, role="policy")
    frequency = ((actions + 1.0) / 2.0).mean(dim=0)
    expected = torch.sigmoid(torch.tensor([2.0, -2.0]))
    assert torch.allclose(frequency, expected, atol=0.02), (frequency, expected)


# ------------------------------------------------------------------ entropy
def test_entropy_covers_the_bernoulli_dims():
    torch.manual_seed(5)
    rows = NUM_AGENTS * ROWS
    observations = torch.randn(rows, OBS_DIM)

    mixed = mixed_actor()
    mixed.act({"observations": observations}, role="policy")
    entropy = mixed.get_entropy(role="policy")
    assert entropy.shape == (rows, 3)  # 2 Gaussian columns + 1 Bernoulli column

    pure = all_bernoulli_actor()
    pure.act({"observations": observations}, role="policy")
    entropy = pure.get_entropy(role="policy")
    assert entropy.shape == (rows, 2)
    # the bonus must be differentiable
    entropy.sum().backward()
    bias = getattr(pure.net, pure.net._mangle("trunk.fc_out.bias"))
    assert bias.grad is not None and bias.grad.abs().sum() > 0


def test_entropy_matches_the_analytic_bernoulli_value():
    torch.manual_seed(6)
    actor = all_bernoulli_actor(num_agents=1)
    set_constant_logits(actor, [2.0, -2.0])
    observations = torch.randn(ROWS, OBS_DIM)
    actor.act({"observations": observations}, role="policy")
    entropy = actor.get_entropy(role="policy")
    p = torch.sigmoid(torch.tensor([2.0, -2.0]))
    expected = -(p * p.log() + (1 - p) * (1 - p).log())
    assert torch.allclose(entropy, expected.expand(ROWS, 2), atol=1e-6)

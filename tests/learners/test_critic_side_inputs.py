"""A critic action transform's ``side_inputs`` (per-transition tensors only the transform reads) and its
``critic_action_width`` (the critics' action input at another width than the agent's)."""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner


class Widening:
    """Critic action = [r, base]: reads a 3-wide side input, emits ACT_DIM + 3."""

    side_inputs = {"base": 3}
    critic_action_width = ACT_DIM + 3

    def __init__(self):
        self.calls = []

    def __call__(self, critic_inputs, actions, side):
        self.calls.append(side["base"])
        return torch.cat((actions, side["base"]), dim=-1)


def record(learner, steps=4, seed=1):
    torch.manual_seed(seed)
    n = learner.num_envs
    for step in range(steps):
        obs = torch.randn(n, OBS_DIM)
        actions, _ = learner.act(obs, None, timestep=step, timesteps=100)
        learner.record_transition(
            observations=obs, states=None, actions=actions, rewards=torch.randn(n, 1),
            next_observations=torch.randn(n, OBS_DIM), next_states=None,
            terminated=torch.zeros(n, 1, dtype=torch.bool), truncated=torch.zeros(n, 1, dtype=torch.bool),
            infos={"base": torch.full((n, 3), float(step)), "next_base": torch.full((n, 3), float(step + 1))},
            timestep=step, timesteps=100,
        )


@pytest.mark.parametrize("architecture", ["simba", "mlp"])
def test_the_side_inputs_are_stored_and_reach_the_transform_and_the_critics_are_wider(architecture):
    transform = Widening()
    learner = build_learner("sac", num_agents=2, architecture=architecture, critic_action_transform=transform)
    assert learner.critics.num_actions == ACT_DIM + 3
    assert learner.policy.num_actions == ACT_DIM
    record(learner)
    learner.update(timestep=4, timesteps=100)
    current, following = transform.calls[0], transform.calls[1]  # the target's next_ call comes first
    assert torch.all(following - current == 1.0) or torch.all(current - following == 1.0)
    for value in transform.calls:
        assert value.shape[1] == 3


def test_a_missing_side_input_fails_loud():
    learner = build_learner("sac", num_agents=1, critic_action_transform=Widening())
    n = learner.num_envs
    with pytest.raises(RuntimeError, match="next_base"):
        learner.record_transition(
            observations=torch.randn(n, OBS_DIM), states=None, actions=torch.zeros(n, ACT_DIM),
            rewards=torch.zeros(n, 1), next_observations=torch.randn(n, OBS_DIM), next_states=None,
            terminated=torch.zeros(n, 1, dtype=torch.bool), truncated=torch.zeros(n, 1, dtype=torch.bool),
            infos={"base": torch.zeros(n, 3)}, timestep=0, timesteps=10,
        )


def test_a_transform_set_after_init_is_refused():
    learner = build_learner("sac", num_agents=1)
    record_infos = Widening()
    learner.critic_action_transform = record_infos
    with pytest.raises(RuntimeError, match="before init"):
        learner._critic_actions({"observations": torch.zeros(1, OBS_DIM)}, "observations", torch.zeros(1, ACT_DIM))

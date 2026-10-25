"""Both learners train the plain MLP architecture: one CPU update each, a frozen std that
stays frozen under training, the soft log-std bound holding through updates, checkpoints, and
the independence rule on agent 1's data."""

from __future__ import annotations

import copy

import pytest
import torch

from helpers import OBS_DIM, build_learner, fill_memory
from test_independence import AGENT_UNDER_TEST, UNTOUCHED, assert_same, make_extreme, snapshot


def weights(module, agent: int = 0):
    return {k: v.clone() for k, v in module.agent_state_dict(agent).items()}


def changed(before, module, agent: int = 0) -> bool:
    after = module.agent_state_dict(agent)
    return any(not torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_one_update_moves_every_model_and_the_checkpoint_round_trips(learner_name, tmp_path):
    learner = build_learner(learner_name, architecture="mlp", num_agents=3, envs_per_agent=2, run_dirs_list=[tmp_path / f"a{i}" for i in range(3)])
    fill_memory(learner)
    before = {key: weights(getattr(learner, key), 1) for key in learner._checkpoint_model_keys()}
    torch.manual_seed(1)
    learner.update(timestep=0, timesteps=100)
    for key in learner._checkpoint_model_keys():
        if key == "target_critics":
            continue  # Polyak: moved, but through its own rule
        assert changed(before[key], getattr(learner, key), 1), key
    paths = learner.save_checkpoints(step=1)
    fresh = build_learner(learner_name, architecture="mlp", num_agents=1, envs_per_agent=2)
    meta = fresh.load_agent(paths[1], slot=0)
    assert meta["agent_idx"] == 1
    for key in learner._checkpoint_model_keys():
        assert_same(getattr(fresh, key).agent_state_dict(0), getattr(learner, key).agent_state_dict(1), key)


def test_sac_trains_a_q_ensemble_of_mlps():
    learner = build_learner("sac", architecture="mlp", num_agents=2, envs_per_agent=2, num_critics=4, critic_subset=2, actor_q_reduction="mean")
    assert learner.critics.members == 4
    fill_memory(learner)
    before = weights(learner.critics)
    learner.update(timestep=0, timesteps=100)
    assert changed(before, learner.critics)


@pytest.mark.parametrize("learner_name, squash", [("ppo", False), ("ppo", True), ("sac", True)])
def test_a_frozen_std_stays_frozen_under_training(learner_name, squash):
    learner = build_learner(
        learner_name, architecture="mlp", num_agents=2, envs_per_agent=2,
        model_overrides={"learn_std": False, "init_log_std": -1.0, "squash": squash},
    )
    fill_memory(learner)
    for step in range(3):
        learner.update(timestep=step, timesteps=100)
        if learner_name == "ppo":
            fill_memory(learner, seed=step + 2)
    for agent in range(2):
        assert torch.equal(learner.policy.agent_state_dict(agent)["log_std"], torch.full((2,), -1.0))
    _, outputs = learner.policy.act({"observations": torch.randn(4, OBS_DIM)}, role="policy")
    assert torch.all(outputs["log_std"] == -1.0)
    # not a parameter: the optimizer never sees it
    assert all("log_std" not in name for name, _ in learner.policy.named_parameters())
    assert "log_std" in learner.policy.agent_state_dict(0)  # but the checkpoint carries it


def test_a_learned_std_does_move():
    learner = build_learner("ppo", architecture="mlp", num_agents=2, envs_per_agent=2, model_overrides={"init_log_std": -1.0}, entropy_loss_scale=0.1)
    fill_memory(learner)
    learner.update(timestep=0, timesteps=100)
    assert not torch.equal(learner.policy.agent_state_dict(0)["log_std"], torch.full((2,), -1.0))


def test_the_soft_bound_holds_through_sac_updates_on_wild_inputs():
    learner = build_learner(
        "sac", architecture="mlp", num_agents=2, envs_per_agent=2, normalize_observations=False,
        model_overrides={"use_state_dependent_std": True, "log_std_bound": "tanh", "min_log_std": -5.0, "max_log_std": 1.0},
    )
    fill_memory(learner)
    learner.memory.get_tensor_by_name("observations").mul_(1.0e4)
    learner.memory.get_tensor_by_name("next_observations").mul_(1.0e4)
    for step in range(3):
        learner.update(timestep=step, timesteps=100)
    _, outputs = learner.policy.act({"observations": torch.randn(4, OBS_DIM) * 1.0e4}, role="policy")
    assert torch.all(outputs["log_std"] >= -5.0) and torch.all(outputs["log_std"] <= 1.0)
    assert float(outputs["log_std"].min()) < -4.0  # the bound did the work


@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_agents_stay_independent(learner_name):
    learner = build_learner(learner_name, architecture="mlp", num_agents=3, envs_per_agent=2)
    fill_memory(learner)
    extreme = copy.deepcopy(learner)
    make_extreme(extreme, AGENT_UNDER_TEST)
    for step in range(2):
        torch.manual_seed(step)
        learner.update(timestep=step, timesteps=100)
        torch.manual_seed(step)
        extreme.update(timestep=step, timesteps=100)
    for agent in UNTOUCHED:
        assert_same(snapshot(learner, agent), snapshot(extreme, agent), f"agent {agent}")
    with pytest.raises(AssertionError):
        assert_same(snapshot(learner, AGENT_UNDER_TEST), snapshot(extreme, AGENT_UNDER_TEST))

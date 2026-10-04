"""Learner paths that the independence and base tests do not reach.

The SimBa periodic reset, the cosine LR schedule, a reward shaper, the random-action
warmup, asymmetric actor-critic, PPO's time-limit bootstrap, and the checkpoint lookup.
"""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, STATE_DIM, build_learner, fill_memory


def params_of(module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


# ------------------------------------------------------------------ SimBa periodic reset
def test_periodic_reset_rebuilds_the_networks_and_keeps_the_buffer():
    learner = build_learner(
        "sac",
        num_agents=2,
        envs_per_agent=2,
        periodic_reset_enabled=True,
        periodic_reset_frequency=4,
        periodic_reset_max=1,
    )
    fill_memory(learner)
    torch.manual_seed(1)
    learner.update(timestep=0, timesteps=100)

    before_policy = params_of(learner.policy)
    before_stats = learner.observation_normalizer.running_mean.clone()
    filled = learner.memory.memory_index

    learner._maybe_periodic_reset(timestep=4)  # on the frequency boundary
    assert learner._n_periodic_resets == 1
    assert not torch.equal(before_policy, params_of(learner.policy))  # fresh weights
    assert learner.policy is learner.models["policy"]  # the model dict was updated too
    for moment in learner.policy_optimizer.exp_avg:  # fresh Adam moments
        assert float(moment.abs().max()) == 0.0
    assert learner.policy_optimizer.t.tolist() == [0] * learner.num_agents
    assert float(learner._entropy_coefficient[0]) == pytest.approx(
        learner.cfg.initial_entropy_value
    )
    # the replay buffer and the normalizer statistics survive
    assert learner.memory.memory_index == filled
    assert torch.equal(before_stats, learner.observation_normalizer.running_mean)

    # the targets start equal to the fresh critics
    for target, critic in (
        (learner.target_critic_1, learner.critic_1),
        (learner.target_critic_2, learner.critic_2),
    ):
        assert torch.equal(params_of(target), params_of(critic))

    # and it stops at periodic_reset_max
    learner._maybe_periodic_reset(timestep=8)
    assert learner._n_periodic_resets == 1

    # the learner still updates after a reset
    torch.manual_seed(1)
    learner.update(timestep=8, timesteps=100)


def test_periodic_reset_without_a_model_cfg_raises():
    learner = build_learner(
        "sac", num_agents=2, envs_per_agent=2, periodic_reset_enabled=True,
        periodic_reset_frequency=2,
    )
    learner._model_cfg = None  # as if the caller never passed one
    with pytest.raises(RuntimeError) as err:
        learner._maybe_periodic_reset(timestep=2)
    assert "model_cfg" in str(err.value)


def test_periodic_reset_is_off_by_default():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2)
    before = params_of(learner.policy)
    for step in (1, 2, 4, 8, 16):
        learner._maybe_periodic_reset(timestep=step)
    assert learner._n_periodic_resets == 0
    assert torch.equal(before, params_of(learner.policy))


# ------------------------------------------------------------------ cosine LR
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_cosine_lr_decays_and_constant_does_not(learner_name):
    for schedule, should_decay in (("cosine", True), ("constant", False)):
        learner = build_learner(
            learner_name, num_agents=2, envs_per_agent=2, lr_schedule=schedule, lr_end=1.0e-6
        )
        fill_memory(learner)
        optimizer = learner.policy_optimizer
        start = float(optimizer.lr[0])
        torch.manual_seed(2)
        for _ in range(3):
            learner.update(timestep=0, timesteps=10)
        end = float(optimizer.lr[0])
        if should_decay:
            assert end < start, f"{learner_name}: cosine LR did not decay"
        else:
            assert end == start, f"{learner_name}: constant LR changed"


# ------------------------------------------------------------------ reward shaper
def scale_rewards(rewards, timestep, timesteps):
    """A "module:name" shaper, the only way a callable enters a config."""
    return rewards * 10.0


def test_a_reward_shaper_is_resolved_and_applied():
    learner = build_learner(
        "sac",
        num_agents=2,
        envs_per_agent=2,
        rewards_shaper="test_paths:scale_rewards",
    )
    rewards = torch.ones(learner.num_envs, 1)
    learner.record_transition(
        observations=torch.zeros(learner.num_envs, OBS_DIM),
        states=None,
        actions=torch.zeros(learner.num_envs, ACT_DIM),
        rewards=rewards,
        next_observations=torch.zeros(learner.num_envs, OBS_DIM),
        next_states=None,
        terminated=torch.zeros(learner.num_envs, 1, dtype=torch.bool),
        truncated=torch.zeros(learner.num_envs, 1, dtype=torch.bool),
        infos={},
        timestep=0,
        timesteps=100,
    )
    stored = learner.memory.get_tensor_by_name("rewards")[0]
    assert torch.allclose(stored, rewards * 10.0)
    # the episode statistics use the RAW reward, not the shaped one
    assert float(learner._episode_return[0]) == pytest.approx(1.0)


def test_a_bad_shaper_reference_raises():
    with pytest.raises(ValueError) as err:
        build_learner("sac", num_agents=2, envs_per_agent=2, rewards_shaper="test_paths:nope")
    assert "nope" in str(err.value)


# ------------------------------------------------------------------ random warmup
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_random_timesteps_emit_uniform_actions(learner_name):
    learner = build_learner(learner_name, num_agents=2, envs_per_agent=2, random_timesteps=5)
    torch.manual_seed(0)
    actions, _ = learner.act(
        torch.zeros(learner.num_envs, OBS_DIM), None, timestep=0, timesteps=100
    )
    assert actions.shape == (learner.num_envs, ACT_DIM)
    assert float(actions.min()) >= -1.0 and float(actions.max()) <= 1.0
    # after the warmup the policy is used; a zero observation gives a repeatable action
    torch.manual_seed(0)
    policy_actions, _ = learner.act(
        torch.zeros(learner.num_envs, OBS_DIM), None, timestep=5, timesteps=100
    )
    assert not torch.equal(actions, policy_actions)


# ------------------------------------------------------------------ asymmetric actor-critic
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_asymmetric_critic_consumes_the_state(learner_name):
    learner = build_learner(learner_name, num_agents=2, envs_per_agent=2, asymmetric=True)
    assert learner._asymmetric
    assert learner.state_normalizer is not None
    fill_memory(learner)
    torch.manual_seed(4)
    learner.update(timestep=10, timesteps=100)
    # the critic was built on the state space, not the observation space
    critic = learner.critic_1 if learner_name != "ppo" else learner.value
    assert critic.num_observations == STATE_DIM
    assert learner.policy.num_observations == OBS_DIM
    # both normalizers saw data
    assert float(learner.state_normalizer.current_count[0]) > 1.0


def test_asymmetric_sac_without_states_raises():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2, asymmetric=True)
    with pytest.raises(RuntimeError) as err:
        learner.record_transition(
            observations=torch.zeros(learner.num_envs, OBS_DIM),
            states=None,
            actions=torch.zeros(learner.num_envs, ACT_DIM),
            rewards=torch.zeros(learner.num_envs, 1),
            next_observations=torch.zeros(learner.num_envs, OBS_DIM),
            next_states=None,
            terminated=torch.zeros(learner.num_envs, 1, dtype=torch.bool),
            truncated=torch.zeros(learner.num_envs, 1, dtype=torch.bool),
            infos={},
            timestep=0,
            timesteps=100,
        )
    assert "states" in str(err.value)


# ------------------------------------------------------------------ PPO time-limit bootstrap
def test_time_limit_bootstrap_adds_the_next_value_on_truncation():
    learner = build_learner(
        "ppo", num_agents=2, envs_per_agent=2, time_limit_bootstrap=True, normalize_values=False
    )
    num_envs = learner.num_envs
    observations = torch.zeros(num_envs, OBS_DIM)
    truncated = torch.zeros(num_envs, 1, dtype=torch.bool)
    truncated[0] = True

    learner.act(observations, None, timestep=0, timesteps=100)
    with torch.no_grad():
        next_values, _ = learner.value.act({"observations": observations}, role="value")

    learner.record_transition(
        observations=observations,
        states=None,
        actions=torch.zeros(num_envs, ACT_DIM),
        rewards=torch.zeros(num_envs, 1),
        next_observations=observations,
        next_states=None,
        terminated=torch.zeros(num_envs, 1, dtype=torch.bool),
        truncated=truncated,
        infos={},
        timestep=0,
        timesteps=100,
    )
    stored = learner.memory.get_tensor_by_name("rewards")[0]
    expected = learner.cfg.discount_factor * float(next_values[0])
    assert float(stored[0]) == pytest.approx(expected, rel=1e-5)
    assert float(stored[1]) == 0.0  # not truncated: unchanged


# ------------------------------------------------------------------ checkpoint lookup
def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    from robonuke_rl_core.learners.base import LearnerBase

    for step in (5, 40, 7):
        (tmp_path / f"ckpt_{step}.pt").write_text("")
    (tmp_path / "ckpt_best.pt").write_text("")  # never the "latest"
    assert LearnerBase.latest_checkpoint(tmp_path).name == "ckpt_40.pt"

    with pytest.raises(FileNotFoundError):
        LearnerBase.latest_checkpoint(tmp_path / "empty")

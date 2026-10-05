"""Learner paths that the independence and base tests do not reach.

The SimBa periodic reset, the cosine LR schedule, a reward shaper, the random-action
warmup, asymmetric actor-critic, PPO's time-limit bootstrap, and the checkpoint lookup.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.learners.base import CHECKPOINT_BEST, checkpoint_name
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
    filled = learner.memory.size

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
    assert learner.memory.size == filled
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
    # step 0 of every env, back in env order: (num_agents, envs_per_agent, 1) -> (num_envs, 1)
    stored = learner.memory.time_view("rewards")[:, 0].reshape(learner.num_envs, 1)
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
    stored = learner.memory.time_view("rewards")[:, 0].reshape(num_envs, 1)
    expected = learner.cfg.discount_factor * float(next_values[0])
    assert float(stored[0]) == pytest.approx(expected, rel=1e-5)
    assert float(stored[1]) == 0.0  # not truncated: unchanged


# ------------------------------------------------------------------ checkpoint lookup
def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    from robonuke_rl_core.learners.base import LearnerBase

    for step in (5, 40, 7):
        (tmp_path / checkpoint_name(step)).write_text("")
    (tmp_path / CHECKPOINT_BEST).write_text("")  # never the "latest"
    assert LearnerBase.latest_checkpoint(tmp_path).name == checkpoint_name(40)

    with pytest.raises(FileNotFoundError):
        LearnerBase.latest_checkpoint(tmp_path / "empty")


# ------------------------------------------------------------------ GAE over the memory layout
def test_ppo_gae_matches_a_per_env_reference_loop():
    """The time view really is (agent, step, env): GAE must not mix steps across envs."""
    num_agents, envs_per_agent, rollout = 3, 2, 4
    learner = build_learner(
        "ppo",
        num_agents=num_agents,
        envs_per_agent=envs_per_agent,
        rollout=rollout,
        normalize_observations=False,
        normalize_values=False,
        time_limit_bootstrap=False,
    )
    torch.manual_seed(5)
    fill_memory(learner, steps=rollout)
    memory = learner.memory

    # what the update will read, snapshotted in (agent, step, env) form
    rewards = memory.time_view("rewards").clone()
    values = memory.time_view("values").clone()
    terminated = memory.time_view("terminated").clone()
    with torch.no_grad():
        last_values, _ = learner.value.act(
            {"observations": learner._next_observations}, role="value"
        )
    last_values = last_values.view(num_agents, envs_per_agent, 1)

    learner.update(timestep=0, timesteps=100)

    discount, lambda_ = learner.cfg.discount_factor, learner.cfg.gae_lambda
    reference = torch.zeros_like(rewards)
    for agent in range(num_agents):
        for env in range(envs_per_agent):
            advantage = 0.0
            for step in reversed(range(rollout)):
                next_value = (
                    values[agent, step + 1, env]
                    if step < rollout - 1
                    else last_values[agent, env]
                )
                not_done = 0.0 if bool(terminated[agent, step, env]) else 1.0
                advantage = (
                    rewards[agent, step, env]
                    - values[agent, step, env]
                    + discount * not_done * (next_value + lambda_ * advantage)
                )
                reference[agent, step, env] = advantage

    flat = reference.reshape(num_agents, -1)
    standardized = (reference - flat.mean(1).view(-1, 1, 1, 1)) / (
        flat.std(1).view(-1, 1, 1, 1) + 1e-8
    )
    assert torch.allclose(memory.time_view("returns"), reference + values, atol=1e-5)
    assert torch.allclose(memory.time_view("advantages"), standardized, atol=1e-5)


# ------------------------------------------------------------------ the SAC bootstrap
def test_sac_bootstraps_through_a_timeout_but_not_a_terminal_state():
    """A time limit is not the end of the world; only ``terminated & ~truncated`` is.

    Isaac Lab's Factory and Forge raise both flags at the limit, so reading ``terminated``
    alone would drop gamma*V(next) on every single episode.
    """
    learner = build_learner("sac", num_agents=2, envs_per_agent=2, normalize_observations=False)
    fill_memory(learner, steps=4)
    rows = 3
    sampled = dict(
        zip(
            learner._tensors_names,
            learner.memory.sample(names=learner._tensors_names, batch_size=rows)[0],
        )
    )
    inputs = {"observations": sampled["observations"]}
    next_inputs = {"observations": sampled["next_observations"]}

    def target_for(terminated: bool, truncated: bool) -> torch.Tensor:
        flags = dict(
            terminated=torch.full_like(sampled["terminated"], terminated),
            truncated=torch.full_like(sampled["truncated"], truncated),
        )
        torch.manual_seed(0)  # the target policy samples next_actions
        _, _, _, target_values = learner._compute_critic_loss(
            sampled={**sampled, **flags},
            inputs=inputs,
            next_inputs=next_inputs,
            critic_inputs=inputs,
            critic_next_inputs=next_inputs,
            rows=rows,
        )
        return target_values

    rewards = sampled["rewards"]
    running = target_for(False, False)
    timeout = target_for(True, True)  # the Factory/Forge time limit: both flags
    truncated_only = target_for(False, True)
    terminal = target_for(True, False)

    # the bootstrap survives a timeout, however the env labels it
    assert torch.allclose(timeout, running)
    assert torch.allclose(truncated_only, running)
    assert not torch.allclose(running, rewards)  # there IS a bootstrap term to keep
    # a genuine terminal state has no future: the target is the reward alone
    assert torch.allclose(terminal, rewards)

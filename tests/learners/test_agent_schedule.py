"""Agents trained side by side stay fully independent in time as well as in data: each has its own phase clock
(route_clock), its own learning start (learning_starts_rows), its own ResFiT warm-ups (uniform actions, the critic burst)
and its own actor delay, and an agent that is not due takes no step at all -- its weights, target networks, optimizer
moments and step counts are left exactly as they were."""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner

NAMES = ["corner", "grasp", "insert", "screw"]
INSERT = NAMES.index("insert")


def routed_ddpg(**extra):
    kwargs = dict(num_agents=3, envs_per_agent=1, architecture="mlp", rollout=64, std_by_label="eval_state/phase/id",
                  std_label_names=NAMES, route_label="insert", route_clock=True, learning_starts=0, learning_starts_rows=3,
                  batch_size=2, gradient_steps=1, policy_delay=1, random_action_scale=0.2, exploration_std=0.0,
                  critic_warmup_updates=2)
    kwargs.update(extra)
    return build_learner("ddpg", **kwargs)


def step(learner, t, phases):
    """One env step: each env's next observation is in ``phases[env]``; its start label is the previous step's."""
    n = learner.num_envs
    obs = torch.full((n, OBS_DIM), float(t))
    learner.record_transition(
        observations=obs, states=None, actions=torch.zeros(n, ACT_DIM), rewards=torch.zeros(n, 1),
        next_observations=obs + 1, next_states=None, terminated=torch.zeros(n, 1, dtype=torch.bool),
        truncated=torch.zeros(n, 1, dtype=torch.bool),
        infos={"eval_state": {"phase/id": torch.tensor([float(p) for p in phases])}, "episode_metrics_to_log": {}},
        timestep=t, timesteps=100,
    )
    learner.post_interaction(timestep=t, timesteps=100)


def snapshot(learner, agent):
    state = {}
    for name in ("policy", "target_policy", "critics", "target_critics"):
        for k, v in getattr(learner, name).agent_state_dict(agent).items():
            state[f"{name}.{k}"] = v.detach().clone()
    for opt_name in ("policy_optimizer", "critic_optimizer"):
        opt = getattr(learner, opt_name)
        state[f"{opt_name}.t"] = opt.t[agent].clone()
        for i, (m, v) in enumerate(zip(opt.exp_avg, opt.exp_avg_sq)):
            state[f"{opt_name}.m{i}"] = m[agent].clone()
            state[f"{opt_name}.v{i}"] = v[agent].clone()
    return state


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_each_agent_counts_its_own_phase_steps_and_rows():
    learner = routed_ddpg(learning_starts_rows=100)
    learner.enable_training_mode(True)
    step(learner, 0, [INSERT, INSERT, 0])  # the first labels: nothing starts in a phase yet
    for t in range(1, 5):
        step(learner, t, [INSERT, INSERT if t % 2 else 0, 0])  # agent 1 alternates, agent 2 never in insert
    assert learner._agent_route_steps == [4, 3, 0]  # agent 1 started steps 1, 2 and 4 in insert
    assert list(learner.memory._agent_size) == [4, 3, 0]


def test_an_agent_starts_warms_up_and_learns_on_its_own_while_the_others_stay_untouched():
    learner = routed_ddpg()
    learner.enable_training_mode(True)
    before = [snapshot(learner, a) for a in range(3)]
    step(learner, 0, [INSERT, 0, 0])
    for t in range(1, 4):  # agent 0 collects its 3 rows; agents 1 and 2 none
        step(learner, t, [INSERT, 0, 0])
    assert learner._agent_warmed == [True, False, False]  # agent 0's critic burst ran, alone
    assert learner._agent_updates[0] >= 2 and learner._agent_updates[1:] == [0, 0]
    after = [snapshot(learner, a) for a in range(3)]
    assert not same(before[0], after[0])  # agent 0 learned
    assert same(before[1], after[1]) and same(before[2], after[2])  # bit for bit: weights, targets, moments, step counts
    # acting: agent 0 with its actor (noise-free here), agents 1 and 2 still in their uniform warm-up
    obs = torch.randn(3, OBS_DIM)
    torch.manual_seed(0)
    acts = torch.stack([learner.act(obs, None, timestep=10, timesteps=100)[0] for _ in range(50)])
    assert float(acts[:, 0].std(dim=0).max()) == pytest.approx(0.0, abs=1e-7)  # the same every draw: the actor
    assert float(acts[:, 1:].std(dim=0).min()) > 0.05 and float(acts[:, 1:].abs().max()) <= 0.2  # uniform +-0.2
    # agent 1 now starts later on its own: its burst runs without touching agent 0's weights in that step
    step(learner, 4, [0, INSERT, 0])  # agent 0's last step in insert (a transition's phase is the previous label)
    mid0 = snapshot(learner, 0)
    for t in range(5, 8):  # agent 1's three steps in insert; agent 0 in corner
        step(learner, t, [0, INSERT, 0])
    assert learner._agent_warmed == [True, True, False]
    assert same(mid0, snapshot(learner, 0))  # agent 0 not in its phase these steps: untouched
    assert not same(after[1], snapshot(learner, 1)) and same(after[2], snapshot(learner, 2))


def test_the_actor_delay_counts_each_agents_own_gradient_steps():
    learner = routed_ddpg(policy_delay=2, critic_warmup_updates=0, learning_starts_rows=1)
    learner.enable_training_mode(True)
    step(learner, 0, [INSERT, INSERT, 0])
    step(learner, 1, [INSERT, 0, 0])  # agents 0 and 1 each take their first step
    step(learner, 2, [INSERT, 0, 0])  # only agent 0 is in its phase: its 2nd step, its actor's turn; agent 1 none
    assert learner._agent_updates == [2, 1, 0]
    assert learner.policy_optimizer.t.tolist() == [1, 0, 0]  # agent 0's actor stepped once, nobody else's


def test_one_agent_behaves_as_before():
    """One agent: no masking, the learner-wide path (the existing single-agent tests hold it to its behavior)."""
    learner = routed_ddpg(num_agents=1, learning_starts_rows=1, critic_warmup_updates=0)
    assert not learner._masked
    learner.enable_training_mode(True)
    step(learner, 0, [INSERT])
    step(learner, 1, [INSERT])
    assert learner._agent_updates == [1] and learner._updates == 1


def test_learner_wide_options_are_refused_with_per_agent_schedules():
    from types import SimpleNamespace

    from robonuke_rl_core.learners.cfg import DDPGCfg, SACCfg

    three = SimpleNamespace(experiment=SimpleNamespace(num_agents=3))
    base = dict(route_labels=["insert"], route_clock=True, std_by_label="eval_state/phase/id", std_label_names=NAMES)
    DDPGCfg(**base).validate(three)
    with pytest.raises(ValueError, match="state_preprocessor"):
        SACCfg(**base, state_preprocessor="running").validate(three)
    with pytest.raises(ValueError, match="periodic_reset_enabled"):
        SACCfg(**base, state_preprocessor="none", periodic_reset_enabled=True, periodic_reset_frequency=10).validate(three)
    SACCfg(**base, state_preprocessor="running").validate(SimpleNamespace(experiment=SimpleNamespace(num_agents=1)))

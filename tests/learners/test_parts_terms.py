"""PARTS' actor terms (sac.success_clone_weight / failure_anchor_weight / actor_q_scale, arXiv:2609.21788 eq. 4) and
sac.executed_action_mask."""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner


def step(learner, t, *, done=None, success=None, actions=None, infos_extra=None):
    n = learner.num_envs
    obs = torch.randn(n, OBS_DIM)
    acts = actions if actions is not None else torch.rand(n, ACT_DIM) * 2 - 1
    terminated = torch.zeros(n, 1, dtype=torch.bool) if done is None else done.reshape(n, 1)
    infos = {"episode_metrics_to_log": {"episode/success": torch.zeros(n) if success is None else success}}
    infos.update(infos_extra or {})
    learner.record_transition(observations=obs, states=None, actions=acts, rewards=torch.zeros(n, 1),
                              next_observations=torch.randn(n, OBS_DIM), next_states=None, terminated=terminated,
                              truncated=torch.zeros(n, 1, dtype=torch.bool), infos=infos, timestep=t, timesteps=100)


def test_each_transition_gets_its_episodes_outcome_when_it_ends():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8, success_clone_weight=1.0, failure_anchor_weight=1.0)
    for t in range(3):
        step(learner, t)
    ids_before = learner._env_episode.clone()
    assert torch.all(learner._outcomes[ids_before] == -1.0)  # still running
    step(learner, 3, done=torch.tensor([True, True]), success=torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(learner._outcomes[ids_before], torch.tensor([1.0, 0.0]))
    assert not torch.equal(learner._env_episode, ids_before)  # new episodes
    stored = learner.memory.get_tensor_by_name("episode_ids").reshape(-1)
    written = stored[torch.isfinite(stored) & (stored >= 0) & (stored < 1e6)]
    # 4 steps x 2 envs, every row tagged with its episode's id
    assert written.numel() == 8 and set(written.long().tolist()) == set(ids_before.tolist())


def test_the_clone_and_anchor_terms_pull_toward_the_successes_actions_and_toward_zero():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8, success_clone_weight=1.0,
                            failure_anchor_weight=1.0, normalize_observations=False)
    learner._outcomes[:4] = torch.tensor([1.0, 0.0, -1.0, 1.0])
    sampled = {"episode_ids": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
               "actions": torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [-0.5, 0.0]])}
    mean = torch.tensor([[0.0, 0.0], [0.2, -0.2], [0.9, 0.9], [-0.5, 0.0]], requires_grad=True)
    loss, metrics = learner._parts_terms(sampled, mean, rows=4)
    # B+ rows 0 and 3: ((0.25 + 0.25) + 0) / 2; B- row 1: 0.04 + 0.04; row 2 (running) in neither
    torch.testing.assert_close(metrics["actor/clone_loss"], torch.tensor([0.25]))
    torch.testing.assert_close(metrics["actor/anchor_loss"], torch.tensor([0.08]))
    torch.testing.assert_close(loss, torch.tensor(0.33))
    loss.backward()
    assert torch.all(mean.grad[2] == 0)  # the running episode's row pulls nothing


def test_the_actor_trains_with_the_terms_and_reports_them():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2, rollout=8, success_clone_weight=1.0,
                            failure_anchor_weight=1.0, actor_q_scale=2.5)
    seen = []
    learner.on_log.append(lambda agent, metrics, step_: seen.append(metrics))
    for t in range(6):
        step(learner, t, done=torch.tensor([t == 3] * 4), success=torch.tensor([1.0, 0.0, 1.0, 0.0]))
    learner.update(timestep=6, timesteps=100)
    keys = {k for m in seen for k in m}
    assert {"actor/clone_loss", "actor/anchor_loss", "actor/success_rows", "actor/failure_rows"} <= keys


def test_the_executed_action_mask_zeroes_the_stored_action():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=4, executed_action_mask="residual_active")
    acts = torch.tensor([[0.3, -0.4], [0.7, 0.1]])
    step(learner, 0, actions=acts, infos_extra={"residual_active": torch.tensor([0.0, 1.0])})
    batch = learner.memory.sample(names=["actions"], batch_size=64)[0][0]  # every row is one of the two
    rows = {tuple(round(v, 4) for v in row.tolist()) for row in batch}
    assert rows == {(0.0, 0.0), (0.7, 0.1)}
    with pytest.raises(RuntimeError, match="residual_active"):
        step(learner, 1, actions=acts)


def test_the_terms_refuse_demo_transitions():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="episode outcomes"):
        SACCfg(success_clone_weight=1.0, demo_transitions="x.pt", demo_sampling_end=10).validate(None)



def test_add_rows_writes_only_the_kept_envs_each_agent_at_its_own_pace():
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory

    memory = MultiRandomMemory(capacity=6, num_envs=4, num_agents=2)
    memory.create_tensor(name="x", size=1, dtype=torch.float32)
    memory.add_rows(torch.tensor([True, False, False, False]), x=torch.arange(4.0).unsqueeze(-1))
    assert memory.size == 0  # agent 1 holds nothing yet
    memory.add_rows(torch.tensor([True, True, False, True]), x=torch.arange(10.0, 14.0).unsqueeze(-1))
    assert memory._agent_size == [3, 1] and memory.size == 1
    x = memory.sample(names=["x"], batch_size=200)[0][0].reshape(-1)
    assert set(x[:200].tolist()) == {0.0, 10.0, 11.0} and set(x[200:].tolist()) == {13.0}
    with pytest.raises(RuntimeError, match="add_rows"):
        memory.add_samples(x=torch.zeros(4, 1))


def test_store_labels_keeps_only_transitions_starting_in_a_listed_phase():
    learner = build_learner("sac", num_agents=1, envs_per_agent=3, rollout=8, std_by_label="eval_state/phase/id",
                            std_label_names=["corner", "grasp", "insert", "screw"], store_labels=["insert", "screw"])
    phases = [torch.tensor([0.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 5.0])]
    for t, phase in enumerate(phases):
        # this step's infos give the phase of the NEXT observation; step 0 starts with no label: nothing stored
        step(learner, t, actions=torch.full((3, ACT_DIM), float(t)), infos_extra={"eval_state": {"phase/id": phase}})
    # stored: step 1 from phases [0, 2, 3] -> envs 1, 2; step 2 from [1, 2, 3] -> envs 1, 2
    assert learner.memory.size == 4
    acts = learner._sample(100)["actions"][:, 0]
    assert set(acts.tolist()) == {1.0, 2.0}
    learner.update(timestep=3, timesteps=100)


def test_a_time_limit_is_terminal_when_not_bootstrapped():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=4, bootstrap_time_limits=False,
                            backup_entropy=False, normalize_observations=False)
    n = 2
    sampled = {"observations": torch.randn(n, OBS_DIM), "next_observations": torch.randn(n, OBS_DIM),
               "actions": torch.zeros(n, ACT_DIM), "rewards": torch.tensor([[1.0], [2.0]]),
               "terminated": torch.tensor([[False], [False]]), "truncated": torch.tensor([[True], [False]])}
    inputs, nxt = {"observations": sampled["observations"]}, {"observations": sampled["next_observations"]}
    _, _, _, target, _, _ = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=nxt,
                                                         critic_inputs=inputs, critic_next_inputs=nxt, rows=n)
    assert float(target[0]) == 1.0  # the time limit: r alone
    assert float(target[1]) != 2.0  # a running episode still bootstraps
    learner.cfg.bootstrap_time_limits = True
    _, _, _, target, _, _ = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=nxt,
                                                         critic_inputs=inputs, critic_next_inputs=nxt, rows=n)
    assert float(target[0]) != 1.0  # the default bootstraps through it


def test_store_labels_are_validated():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="std_by_label"):
        SACCfg(store_labels=["insert"]).validate(None)
    with pytest.raises(ValueError, match="not in std_label_names"):
        SACCfg(std_by_label="x", std_label_names=["a"], store_labels=["b"]).validate(None)

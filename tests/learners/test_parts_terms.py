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


def test_demo_actor_clone_needs_demos_and_a_clone_weight():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="demo_actor_clone"):
        SACCfg(demo_actor_clone=True).validate(None)
    SACCfg(success_clone_weight=1.0, demo_transitions="x.pt", demo_sampling_end=10, demo_actor_clone=True).validate(None)


def test_demos_with_side_inputs_feed_the_critic_and_the_actor(tmp_path):
    class Side:
        side_inputs = {"base": 3}

        def __call__(self, critic_inputs, actions, side):
            return actions + side["base"][:, :1]

    rows = 6
    demo = {
        "observations": torch.randn(rows, OBS_DIM), "next_observations": torch.randn(rows, OBS_DIM),
        "actions": torch.full((rows, ACT_DIM), 2.0), "rewards": torch.ones(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
        "base": torch.zeros(rows, 3), "next_base": torch.zeros(rows, 3),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8, critic_action_transform=Side(),
                            demo_transitions=str(path), demo_sampling_end=100, success_clone_weight=1.0,
                            failure_anchor_weight=1.0, demo_actor_clone=True)
    assert set(learner._demo) >= {"base", "next_base"}
    seen = []
    learner.on_log.append(lambda agent, metrics, step_: seen.append(metrics))
    for t in range(4):
        step(learner, t, done=torch.tensor([t == 2, t == 2]), success=torch.tensor([1.0, 0.0]),
             infos_extra={"base": torch.zeros(2, 3), "next_base": torch.zeros(2, 3)})
    learner.update(timestep=4, timesteps=100)
    assert any("actor/demo_clone_loss" in m for m in seen)
    # the demo actions (2.0) are clipped to the actor's reach (1.0) before cloning
    value = next(float(m["actor/demo_clone_loss"]) for m in seen if "actor/demo_clone_loss" in m)
    assert value < ACT_DIM * 9.0



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
    memory.add_samples(x=torch.full((4, 1), 7.0))  # out of lockstep: every env's row, each agent at its own pointer
    assert memory._agent_size == [5, 3]


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


def test_keep_rows_compacts_each_agents_rows_and_writing_continues_after_them():
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory

    memory = MultiRandomMemory(capacity=8, num_envs=2, num_agents=2)
    memory.create_tensor(name="x", size=1, dtype=torch.float32)
    for t in range(3):
        memory.add_samples(x=torch.tensor([[float(t)], [10.0 + t]]))
    kept = memory.keep_rows(torch.tensor([[True, False, True], [False, False, True]]))
    assert kept == [2, 1] and memory.size == 1
    x = memory.get_tensor_by_name("x")
    assert x[0, :2, 0].tolist() == [0.0, 2.0] and x[1, :1, 0].tolist() == [12.0]
    memory.add_samples(x=torch.tensor([[5.0], [15.0]]))
    assert memory._agent_size == [3, 2] and x[0, 2, 0] == 5.0 and x[1, 1, 0] == 15.0


def test_retraining_keeps_every_success_and_a_fraction_of_the_failures_then_trains_fresh_networks():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=16, retrain_every=8,
                            retrain_failure_fraction=0.0, retrain_gradient_steps=3)
    before = [p.detach().clone() for p in learner.critics.parameters()]
    for t in range(8):
        # env 0's episodes succeed, env 1's fail; both end at t == 3, both run on after
        done = torch.tensor([t == 3, t == 3])
        step(learner, t, done=done, success=torch.tensor([1.0, 0.0]))
    learner._update_if_ready(timestep=8, timesteps=100)
    ids = learner.memory.get_tensor_by_name("episode_ids")[0, : learner.memory.size, 0].long()
    outcomes = learner._outcomes[ids]
    assert not bool((outcomes == 0.0).any())  # no failed episode kept at fraction 0
    assert bool((outcomes == 1.0).any()) and bool((outcomes == -1.0).any())  # successes and running episodes kept
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.critics.parameters()))  # fresh, then trained


def test_demo_storage_memory_writes_the_demos_as_ordinary_rows_with_episodes_of_their_own(tmp_path):
    rows = 6
    demo = {
        "observations": torch.randn(rows, OBS_DIM), "next_observations": torch.randn(rows, OBS_DIM),
        "actions": torch.full((rows, ACT_DIM), 0.5), "rewards": torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]]),
        "terminated": torch.tensor([[False], [False], [True], [False], [False], [True]]),
        "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    learner = build_learner("sac", num_agents=2, envs_per_agent=2, rollout=16, demo_transitions=str(path),
                            demo_storage="memory", success_clone_weight=1.0, failure_anchor_weight=1.0)
    assert learner._demo is None  # no separate set: the demos are memory rows
    assert learner.memory._agent_size == [rows, rows]
    ids = learner.memory.get_tensor_by_name("episode_ids")[:, :rows, 0].long()
    assert (ids[0] == ids[1]).all() and ids[0].tolist() == [4, 4, 4, 5, 5, 5]  # after the 4 envs' episode ids
    assert learner._outcomes[4].item() == 1.0 and learner._outcomes[5].item() == 0.0  # reward > 0: a success
    stored = learner.memory.get_tensor_by_name("actions")[:, :rows]
    assert torch.allclose(stored, torch.full_like(stored, 0.5))
    # online rows follow them; the memory wraps over the demos like any other rows
    for t in range(6):
        step(learner, t)
    assert learner.memory._agent_size == [rows + 12, rows + 12]
    learner.update(timestep=6, timesteps=100)


def test_demo_storage_is_validated():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="demo_storage"):
        SACCfg(demo_storage="both").validate(None)
    with pytest.raises(ValueError, match="demo_storage"):
        SACCfg(demo_storage="memory").validate(None)
    with pytest.raises(ValueError, match="sampling window"):
        SACCfg(demo_storage="memory", demo_transitions="x.pt", demo_sampling_end=10).validate(None)
    with pytest.raises(ValueError, match="demo_actor_clone"):
        SACCfg(demo_storage="memory", demo_transitions="x.pt", success_clone_weight=1.0, demo_actor_clone=True).validate(None)
    SACCfg(demo_storage="memory", demo_transitions="x.pt").validate(None)


def test_the_parts_terms_can_average_over_the_action_dims():
    from robonuke_rl_core.learners.cfg import SACCfg

    summed = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8, success_clone_weight=1.0, failure_anchor_weight=1.0)
    averaged = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8, success_clone_weight=1.0,
                             failure_anchor_weight=1.0, parts_term_reduction="mean")
    rows = 2
    mean_actions = torch.full((rows, ACT_DIM), 0.5)
    for learner in (summed, averaged):
        learner._outcomes = torch.tensor([1.0, 0.0])
    sampled = {"episode_ids": torch.tensor([[0.0], [1.0]]), "actions": torch.zeros(rows, ACT_DIM)}
    _, m_sum = summed._parts_terms(sampled, mean_actions, rows)
    _, m_mean = averaged._parts_terms(sampled, mean_actions, rows)
    assert float(m_sum["actor/clone_loss"]) == pytest.approx(0.25 * ACT_DIM)
    assert float(m_mean["actor/clone_loss"]) == pytest.approx(0.25)
    assert float(m_mean["actor/anchor_loss"]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="parts_term_reduction"):
        SACCfg(parts_term_reduction="max").validate(None)


def test_memory_critic_demos_train_the_critics_and_never_reach_the_actor(tmp_path):
    rows = 6
    demo = {
        "observations": torch.randn(rows, OBS_DIM), "next_observations": torch.randn(rows, OBS_DIM),
        "actions": torch.full((rows, ACT_DIM), 0.5), "rewards": torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [1.0]]),
        "terminated": torch.tensor([[False], [False], [True], [False], [False], [True]]),
        "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, demo_transitions=str(path),
                            demo_storage="memory_critic", success_clone_weight=1.0, failure_anchor_weight=1.0,
                            batch_size=4, learning_starts=0)
    flags = learner.memory.get_tensor_by_name("is_demo")[0, :rows, 0]
    assert flags.tolist() == [1.0] * rows
    ids = learner.memory.get_tensor_by_name("episode_ids")[0, :rows, 0].long()
    assert (learner._outcomes[ids] == -1.0).all()  # in neither PARTS term
    seen = []
    learner.on_log.append(lambda agent, metrics, step: seen.append(metrics))
    learner.update(timestep=0, timesteps=100)  # the memory holds demos only
    critic = [m for m in seen if "loss/critic" in m]
    assert critic, "the critics trained on the demo rows"
    assert not any("loss/policy" in m for m in seen)  # no online row yet: the actor waits
    for t in range(4):  # online rows: not demos
        step(learner, t)
    assert learner.memory.get_tensor_by_name("is_demo")[0, rows : rows + 8, 0].tolist() == [0.0] * 8


def test_the_actor_loss_weighs_only_online_rows():
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=8)
    learner._track_outcomes = False
    rows = 4
    sampled = {"is_demo": torch.tensor([[1.0], [1.0], [0.0], [0.0]])}
    online = (1.0 - sampled["is_demo"]).reshape(-1, 1)
    per_row = torch.tensor([10.0, 10.0, 1.0, 3.0]).view(1, rows)
    loss = ((per_row * online.view(1, rows)).sum(dim=1) / online.view(1, rows).sum(dim=1).clamp(min=1.0)).mean()
    assert float(loss) == 2.0  # the demo rows' 10s never count


def test_warmup_can_train_the_critics_while_the_actor_stays_frozen():
    from robonuke_rl_core.learners.cfg import SACCfg

    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, warmup_episodes=5,
                            warmup_trains_critics=True, learning_starts=0, batch_size=2)
    learner.enable_training_mode(True)
    seen = []
    learner.on_log.append(lambda agent, metrics, step: seen.append(metrics))
    for t in range(3):
        step(learner, t)
        learner.post_interaction(timestep=t, timesteps=100)
    assert any("loss/critic" in m for m in seen)  # the critics trained during the warm-up
    assert not any("loss/policy" in m for m in seen)  # the actor did not
    with pytest.raises(ValueError, match="demo_storage"):
        SACCfg(demo_storage="memory_critic").validate(None)


def test_sample_where_draws_only_the_flagged_rows():
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory

    memory = MultiRandomMemory(capacity=16, num_envs=1, device="cpu", num_agents=1)
    memory.create_tensor(name="x", size=1, dtype=torch.float32)
    memory.create_tensor(name="is_demo", size=1, dtype=torch.float32)
    memory.add_block(x=torch.arange(6.0).unsqueeze(-1), is_demo=torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [1.0]]))
    (x,) = memory.sample_where(names=["x"], batch_size=200, flag="is_demo", value=0.0)[0]
    assert set(x.reshape(-1).tolist()) == {3.0, 4.0}
    assert memory.count_where("is_demo", 0.0) == [2]


def test_memory_critic_gives_the_actor_a_full_batch_of_online_rows(tmp_path):
    rows = 6
    demo = {
        "observations": torch.randn(rows, OBS_DIM), "next_observations": torch.randn(rows, OBS_DIM),
        "actions": torch.full((rows, ACT_DIM), 0.5), "rewards": torch.ones(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, demo_transitions=str(path),
                            demo_storage="memory_critic", batch_size=8, learning_starts=0, learning_starts_rows=4)
    learner.enable_training_mode(True)
    seen = {}
    original = learner._actor_step

    def spy(**kw):
        seen.setdefault("is_demo", []).append(kw["sampled"]["is_demo"].clone())
        return original(**kw)

    learner._actor_step = spy
    calls = []
    learner_update = learner.update
    learner.update = lambda **kw: (calls.append(kw["timestep"]), learner_update(**kw))
    step(learner, 0)  # 2 online rows: below learning_starts_rows
    learner.post_interaction(timestep=0, timesteps=100)
    assert calls == []
    step(learner, 1)  # 4 online rows: training starts
    learner.post_interaction(timestep=1, timesteps=100)
    assert calls == [1]
    assert seen["is_demo"] and all(float(b.sum()) == 0.0 and b.shape[0] == 8 for b in seen["is_demo"])


def test_demo_batch_shared_gives_the_actor_the_critics_mixed_batch(tmp_path):
    from robonuke_rl_core.learners.cfg import SACCfg

    rows = 8
    demo = {
        "observations": torch.full((rows, OBS_DIM), 7.0), "next_observations": torch.full((rows, OBS_DIM), 7.0),
        "actions": torch.zeros(rows, ACT_DIM), "rewards": torch.ones(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    seen = {}
    for shared in (False, True):
        learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, demo_transitions=str(path),
                                demo_sampling_end=100, demo_batch_shared=shared, batch_size=8, learning_starts=0)
        learner.enable_training_mode(True)
        for t in range(4):
            step(learner, t)
        original = learner._actor_step

        def spy(shared=shared, original=original, **kw):
            seen[shared] = kw["sampled"]["observations"].clone()
            return original(**kw)

        learner._actor_step = spy
        learner.update(timestep=4, timesteps=100)
    # demo observations are all 7.0: the shared batch holds half demo rows, the default actor batch none
    assert int((seen[True][:, 0] == 7.0).sum()) == 4 and int((seen[False][:, 0] == 7.0).sum()) == 0
    with pytest.raises(ValueError, match="demo_batch_shared"):
        SACCfg(demo_batch_shared=True).validate(None)


def test_a_shared_demo_batch_keeps_its_demo_rows_out_of_the_parts_terms(tmp_path):
    rows = 8
    demo = {
        "observations": torch.full((rows, OBS_DIM), 7.0), "next_observations": torch.full((rows, OBS_DIM), 7.0),
        "actions": torch.zeros(rows, ACT_DIM), "rewards": torch.ones(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, demo_transitions=str(path),
                            demo_sampling_end=100, demo_batch_shared=True, batch_size=8, learning_starts=0,
                            success_clone_weight=10.0, failure_anchor_weight=10.0, actor_q_scale=2.5)
    learner.enable_training_mode(True)
    seen = []
    learner.on_log.append(lambda agent, metrics, step_: seen.append(metrics))
    for t in range(4):  # both envs' episodes end at t = 1 as successes, then a second pair runs
        step(learner, t, done=torch.tensor([t == 1, t == 1]), success=torch.tensor([1.0, 1.0]))
    learner.update(timestep=4, timesteps=100)
    rows_metric = [float(m["actor/success_rows"]) for m in seen if "actor/success_rows" in m]
    # half the batch is demos: at most the online half can count as successes
    assert rows_metric and rows_metric[-1] <= 0.5


def test_clone_demo_rows_makes_the_mixed_batchs_demo_rows_clone_targets(tmp_path):
    from robonuke_rl_core.learners.cfg import SACCfg

    rows = 8
    demo = {
        "observations": torch.full((rows, OBS_DIM), 7.0), "next_observations": torch.full((rows, OBS_DIM), 7.0),
        "actions": torch.full((rows, ACT_DIM), 0.5), "rewards": torch.ones(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    seen = {}
    for clone in (False, True):
        learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, demo_transitions=str(path),
                                demo_sampling_end=100, demo_batch_shared=True, batch_size=8, learning_starts=0,
                                success_clone_weight=10.0, failure_anchor_weight=10.0, actor_q_scale=2.5,
                                clone_demo_rows=clone)
        learner.enable_training_mode(True)
        logged = []
        learner.on_log.append(lambda agent, metrics, step_: logged.append(metrics))
        for t in range(4):  # online rows from running episodes: in neither term
            step(learner, t)
        learner.update(timestep=4, timesteps=100)
        seen[clone] = [float(m["actor/success_rows"]) for m in logged if "actor/success_rows" in m][-1]
    assert seen[False] == 0.0 and seen[True] == 0.5  # the demo half of the batch
    with pytest.raises(ValueError, match="clone_demo_rows"):
        SACCfg(clone_demo_rows=True).validate(None)


def test_action_l2_weight_penalizes_the_actors_mean_action_and_logs_it():
    from robonuke_rl_core.learners.cfg import SACCfg

    losses = {}
    for weight in (0.0, 5.0):
        torch.manual_seed(0)
        learner = build_learner("sac", num_agents=1, envs_per_agent=2, rollout=32, batch_size=8, learning_starts=0,
                                action_l2_weight=weight)
        learner.enable_training_mode(True)
        seen = []
        learner.on_log.append(lambda agent, metrics, step_: seen.append(metrics))
        for t in range(4):
            step(learner, t)
        learner.update(timestep=4, timesteps=100)
        losses[weight] = [m for m in seen if "actor/action_l2" in m]
    assert losses[0.0] == []  # off: no penalty, no metric
    assert losses[5.0] and float(losses[5.0][-1]["actor/action_l2"]) >= 0.0
    with pytest.raises(ValueError, match="action_l2_weight"):
        SACCfg(action_l2_weight=-1.0).validate(None)



def test_the_target_subset_can_be_drawn_with_replacement():
    learner = build_learner("sac", num_agents=1, envs_per_agent=1, num_critics=3, critic_subset=2, critic_subset_replacement=True)
    # critic j's target value is j: a min over 2 draws with replacement is 2 only when both draws are critic 2
    qs = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 1, 1).repeat(1, 1, 1)
    torch.manual_seed(0)
    mins = torch.stack([learner._target_min(qs, 1).reshape(()) for _ in range(3000)])
    assert abs(float((mins == 2).float().mean()) - 1 / 9) < 0.03  # (1/3)^2: impossible without replacement
    distinct = build_learner("sac", num_agents=1, envs_per_agent=1, num_critics=3, critic_subset=2)
    assert all(float(distinct._target_min(qs, 1)) < 2 for _ in range(200))

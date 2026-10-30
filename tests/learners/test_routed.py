"""``sac.route_labels`` (learners/routed.py): one SAC per phase, routed per env; their local rewards, attempt ends,
storage and the routed eval policy."""

from __future__ import annotations

import copy

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner
from robonuke_rl_core.evaluation import _RoutedEvalPolicy
from robonuke_rl_core.learners.routed import RoutedSAC, route_hook

NAMES = ["corner", "grasp", "insert", "screw", "success"]


def routed_pair():
    kwargs = dict(num_agents=1, envs_per_agent=3, rollout=16, std_by_label="eval_state/phase/id", std_label_names=NAMES,
                  success_clone_weight=1.0, failure_anchor_weight=1.0)
    learners = {route: build_learner("sac", route_label=route, **kwargs) for route in ("insert", "screw")}
    return RoutedSAC(learners, NAMES)


def step(routed, t, phases_next, done=None, actions=None):
    n = routed.num_envs
    obs = torch.randn(n, OBS_DIM)
    acts = actions if actions is not None else torch.zeros(n, ACT_DIM)
    terminated = torch.zeros(n, 1, dtype=torch.bool) if done is None else done.reshape(n, 1)
    routed.record_transition(
        observations=obs, states=None, actions=acts, rewards=torch.full((n, 1), 7.0), next_observations=obs,
        next_states=None, terminated=terminated, truncated=torch.zeros(n, 1, dtype=torch.bool),
        infos={"eval_state": {"phase/id": torch.tensor(phases_next, dtype=torch.float32)},
               "episode_metrics_to_log": {"episode/success": torch.zeros(n)}},
        timestep=t, timesteps=100,
    )


def test_each_env_acts_with_its_phases_residual_and_the_base_elsewhere():
    routed = routed_pair()
    routed.enable_training_mode(True)
    obs = torch.randn(3, OBS_DIM)
    actions, _ = routed.act(obs, None, timestep=0, timesteps=10)
    assert torch.all(actions == 0)  # no phase known yet: the base alone
    step(routed, 0, [1, 2, 3])  # env 0 in grasp, env 1 in insert, env 2 in screw
    torch.manual_seed(0)
    a_insert, _ = routed.learners["insert"].act(obs, None, timestep=1, timesteps=10)
    torch.manual_seed(0)
    a_screw, _ = routed.learners["screw"].act(obs, None, timestep=1, timesteps=10)
    torch.manual_seed(0)
    actions, _ = routed.act(obs, None, timestep=1, timesteps=10)
    assert torch.all(actions[0] == 0)
    torch.testing.assert_close(actions[1], a_insert[1])
    torch.manual_seed(1)


def test_each_residual_stores_only_its_phase_with_its_local_reward_and_attempt_end():
    routed = routed_pair()
    routed.enable_training_mode(True)
    step(routed, 0, [2, 2, 3])  # no label before this step: nothing stored; now envs 0, 1 in insert, env 2 in screw
    step(routed, 1, [2, 3, 4])  # env 1 passes insert (-> screw), env 2 succeeds (-> success code 4)
    step(routed, 2, [2, 3, 6])  # env 0 still inserting, env 1 screwing; env 2 sits at an end code (no residual)
    insert = routed.learners["insert"].memory
    screw = routed.learners["screw"].memory
    assert insert.size == 3 and screw.size == 2  # insert: envs 0, 1 then env 0; screw: env 2 then env 1
    i_rewards = insert.get_tensor_by_name("rewards")[0, :3, 0].tolist()
    s_rewards = screw.get_tensor_by_name("rewards")[0, :2, 0].tolist()
    assert sorted(i_rewards) == [0.0, 0.0, 1.0]  # only env 1's passing step pays; never the env's own reward 7
    assert sorted(s_rewards) == [0.0, 1.0]  # env 2's success
    i_term = insert.get_tensor_by_name("terminated")[0, :3, 0].tolist()
    assert sum(i_term) == 1  # the passed attempt ends; the ongoing ones do not


def test_learner_metrics_are_prefixed_by_phase_and_env_metrics_pass_once():
    seen = []
    hook = lambda agent, metrics, step: seen.append(dict(metrics))  # noqa: E731
    route_hook(hook, "insert", first=True)(0, {"loss/critic": 1.0, "episode/success": 0.5}, 0)
    route_hook(hook, "screw", first=False)(0, {"loss/critic": 2.0, "episode/success": 0.5}, 0)
    assert seen == [{"insert/loss/critic": 1.0, "episode/success": 0.5}, {"screw/loss/critic": 2.0}]


def test_the_routed_eval_policy_follows_the_phases_it_observes():
    calls = {"insert": torch.full((2, ACT_DIM), 1.0), "screw": torch.full((2, ACT_DIM), 2.0)}
    learner = build_learner("sac", num_agents=1, envs_per_agent=2, std_by_label="eval_state/phase/id", std_label_names=NAMES)
    parts = {route: ((lambda value: (lambda o, s: value))(calls[route]), learner) for route in calls}
    policy = _RoutedEvalPolicy(parts, NAMES, deterministic=True)
    assert torch.all(policy(torch.zeros(2, OBS_DIM), None) == 0)
    policy.observe({"eval_state": {"phase/id": torch.tensor([2.0, 3.0])}})
    out = policy(torch.zeros(2, OBS_DIM), None)
    assert torch.all(out[0] == 1.0) and torch.all(out[1] == 2.0)
    policy.reset()
    assert torch.all(policy(torch.zeros(2, OBS_DIM), None) == 0)


def _held_step(learner, t, phase_next, still, stalled=0.0, done=False):
    n = learner.num_envs
    obs = torch.full((n, OBS_DIM), float(t))
    learner.record_transition(
        observations=obs, states=None, actions=torch.zeros(n, ACT_DIM), rewards=torch.zeros(n, 1),
        next_observations=obs + 1, next_states=None, terminated=torch.full((n, 1), done),
        truncated=torch.zeros(n, 1, dtype=torch.bool),
        infos={"eval_state": {"phase/id": torch.full((n,), float(phase_next)), "phase/still": torch.full((n,), float(still)),
                              "phase/stalled": torch.full((n,), float(stalled))},
               "episode_metrics_to_log": {"episode/success": torch.zeros(n)}},
        timestep=t, timesteps=100,
    )


def _held_learner(**extra):
    return build_learner("sac", num_agents=1, envs_per_agent=1, rollout=64, std_by_label="eval_state/phase/id",
                         std_label_names=NAMES, route_label="insert", hold_still_key="eval_state/phase/still",
                         hold_stall_key="eval_state/phase/stalled", **extra)


def test_still_transitions_are_held_until_progress_then_written_in_order():
    learner = _held_learner()
    learner.enable_training_mode(True)
    _held_step(learner, 0, 2, still=0)  # the first label: nothing stored before it
    _held_step(learner, 1, 2, still=0)
    _held_step(learner, 2, 2, still=1)
    _held_step(learner, 3, 2, still=1)
    assert learner.memory._agent_size == [1]  # the two still steps are held
    _held_step(learner, 4, 2, still=0)  # progress: both written, then this one
    assert learner.memory._agent_size == [4]
    obs = learner.memory.get_tensor_by_name("observations")[0, :4, 0].tolist()
    assert obs == [1.0, 2.0, 3.0, 4.0]


def test_a_stall_keeps_only_the_first_held_transition_as_a_terminal_failure():
    learner = _held_learner()
    learner.enable_training_mode(True)
    _held_step(learner, 0, 2, still=0)
    _held_step(learner, 1, 2, still=0)
    for t in (2, 3, 4):
        _held_step(learner, t, 2, still=1)
    _held_step(learner, 5, 5, still=1, stalled=1, done=True)  # the episode ends for lack of progress
    assert learner.memory._agent_size == [2]
    obs = learner.memory.get_tensor_by_name("observations")[0, :2, 0].tolist()
    terminated = learner.memory.get_tensor_by_name("terminated")[0, :2, 0].tolist()
    rewards = learner.memory.get_tensor_by_name("rewards")[0, :2, 0].tolist()
    assert obs == [1.0, 2.0] and terminated == [False, True] and rewards == [0.0, 0.0]


def test_the_route_clock_updates_only_on_the_residuals_own_steps():
    learner = _held_learner(route_clock=True, learning_starts=0, batch_size=2)
    learner.enable_training_mode(True)
    updates = []
    learner.update = lambda **kw: updates.append(kw["timestep"])
    _held_step(learner, 0, 2, still=0)
    learner.post_interaction(timestep=0, timesteps=100)  # no label before this step: not its own
    for t, phase in ((1, 2), (2, 3), (3, 3), (4, 3)):
        _held_step(learner, t, phase, still=0)  # a step starts in the label the step before published
        learner.post_interaction(timestep=t, timesteps=100)
    assert updates == [1, 2] and learner._route_steps == 2  # t = 1, 2 start in insert; 3, 4 in screw


def test_reference_dropout_zeroes_the_last_columns_of_some_rows():
    learner = build_learner("sac", num_agents=1, envs_per_agent=1, rollout=8, reference_dropout=0.5, reference_columns=2)
    torch.manual_seed(0)
    obs = torch.ones(1000, OBS_DIM)
    out = learner._drop_reference(obs)
    dropped = out[:, -2:].sum(dim=1) == 0
    assert 0.4 < dropped.float().mean() < 0.6
    assert torch.equal(out[:, :-2], obs[:, :-2])
    assert torch.equal(out[~dropped], obs[~dropped])


def test_holding_works_through_an_n_step_window():
    learner = _held_learner(n_step=3)
    learner.enable_training_mode(True)
    _held_step(learner, 0, 2, still=0)
    _held_step(learner, 1, 2, still=0)
    _held_step(learner, 2, 2, still=0)
    for t in (3, 4, 5):
        _held_step(learner, t, 2, still=1)
    _held_step(learner, 6, 5, still=1, stalled=1, done=True)
    for t in (7, 8, 9):  # the next episode: the window drains the stall's later rows, all dropped
        _held_step(learner, t, 0, still=0)
    obs = learner.memory.get_tensor_by_name("observations")[0, : learner.memory._agent_size[0], 0].tolist()
    terminated = learner.memory.get_tensor_by_name("terminated")[0, : learner.memory._agent_size[0], 0].tolist()
    # rows 1, 2 (row 0 has no label), then the first still row (3) as the terminal failure; 4, 5, 6 dropped;
    # the new episode's rows start in corner, not this residual's
    assert obs == [1.0, 2.0, 3.0] and terminated == [False, False, True]


def test_a_routed_residual_logs_its_steps_memory_and_updates():
    learner = _held_learner(route_clock=True, learning_starts=0, batch_size=2)
    learner.enable_training_mode(True)
    seen = []
    learner.on_log.append(lambda agent, metrics, step: seen.append(metrics))
    for t in range(4):
        _held_step(learner, t, 2, still=0)
        learner.post_interaction(timestep=t, timesteps=100)
    logged = [m for m in seen if "route/steps" in m]
    assert logged, "no route/steps metric"
    last = logged[-1]
    assert float(last["route/steps"]) == 3.0  # steps 1, 2, 3 start in insert
    assert float(last["route/memory_rows"]) == 3.0 and float(last["route/updates"]) >= 1.0


def test_warmup_episodes_run_on_zero_actions_with_no_updates_then_learning_starts():
    learner = _held_learner(warmup_episodes=2, learning_starts=0, batch_size=2)
    learner.enable_training_mode(True)
    updates = []
    learner.update = lambda **kw: updates.append(kw["timestep"])
    obs = torch.randn(1, OBS_DIM)
    t = 0
    for done in (False, False, True, False, True):  # two episodes end
        actions, _ = learner.act(obs, None, timestep=t, timesteps=100)
        assert torch.equal(actions, torch.zeros_like(actions))
        _held_step(learner, t, 2, still=0, done=done)
        learner.post_interaction(timestep=t, timesteps=100)
        t += 1
    assert updates == [4]  # nothing trained during the base's episodes; the first update once the 2nd has ended
    assert learner.memory._agent_size[0] > 0  # their transitions are stored
    actions, _ = learner.act(obs, None, timestep=t, timesteps=100)
    assert not torch.equal(actions, torch.zeros_like(actions))  # the policy acts (with its noise) from here
    _held_step(learner, t, 2, still=0)
    learner.post_interaction(timestep=t, timesteps=100)
    assert updates == [4, t]


def _shaped_step(learner, t, phase_next, tip, done=False):
    n = learner.num_envs
    obs = torch.full((n, OBS_DIM), float(t))
    learner.record_transition(
        observations=obs, states=None, actions=torch.zeros(n, ACT_DIM), rewards=torch.zeros(n, 1),
        next_observations=obs + 1, next_states=None, terminated=torch.full((n, 1), done),
        truncated=torch.zeros(n, 1, dtype=torch.bool),
        infos={"eval_state": {"phase/id": torch.full((n,), float(phase_next)), "phase/tip_mm": torch.full((n,), float(tip))}},
        timestep=t, timesteps=100,
    )


def PT(route, channel, weight):
    from robonuke_rl_core.learners.cfg import PotentialTermCfg

    return PotentialTermCfg(route=route, channel=channel, weight=weight)


def _shaped_learner(**extra):
    return build_learner("sac", num_agents=1, envs_per_agent=1, rollout=64, std_by_label="eval_state/phase/id",
                         std_label_names=NAMES, route_label="insert", discount_factor=0.9,
                         potential_terms=[PT("insert", "eval_state/phase/tip_mm", 0.1), PT("screw", "eval_state/phase/x", 1.0)],
                         **extra)


def test_potential_shaping_adds_gamma_phi_next_minus_phi_to_the_local_reward():
    learner = _shaped_learner()
    learner.enable_training_mode(True)
    _shaped_step(learner, 0, 2, tip=30.0)  # the first label: Phi(s) = -3, nothing stored
    _shaped_step(learner, 1, 2, tip=20.0)  # 0.9 * -2 - -3 = 1.2
    _shaped_step(learner, 2, 3, tip=1.0)  # passes insert: 1 + 0.9 * -0.1 - -2 = 2.91 (Phi(s') kept on the last step)
    rewards = learner.memory.get_tensor_by_name("rewards")[0, :2, 0].tolist()
    assert rewards == pytest.approx([1.2, 2.91])


def test_no_shaping_across_an_episode_end_and_a_missing_or_nan_channel_fails():
    learner = _shaped_learner()
    learner.enable_training_mode(True)
    _shaped_step(learner, 0, 2, tip=30.0)
    _shaped_step(learner, 1, 2, tip=50.0, done=True)  # ends (timeout) far off: 0.9 * -5 - -3 = -1.5
    _shaped_step(learner, 2, 2, tip=10.0)  # starts from the new episode's first observation: Phi(s) unknown, 0
    rewards = learner.memory.get_tensor_by_name("rewards")[0, :2, 0].tolist()
    assert rewards == pytest.approx([-1.5, 0.0])
    with pytest.raises(ValueError, match="not finite"):
        _shaped_step(learner, 3, 2, tip=float("nan"))
    with pytest.raises(RuntimeError, match="phase/tip_mm"):
        learner.record_transition(
            observations=torch.zeros(1, OBS_DIM), states=None, actions=torch.zeros(1, ACT_DIM), rewards=torch.zeros(1, 1),
            next_observations=torch.zeros(1, OBS_DIM), next_states=None, terminated=torch.zeros(1, 1, dtype=torch.bool),
            truncated=torch.zeros(1, 1, dtype=torch.bool), infos={"eval_state": {"phase/id": torch.full((1,), 2.0)}},
            timestep=4, timesteps=100,
        )


def test_potential_terms_routes_are_route_labels_for_a_routed_sac_and_star_otherwise():
    from robonuke_rl_core.learners.cfg import SACCfg

    def cfg(**kw):
        c = SACCfg(std_by_label="eval_state/phase/id", std_label_names=NAMES, **kw)
        c.validate(None)

    cfg(route_labels=["insert"], potential_terms=[PT("insert", "a", 1.0)])
    cfg(potential_terms=[PT("*", "a", 1.0)])
    with pytest.raises(ValueError, match="'\\*' is an unrouted"):
        cfg(route_labels=["insert"], potential_terms=[PT("*", "a", 1.0)])
    with pytest.raises(ValueError, match="std_label_names"):
        cfg(route_labels=["insert"], potential_terms=[PT("bogus", "a", 1.0)])
    with pytest.raises(ValueError, match="set route_labels"):
        cfg(potential_terms=[PT("insert", "a", 1.0)])
    with pytest.raises(ValueError, match="twice"):
        cfg(route_labels=["insert"], potential_terms=[PT("insert", "a", 1.0), PT("insert", "a", 2.0)])


def test_a_demo_file_must_carry_the_learners_shaping_and_its_own_success_flags(tmp_path):
    rows = 4
    demo = {
        "observations": torch.randn(rows, OBS_DIM), "next_observations": torch.randn(rows, OBS_DIM),
        "actions": torch.zeros(rows, ACT_DIM), "rewards": torch.tensor([[0.3], [0.2], [-0.1], [0.4]]),
        "terminated": torch.tensor([[False], [True], [False], [True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
        "success": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
    }
    terms = {"eval_state/phase/tip_mm": 0.1}
    good = tmp_path / "good.pt"
    torch.save({"transitions": demo, "meta": {"potential_terms": terms, "discount_factor": 0.9}}, good)
    learner = _shaped_learner(demo_transitions=str(good), demo_storage="memory", success_clone_weight=1.0)
    # the second episode's rewards are positive, but only the first succeeded
    assert learner._demo_episode_success.tolist() == [1.0, 0.0]
    stale = tmp_path / "stale.pt"
    torch.save({"transitions": demo, "meta": {"potential_terms": {}, "discount_factor": 0.9}}, stale)
    with pytest.raises(ValueError, match="remake it"):
        _shaped_learner(demo_transitions=str(stale), demo_storage="memory", success_clone_weight=1.0)
    no_flags = tmp_path / "no_flags.pt"
    torch.save({"transitions": {k: v for k, v in demo.items() if k != "success"},
                "meta": {"potential_terms": terms, "discount_factor": 0.9}}, no_flags)
    with pytest.raises(KeyError, match="success"):
        _shaped_learner(demo_transitions=str(no_flags), demo_storage="memory", success_clone_weight=1.0)

"""SAC beyond the classic twin: the Q ensemble, the delayed / late-starting actor, and demo
transitions mixed into the critic's batches. Each is off by default; the default path's
equivalence with the old twin is what every other SAC test already pins.
"""

from __future__ import annotations

import copy

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner, fill_memory
from robonuke_rl_core.learners.cfg import SACCfg

from test_independence import UNTOUCHED, AGENT_UNDER_TEST, assert_same, make_extreme, snapshot

DEMO_REWARD = 1000.0  # marks a demo row in a sampled batch: online rewards are randn


def write_demos(path, rows: int = 32, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, **replace):
    transitions = {
        "observations": torch.randn(rows, obs_dim),
        "actions": torch.rand(rows, act_dim) * 2 - 1,
        "rewards": torch.full((rows, 1), DEMO_REWARD),
        "next_observations": torch.randn(rows, obs_dim),
        "terminated": torch.zeros(rows, 1, dtype=torch.bool),
        "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    transitions.update(replace)
    torch.save({"transitions": transitions, "meta": {}}, path)
    return str(path)


def weights(module, agent: int = 0):
    return {k: v.clone() for k, v in module.agent_state_dict(agent).items()}


def changed(before, module, agent: int = 0) -> bool:
    after = module.agent_state_dict(agent)
    return any(not torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------------- config rules
@pytest.mark.parametrize(
    "fields, match",
    [
        ({"num_critics": 1, "critic_subset": 1}, "num_critics must be >= 2"),
        ({"num_critics": 4, "critic_subset": 5}, "critic_subset"),
        ({"critic_subset": 0}, "critic_subset"),
        ({"actor_q_reduction": "max"}, "actor_q_reduction"),
        ({"policy_delay": 0}, "policy_delay must be >= 1"),
        ({"gradient_steps": 2, "policy_delay": 3}, "policy_delay"),
        ({"learning_starts": 100, "policy_learning_starts": 50}, "policy_learning_starts"),
        ({"demo_sampling_start": 10, "demo_sampling_end": 5, "demo_transitions": "x"}, "demo_sampling"),
        ({"demo_sampling_end": 100}, "names no file"),
        ({"demo_transitions": "x.pt"}, "window"),
        ({"demo_transitions": "x.pt", "demo_sampling_end": 10, "batch_size": 1}, "batch_size"),
        ({"entropy_observations": "previous"}, "entropy_observations"),
    ],
)
def test_each_rule_rejects_a_bad_value(fields, match):
    with pytest.raises(ValueError, match=match):
        SACCfg(**fields).validate(None)


def test_the_defaults_and_a_full_setup_pass():
    SACCfg().validate(None)
    SACCfg(
        num_critics=10,
        critic_subset=2,
        actor_q_reduction="mean",
        gradient_steps=10,
        policy_delay=10,
        learning_starts=0,
        policy_learning_starts=2240,
        demo_transitions="demos.pt",
        demo_sampling_start=0,
        demo_sampling_end=22400,
    ).validate(None)


# ---------------------------------------------------------------------- the ensemble
def test_the_ensemble_is_one_stacked_module_with_a_target_and_checkpoints_whole(tmp_path):
    learner = build_learner("sac", num_agents=2, num_critics=4, critic_subset=2)
    keys = learner._checkpoint_model_keys()
    assert keys == ["policy", "critics", "target_critics"]
    assert learner.critics.members == 4 and learner.target_critics.members == 4
    # every parameter is (agents, critics, ...): the agent stays the leading dimension
    for param in learner.critics.parameters():
        assert param.shape[:2] == (2, 4)
    assert_same(learner.critics.agent_state_dict(0), learner.target_critics.agent_state_dict(0))

    fill_memory(learner)
    learner.update(timestep=10, timesteps=100)
    path = tmp_path / "a.ckpt"
    torch.save(learner._build_checkpoint(1, 10), path)
    fresh = build_learner("sac", num_agents=2, num_critics=4, critic_subset=2)
    fresh.load_agent(path, slot=0)
    for key in keys:
        assert_same(getattr(fresh, key).agent_state_dict(0), getattr(learner, key).agent_state_dict(1), key)

    twin = build_learner("sac", num_agents=2)
    with pytest.raises(ValueError, match="num_critics"):
        twin.load_agent(path, slot=0)


def test_a_full_subset_takes_the_min_over_all_critics_and_draws_nothing():
    learner = build_learner("sac", num_agents=2, num_critics=3, critic_subset=3)
    qs = torch.randn(3, 2 * 5, 1)
    state = torch.random.get_rng_state()
    out = learner._target_min(qs, rows=5)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(out, qs.min(dim=0).values)


def test_a_subset_min_comes_from_one_subset_per_agent():
    learner = build_learner("sac", num_agents=3, num_critics=5, critic_subset=2)
    rows = 7
    qs = torch.randn(5, 3 * rows, 1)
    out = learner._target_min(qs, rows=rows).view(3, rows)
    stacked = qs.view(5, 3, rows)
    for agent in range(3):
        pair_mins = [
            torch.minimum(stacked[i, agent], stacked[j, agent])
            for i in range(5)
            for j in range(i + 1, 5)
        ]
        # one pair explains every row of the agent: the subset is drawn per agent, not per row
        assert any(torch.equal(out[agent], m) for m in pair_mins)
        assert torch.all(out[agent] >= stacked[:, agent].min(dim=0).values)


def test_the_mean_reduction_feeds_the_actor_the_ensemble_mean(monkeypatch):
    learner = build_learner("sac", num_agents=1, num_critics=3, critic_subset=2, actor_q_reduction="mean")
    fill_memory(learner)
    values = torch.stack([torch.full((4, 1), v) for v in (1.0, 2.0, 6.0)])
    monkeypatch.setattr(learner.critics, "act", lambda inputs, role="": (values, {}))
    inputs = {"observations": torch.randn(4, OBS_DIM)}
    _, _, _, q, _ = learner._compute_actor_loss(sampled=None, inputs=inputs, critic_inputs=inputs, rows=4)
    assert torch.equal(q, torch.full((4, 1), 3.0))


# ---------------------------------------------------------------------- entropy (SERL's options)
def _critic_target(learner, rows: int = 4):
    sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=rows)[0]))
    inputs = {"observations": sampled["observations"]}
    next_inputs = {"observations": sampled["next_observations"]}
    torch.manual_seed(0)  # the policy samples next_actions
    _, _, _, target, next_log_prob = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs,
        critic_next_inputs=next_inputs, rows=rows,
    )
    return target, next_log_prob


def test_without_backup_entropy_the_target_has_no_entropy_bonus():
    """The two targets differ by exactly gamma * alpha * log_pi(a'|s') on non-terminal rows."""
    learner = build_learner("sac", num_agents=1, normalize_observations=False, initial_entropy_value=0.5)
    fill_memory(learner)
    torch.manual_seed(1)
    with_bonus, log_prob = _critic_target(learner)
    learner.cfg.backup_entropy = False
    torch.manual_seed(1)
    without, _ = _critic_target(learner)
    torch.manual_seed(1)
    sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=4)[0]))
    alive = (sampled["terminated"] & sampled["truncated"].logical_not()).logical_not()
    expected = without - learner.cfg.discount_factor * alive * 0.5 * log_prob
    assert torch.allclose(with_bonus, expected, atol=1e-6)
    assert not torch.allclose(with_bonus, without)


@pytest.mark.parametrize("where", ["current", "next"])
def test_the_temperature_reads_the_entropy_entropy_observations_names(where):
    """next: the critic step's log-prob at the batch's next observations, not the actor's."""
    learner = build_learner("sac", num_agents=1, entropy_observations=where)
    fill_memory(learner)
    critic_loss, actor_step = learner._compute_critic_loss, learner._actor_step
    seen = {}

    def critic(**kw):
        out = critic_loss(**kw)
        seen["next"] = out[4]
        return out

    def actor(**kw):
        out = actor_step(**kw)
        seen["current"] = out["log_prob"]
        return out

    fed = []
    update = learner._update_entropy
    learner._compute_critic_loss, learner._actor_step = critic, actor
    learner._update_entropy = lambda log_prob, rows: (fed.append(log_prob.detach().clone()), update(log_prob, rows))[1]
    learner.update(timestep=10, timesteps=100)
    assert len(fed) == 1 and torch.equal(fed[0], seen[where].detach())
    assert not torch.equal(seen["current"], seen["next"].detach())


# ---------------------------------------------------------------------- the actor's schedule
def test_policy_delay_steps_the_actor_once_per_delay_and_the_critics_every_step():
    learner = build_learner("sac", num_agents=1, gradient_steps=4, policy_delay=4, learn_entropy=False)
    fill_memory(learner)
    seen = []
    original = learner._actor_step
    learner._actor_step = lambda **kw: (seen.append(learner._updates), original(**kw))[1]
    critic_steps = []
    step = learner.critic_optimizer.step
    learner.critic_optimizer.step = lambda *a, **k: (critic_steps.append(1), step(*a, **k))[1]
    learner.update(timestep=10, timesteps=100)
    learner.update(timestep=11, timesteps=100)
    assert seen == [3, 7]  # the last of each env step's four gradient steps
    assert len(critic_steps) == 8


def test_before_policy_learning_starts_only_the_critics_learn():
    learner = build_learner("sac", num_agents=1, learning_starts=0, policy_learning_starts=20)
    fill_memory(learner)
    policy, critic = weights(learner.policy), weights(learner.critics)
    alpha = learner._entropy_coefficient.clone()
    learner._update_if_ready(timestep=5, timesteps=100)
    assert not changed(policy, learner.policy)
    assert changed(critic, learner.critics)
    assert torch.equal(alpha, learner._entropy_coefficient)

    learner._update_if_ready(timestep=20, timesteps=100)
    assert changed(policy, learner.policy)


# ---------------------------------------------------------------------- demo transitions
def _demo_learner(tmp_path, **overrides):
    fields = dict(
        num_agents=2,
        rollout=8,
        batch_size=8,
        demo_transitions=write_demos(tmp_path / "demos.pt"),
        demo_sampling_start=0,
        demo_sampling_end=50,
    )
    fields.update(overrides)
    return build_learner("sac", **fields)


def _capture_batches(learner):
    critic_batches, actor_batches = [], []
    critic_loss, actor_loss = learner._compute_critic_loss, learner._compute_actor_loss
    learner._compute_critic_loss = lambda **kw: (critic_batches.append(kw["sampled"]), critic_loss(**kw))[1]
    learner._compute_actor_loss = lambda **kw: (actor_batches.append(kw["sampled"]), actor_loss(**kw))[1]
    return critic_batches, actor_batches


def _demo_rows(batch, agents: int, rows: int) -> list:
    rewards = batch["rewards"].view(agents, rows)
    return [int((rewards[a] == DEMO_REWARD).sum()) for a in range(agents)]


def test_inside_the_window_the_critic_batch_is_half_demo_and_the_actor_batch_online(tmp_path):
    learner = _demo_learner(tmp_path)
    fill_memory(learner)  # 8 rows per agent: at least half a batch
    critic_batches, actor_batches = _capture_batches(learner)
    learner.update(timestep=10, timesteps=100)
    assert _demo_rows(critic_batches[0], 2, 8) == [4, 4]
    assert _demo_rows(actor_batches[0], 2, 8) == [0, 0]


def test_while_the_memory_is_short_demos_fill_the_rest_of_the_critic_batch(tmp_path):
    learner = _demo_learner(tmp_path, policy_learning_starts=40)
    fill_memory(learner, steps=1)  # 2 rows per agent
    critic_batches, actor_batches = _capture_batches(learner)
    learner._update_if_ready(timestep=1, timesteps=100)
    assert _demo_rows(critic_batches[0], 2, 8) == [6, 6]
    assert actor_batches == []  # the actor has not started


def test_outside_the_window_no_demo_reaches_any_batch(tmp_path):
    learner = _demo_learner(tmp_path, demo_sampling_start=20, demo_sampling_end=30)
    fill_memory(learner)
    critic_batches, actor_batches = _capture_batches(learner)
    learner.update(timestep=10, timesteps=100)
    learner.update(timestep=30, timesteps=100)
    assert all(_demo_rows(b, 2, 8) == [0, 0] for b in critic_batches + actor_batches)
    learner.update(timestep=20, timesteps=100)
    assert _demo_rows(critic_batches[-1], 2, 8) == [4, 4]


@pytest.mark.parametrize(
    "replace, match",
    [
        ({"observations": torch.randn(32, OBS_DIM + 1)}, "'observations' must be"),
        ({"actions": torch.full((32, ACT_DIM), float("nan"))}, "non-finite"),
        ({"rewards": torch.zeros(31, 1)}, "rows"),
    ],
)
def test_a_demo_file_that_does_not_fit_the_agent_is_refused(tmp_path, replace, match):
    path = write_demos(tmp_path / "bad.pt", **replace)
    with pytest.raises(ValueError, match=match):
        _demo_learner(tmp_path, demo_transitions=path)


def test_demo_actions_beyond_the_policy_range_reach_the_critic_unchanged(tmp_path):
    """Only the critic sees demo rows, and it takes the action as an input: a correction the
    squashed actor cannot produce is still a true transition, never clipped."""
    actions = torch.full((32, ACT_DIM), 1.5)
    learner = _demo_learner(tmp_path, demo_transitions=write_demos(tmp_path / "wide.pt", actions=actions))
    fill_memory(learner)
    critic_batches, _ = _capture_batches(learner)
    learner.update(timestep=10, timesteps=100)
    rows = critic_batches[0]["rewards"].reshape(-1) == DEMO_REWARD
    assert torch.all(critic_batches[0]["actions"][rows] == 1.5)


def test_a_demo_file_missing_a_tensor_is_refused(tmp_path):
    path = tmp_path / "missing.pt"
    torch.save({"transitions": {"observations": torch.zeros(3, OBS_DIM)}}, path)
    with pytest.raises(KeyError, match="missing"):
        _demo_learner(tmp_path, demo_transitions=str(path))


def test_a_learner_without_memory_never_reads_the_demo_file():
    """Eval builds SAC with no memory; a training run's demo path must not matter there."""
    from helpers import box, learner_cfg
    from robonuke_rl_core.learners.cfg import TrainerCfg
    from robonuke_rl_core.learners.sac import SAC
    from robonuke_rl_core.models.factory import build_models
    from helpers import tiny_model_cfg

    cfg = learner_cfg("sac", demo_transitions="/nonexistent/demos.pt", demo_sampling_end=10)
    model_cfg = tiny_model_cfg()
    models = build_models("sac", model_cfg, box(OBS_DIM), None, box(ACT_DIM), 1, "cpu")
    SAC(
        models=models,
        memory=None,
        observation_space=box(OBS_DIM),
        state_space=None,
        action_space=box(ACT_DIM),
        device="cpu",
        cfg=cfg,
        trainer_cfg=TrainerCfg(learner="sac", total_timesteps=10),
        num_agents=1,
        num_envs=1,
        model_cfg=model_cfg,
    )


# ---------------------------------------------------------------------- independence
@pytest.mark.parametrize(
    "overrides",
    [
        {"num_critics": 4, "critic_subset": 2},
        {"num_critics": 4, "critic_subset": 2, "actor_q_reduction": "mean", "gradient_steps": 2, "policy_delay": 2},
        {"demo": True},
        {"demo": True, "num_critics": 3, "critic_subset": 2, "gradient_steps": 2, "policy_delay": 2},
    ],
)
def test_one_agents_data_cannot_change_another_agents_update(tmp_path, overrides):
    overrides = dict(overrides)
    if overrides.pop("demo", False):
        overrides.update(
            batch_size=8,
            demo_transitions=write_demos(tmp_path / "demos.pt"),
            demo_sampling_start=0,
            demo_sampling_end=50,
        )
    control = build_learner("sac", num_agents=3, envs_per_agent=2, **overrides)
    fill_memory(control)
    before = {agent: snapshot(control, agent) for agent in UNTOUCHED}
    extreme = copy.deepcopy(control)
    make_extreme(extreme, AGENT_UNDER_TEST)

    for learner in (control, extreme):
        torch.manual_seed(7)
        for _ in range(3):
            learner.update(timestep=10, timesteps=100)

    for agent in UNTOUCHED:
        assert_same(snapshot(control, agent), snapshot(extreme, agent), f"agent{agent}")
        with pytest.raises(AssertionError):
            assert_same(before[agent]["model/policy"], snapshot(control, agent)["model/policy"])
    with pytest.raises(AssertionError):
        assert_same(snapshot(control, AGENT_UNDER_TEST), snapshot(extreme, AGENT_UNDER_TEST))


def test_a_periodic_reset_rebuilds_the_whole_ensemble():
    learner = build_learner(
        "sac", num_agents=1, num_critics=3, critic_subset=2,
        periodic_reset_enabled=True, periodic_reset_frequency=5,
    )
    old = learner.critics
    learner._periodic_reset()
    assert learner.critics is not old and learner.models["critics"] is learner.critics
    assert learner.critics.members == 3
    assert_same(learner.critics.agent_state_dict(0), learner.target_critics.agent_state_dict(0))


# ---------------------------------------------------------------------- stacking and graphs
def test_a_stacked_ensemble_equals_its_members_run_separately():
    """The stacked module is the separate critics, batched: same init (member-major build
    order), same outputs and gradients up to float reordering."""
    from robonuke_rl_core.models.ensemble import VmapEnsemble
    from robonuke_rl_core.models.simba import SimbaQCriticNet

    build = lambda: SimbaQCriticNet(OBS_DIM, ACT_DIM, 8, 1, 0.0)
    torch.manual_seed(0)
    stacked = VmapEnsemble(build, 2, members=3)
    torch.manual_seed(0)
    separate = [VmapEnsemble(build, 2) for _ in range(3)]
    named = lambda m: dict(m.named_parameters())
    for name, p in named(stacked).items():
        assert torch.equal(p, torch.stack([named(m)[name] for m in separate], dim=1)), name
    obs, act = torch.randn(2 * 5, OBS_DIM), torch.randn(2 * 5, ACT_DIM)
    out = stacked.forward_flat(obs, act)  # (members, agents * rows, 1)
    for k, member in enumerate(separate):
        assert torch.allclose(out[k], member.forward_flat(obs, act), atol=1e-6)
    out.sum().backward()
    sum(m.forward_flat(obs, act).sum() for m in separate).backward()
    for name, p in named(stacked).items():
        grads = torch.stack([named(m)[name].grad for m in separate], dim=1)
        assert torch.allclose(p.grad, grads, atol=1e-5), name


def test_a_checkpoint_with_separate_critics_loads_into_the_stack(tmp_path):
    learner = build_learner("sac", num_agents=1, num_critics=3, critic_subset=2)
    fill_memory(learner)
    learner.update(timestep=10, timesteps=100)
    ckpt = learner._build_checkpoint(0, 10)
    # written the old way: critic_k / target_critic_k plain state dicts
    for key, prefix in (("critics", "critic_"), ("target_critics", "target_critic_")):
        stacked = ckpt["models"].pop(key)
        for k in range(3):
            ckpt["models"][f"{prefix}{k + 1}"] = {n: t[k] for n, t in stacked.items()}
    path = tmp_path / "old.ckpt"
    torch.save(ckpt, path)
    fresh = build_learner("sac", num_agents=1, num_critics=3, critic_subset=2)
    fresh.load_agent(path, slot=0, with_optimizer=False)
    assert_same(fresh.critics.agent_state_dict(0), learner.critics.agent_state_dict(0))
    assert_same(fresh.target_critics.agent_state_dict(0), learner.target_critics.agent_state_dict(0))
    wrong = build_learner("sac", num_agents=1, num_critics=4, critic_subset=2)
    with pytest.raises(KeyError, match="num_critics"):
        wrong.load_agent(path, slot=0, with_optimizer=False)


def test_the_graph_safe_sampler_only_draws_written_rows():
    learner = build_learner("sac", num_agents=2, rollout=50)
    fill_memory(learner, steps=3)  # 6 rows per agent of 100
    memory = learner.memory
    memory.sync_size()
    sampled = memory.sample_graph_safe(names=["rewards"], batch_size=4000)[0][0].view(2, -1)
    for agent in range(2):
        written = memory.get_tensor_by_name("rewards")[agent, : memory.size].reshape(-1)
        assert torch.isin(sampled[agent], written).all()  # its own rows, never an unwritten (NaN) one
        assert len(torch.unique(sampled[agent])) == memory.size  # and every written row is reachable

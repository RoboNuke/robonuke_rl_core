"""SAC beyond the classic twin: the Q ensemble, the delayed / late-starting actor, demo
transitions mixed into the critic's batches, the critic-action transform, and Policy Decorator's
progressive exploration. Each is off by default; the default path's equivalence
with the old twin is what every other SAC test already pins.
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
        ({"progressive_exploration_steps": -1}, "progressive_exploration_steps"),
        ({"update_every": 0}, "update_every"),
        ({"actor_lr_warmup": -1}, "actor_lr_warmup"),
        ({"critic_lr_warmup": -1}, "critic_lr_warmup"),
        ({"reward_clip": 0.0}, "reward_clip"),
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
        progressive_exploration_steps=30000,
        update_every=4,
        actor_lr_warmup=100,
        normalize_rewards=True,
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
    _, _, _, target, next_log_prob, _ = learner._compute_critic_loss(
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
        {"n_step": 3},
        {"demo": True, "n_step": 3},
    ],
)
def test_one_agents_data_cannot_change_another_agents_update(tmp_path, overrides):
    overrides = dict(overrides)
    if overrides.pop("demo", False):
        obs = torch.randn(33, OBS_DIM)
        # n-step folds demo rows within episodes: one 32-step episode, each row handing on its state
        chained = {"observations": obs[:-1], "next_observations": obs[1:]} if overrides.get("n_step", 1) > 1 else {}
        overrides.update(
            batch_size=8,
            demo_transitions=write_demos(tmp_path / "demos.pt", **chained),
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


# ------------------------------------------------------------------ the critics' action input
def _transform_step(learner, transform):
    """One critic + actor gradient step with ``transform`` on, recording what the critics got."""
    from helpers import fill_memory

    fill_memory(learner)
    learner.critic_action_transform = transform
    taken, batches = [], []
    for name in ("critics", "target_critics"):
        model = getattr(learner, name)
        original = model.act

        def act(inputs, role="", _original=original, _name=name):
            taken.append((_name, inputs["taken_actions"].detach().clone()))
            return _original(inputs, role=role)

        model.act = act
    sample = learner._sample

    def recorded_sample(rows, graph):
        batches.append(sample(rows, graph))
        return batches[-1]

    learner._sample = recorded_sample
    learner._gradient_step((False, True, True), learner.cfg.batch_size, 10, graph=False)
    return taken, batches


def test_a_critic_action_transform_feeds_every_critic_call_with_the_raw_observations():
    learner = build_learner("sac", num_agents=1, gradient_steps=1, policy_delay=1)
    seen = []

    def transform(observations, actions):
        seen.append(observations)
        return actions + 100.0

    taken, batches = _transform_step(learner, transform)
    # the critic loss, its target, and the actor's Q: every one sees the transformed action
    assert sorted(name for name, _ in taken) == ["critics", "critics", "target_critics"]
    assert all(actions.min() > 50.0 for _, actions in taken)
    raw = {id(batches[0]["observations"]), id(batches[0]["next_observations"])}
    assert len(seen) == 3 and all(id(obs) in raw for obs in seen)  # before any normalizer


def test_an_asymmetric_learner_hands_the_transform_the_critic_states_not_the_observations():
    """A state space exists: the transform must see what the critic sees (``states`` for the
    critic loss and the actor's Q, ``next_states`` for the target), never the actor's
    observations, so it can read what is in the state alone."""
    from helpers import STATE_DIM

    learner = build_learner("sac", num_agents=1, asymmetric=True, gradient_steps=1, policy_delay=1)
    seen = []

    def transform(critic_inputs, actions):
        seen.append(critic_inputs)
        return actions + 100.0

    taken, batches = _transform_step(learner, transform)
    assert len(seen) == 3 and all(actions.min() > 50.0 for _, actions in taken)
    batch = batches[0]
    assert all(inputs.shape[-1] == STATE_DIM for inputs in seen)
    # the target's call takes next_states, the critic loss and the actor's Q the states: the raw
    # sampled tensors themselves, before any normalizer
    ids = [id(inputs) for inputs in seen]
    assert ids.count(id(batch["next_states"])) == 1 and ids.count(id(batch["states"])) == 2
    assert not any(i in (id(batch["observations"]), id(batch["next_observations"])) for i in ids)


def test_the_transform_sees_the_states_of_demo_rows_too(tmp_path):
    """Inside the demo window the critic's batch is half demo rows; asymmetric demos carry
    states / next_states, and that is what reaches the transform (widths tell them apart)."""
    from helpers import STATE_DIM, fill_memory

    rows = 32
    path = write_demos(
        tmp_path / "demos.pt", rows=rows,
        states=torch.full((rows, STATE_DIM), 7.0), next_states=torch.full((rows, STATE_DIM), 8.0),
    )
    learner = build_learner(
        "sac", num_agents=1, asymmetric=True, gradient_steps=1, policy_delay=1,
        demo_transitions=path, demo_sampling_start=0, demo_sampling_end=100,
    )
    fill_memory(learner)
    seen = []

    def transform(critic_inputs, actions):
        seen.append(critic_inputs.detach().clone())
        return actions

    learner.critic_action_transform = transform
    mixed = {}
    original = learner._mixed_batch
    learner._mixed_batch = lambda rows, graph=False: mixed.setdefault("batch", original(rows, graph))
    learner._gradient_step((True, True, True), learner.cfg.batch_size, 10, graph=False)
    assert len(seen) == 3 and all(inputs.shape[-1] == STATE_DIM for inputs in seen)
    batch = mixed["batch"]
    # the critic step's two calls: the target on the mixed batch's next_states, the loss on its
    # states; demo rows are recognizable by their constant values in the second half
    assert torch.equal(seen[0], batch["next_states"]) and torch.equal(seen[1], batch["states"])
    half = learner.cfg.batch_size // 2
    assert torch.all(seen[0][half:] == 8.0) and torch.all(seen[1][half:] == 7.0)
    assert seen[2].shape[0] == learner.cfg.batch_size  # the actor's Q on its online batch's states


def test_without_a_transform_the_critics_take_the_agents_action():
    learner = build_learner("sac", num_agents=1, gradient_steps=1, policy_delay=1)
    taken, _ = _transform_step(learner, None)
    assert all(actions.abs().max() <= 1.0 for _, actions in taken)  # the squashed policy's own


def test_a_transform_that_changes_the_action_shape_is_refused():
    learner = build_learner("sac", num_agents=1, gradient_steps=1, policy_delay=1)
    with pytest.raises(ValueError, match="critic action transform"):
        _transform_step(learner, lambda observations, actions: actions[..., :1])


def test_at_most_one_registered_builder_may_answer(monkeypatch):
    from robonuke_rl_core.learners import sac

    monkeypatch.setattr(sac, "CRITIC_ACTION_BUILDERS", [])
    assert sac.critic_action_transform(None, None) is None
    sac.register_critic_action(lambda cfg, env: None)
    sac.register_critic_action(lambda cfg, env: (lambda obs, act: act))
    assert sac.critic_action_transform(None, None) is not None
    sac.register_critic_action(lambda cfg, env: (lambda obs, act: act))
    with pytest.raises(ValueError, match="at most one"):
        sac.critic_action_transform(None, None)


# ---------------------------------------------------------------------- progressive exploration (Policy Decorator)
def _fixed_policy(learner, value: float = 0.5):
    """The policy's act replaced by constant actions, so a kept row is recognizable."""
    learner.policy.act = lambda inputs, role="": (
        torch.full((inputs["observations"].shape[0], ACT_DIM), value), {}
    )


def _shares(learner):
    shares = []
    learner.on_log.append(
        lambda agent, metrics, step: shares.append((agent, float(metrics["exploration/policy_share"])))
        if "exploration/policy_share" in metrics else None
    )
    return shares


def _record(learner, ended, timestep: int) -> None:
    """One env-step into the learner, the envs in ``ended`` timing out."""
    n = learner.num_envs
    learner.record_transition(
        observations=torch.randn(n, OBS_DIM), states=None, actions=torch.zeros(n, ACT_DIM),
        rewards=torch.zeros(n, 1), next_observations=torch.randn(n, OBS_DIM), next_states=None,
        terminated=torch.zeros(n, 1, dtype=torch.bool), truncated=ended.view(-1, 1), infos={},
        timestep=timestep, timesteps=10000,
    )


def test_progressive_exploration_draws_each_env_every_step_with_probability_t_over_h():
    learner = build_learner("sac", num_agents=2, envs_per_agent=500, progressive_exploration_steps=1000)
    _fixed_policy(learner)
    shares = _shares(learner)
    observations = torch.randn(1000, OBS_DIM)
    actions, _ = learner.act(observations, None, timestep=250, timesteps=10000)
    kept = (actions == 0.5).all(dim=1)
    assert torch.equal(kept | (actions == 0.0).all(dim=1), torch.ones(1000, dtype=torch.bool))  # whole rows
    per_agent = kept.float().view(2, 500).mean(dim=1)
    assert all(abs(share - 0.25) < 0.08 for share in per_agent.tolist())
    assert shares == [(0, float(per_agent[0])), (1, float(per_agent[1]))]  # the realized share
    # a fresh draw EVERY env step, mid-episode too (Policy Decorator's per-step mask): the next
    # step's mask is another sample at the same share, not the last one held
    _record(learner, torch.zeros(1000, dtype=torch.bool), timestep=250)
    again, _ = learner.act(observations, None, timestep=251, timesteps=10000)
    again_kept = (again == 0.5).all(dim=1)
    assert not torch.equal(again_kept, kept)
    assert abs(float(again_kept.float().mean()) - 0.25) < 0.08
    # the share follows the step count, and from H on every env runs the policy every step
    later, _ = learner.act(observations, None, timestep=750, timesteps=10000)
    assert abs(float((later == 0.5).all(dim=1).float().mean()) - 0.75) < 0.08
    for timestep in (1000, 5000):
        after, _ = learner.act(observations, None, timestep=timestep, timesteps=10000)
        assert torch.all(after == 0.5)


def test_one_uniform_per_env_per_step_and_the_first_step_runs_the_base_alone():
    learner = build_learner("sac", num_agents=1, envs_per_agent=8, progressive_exploration_steps=1000)
    _fixed_policy(learner)
    start, _ = learner.act(torch.randn(8, OBS_DIM), None, timestep=0, timesteps=10000)
    assert torch.equal(start, torch.zeros_like(start))
    # the draw is rand(num_envs, 1) < t / H and nothing else, so it depends on the step alone
    torch.manual_seed(7)
    actions, _ = learner.act(torch.randn(8, OBS_DIM), None, timestep=500, timesteps=10000)
    torch.manual_seed(7)
    torch.randn(8, OBS_DIM)  # the observations drawn above
    expected = torch.rand(8, 1) < 0.5
    assert torch.equal((actions == 0.5).all(dim=1, keepdim=True), expected)


# ---------------------------------------------------------------------- update cadence (update_every)
def test_update_every_runs_a_round_of_gradient_steps_every_so_many_env_steps_after_learning_starts():
    learner = build_learner("sac", num_agents=1, envs_per_agent=4, learning_starts=2, update_every=3, gradient_steps=2)
    rounds = []
    learner.update = lambda timestep, timesteps: rounds.append(timestep)
    for timestep in range(12):
        learner._update_if_ready(timestep=timestep, timesteps=12)
    assert rounds == [2, 5, 8, 11]
    assert learner._update_rounds(12) == 4 and learner._update_rounds(2) == 0  # what the LR schedule spans

    default = build_learner("sac", num_agents=1, envs_per_agent=4)
    rounds = []
    default.update = lambda timestep, timesteps: rounds.append(timestep)
    for timestep in range(5):
        default._update_if_ready(timestep=timestep, timesteps=5)
    assert rounds == [0, 1, 2, 3, 4]  # update_every 1: every env step, as before


def test_update_every_composes_with_gradient_steps_and_policy_delay():
    """16 gradient steps every 4 env steps on 16 envs (Policy Decorator) is UTD 0.25; here 2
    every 3, the actor on every 2nd gradient step: two rounds give 4 gradient steps and 2 actor
    steps, and the metrics come once per round."""
    learner = build_learner("sac", num_agents=1, envs_per_agent=4, update_every=3, gradient_steps=2, policy_delay=2)
    fill_memory(learner)
    emitted = []
    learner.on_log.append(lambda agent, metrics, step: emitted.append(set(metrics)))
    before = weights(learner.policy)
    for timestep in range(6):
        learner._update_if_ready(timestep=timestep, timesteps=6)
    assert learner._updates == 4
    assert sum("lr/critic" in keys for keys in emitted) == 2  # one point per round
    assert sum("grad_norm/policy" in keys for keys in emitted) == 2  # the actor stepped in each round
    assert changed(before, learner.policy)


def test_progressive_exploration_masks_the_random_warmup_actions_too():
    learner = build_learner("sac", num_agents=1, envs_per_agent=8, random_timesteps=10, progressive_exploration_steps=1000)
    actions, _ = learner.act(torch.randn(8, OBS_DIM), None, timestep=0, timesteps=100)
    assert torch.equal(actions, torch.zeros_like(actions))


@pytest.mark.parametrize("steps, training", [(0, True), (1000, False)])
def test_off_or_outside_training_the_policys_actions_pass_and_nothing_is_drawn(steps, training):
    learner = build_learner("sac", num_agents=1, envs_per_agent=8, progressive_exploration_steps=steps)
    learner.enable_training_mode(training)
    _fixed_policy(learner)
    shares = _shares(learner)
    observations = torch.randn(8, OBS_DIM)
    state = torch.random.get_rng_state()
    actions, _ = learner.act(observations, None, timestep=10, timesteps=100)
    assert torch.equal(actions, torch.full_like(actions, 0.5))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert shares == []


# ---------------------------------------------------------------------- n-step TD targets
def _step(rewards, terminated=None, truncated=None, obs=None):
    """One env-step of two envs: ``obs`` marks the step (observations = obs, next = obs + 1)."""
    rows = len(rewards)
    obs = torch.zeros(rows, 1) if obs is None else torch.full((rows, 1), float(obs))
    flag = lambda v: torch.tensor(v or [False] * rows).view(-1, 1)  # noqa: E731
    return {
        "observations": obs, "next_observations": obs + 1, "actions": obs * 10,
        "rewards": torch.tensor(rewards, dtype=torch.float32).view(-1, 1),
        "terminated": flag(terminated), "truncated": flag(truncated),
    }


def test_an_n_step_row_sums_the_discounted_rewards_and_bootstraps_n_steps_on():
    from robonuke_rl_core.learners.n_step import NStepWindow

    window = NStepWindow(3, 0.5)
    assert window.push(_step([1.0, 1.0], obs=0)) is None
    assert window.push(_step([2.0, 2.0], obs=1)) is None
    row = window.push(_step([4.0, 4.0], obs=2))
    assert row["rewards"].flatten().tolist() == [1 + 0.5 * 2 + 0.25 * 4] * 2
    assert row["discounts"].flatten().tolist() == [0.125] * 2
    assert row["observations"].flatten().tolist() == [0.0, 0.0] and row["actions"].flatten().tolist() == [0.0, 0.0]
    assert row["next_observations"].flatten().tolist() == [3.0, 3.0]  # the state 3 steps on
    row = window.push(_step([8.0, 8.0], obs=3))  # the next row: the step after, one row per env each step
    assert row["observations"].flatten().tolist() == [1.0, 1.0] and row["rewards"].flatten().tolist() == [2 + 2 + 2] * 2


def test_a_window_stops_at_the_episode_end_and_keeps_its_flags():
    """Env 0 terminates on the window's 2nd step, env 1 times out on its 1st: neither window reaches
    into the next episode, each bootstraps (or not) from where it stopped."""
    from robonuke_rl_core.learners.n_step import NStepWindow

    window = NStepWindow(3, 0.5)
    window.push(_step([1.0, 1.0], truncated=[False, True], obs=0))
    window.push(_step([2.0, 100.0], terminated=[True, False], obs=1))
    row = window.push(_step([100.0, 100.0], obs=2))
    assert row["rewards"].flatten().tolist() == [1 + 0.5 * 2, 1.0]
    assert row["discounts"].flatten().tolist() == [0.25, 0.5]
    assert row["next_observations"].flatten().tolist() == [2.0, 1.0]
    assert row["terminated"].flatten().tolist() == [True, False]
    assert row["truncated"].flatten().tolist() == [False, True]


def test_demo_rows_fold_within_their_episodes_and_an_unmarked_end_is_refused():
    from robonuke_rl_core.learners.n_step import n_step_rows

    obs = torch.tensor([[0.0], [1.0], [2.0], [10.0], [11.0]])
    demo = {
        "observations": obs,
        "next_observations": torch.tensor([[1.0], [2.0], [3.0], [11.0], [12.0]]),
        "actions": obs,
        "rewards": torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]]),
        "terminated": torch.tensor([[False], [False], [True], [False], [False]]),  # episode 1: rows 0-2
        "truncated": torch.zeros(5, 1, dtype=torch.bool),  # episode 2: rows 3-4, the file's end
    }
    out = n_step_rows(demo, 2, 0.5)
    assert out["rewards"].flatten().tolist() == [0.0, 0.5, 1.0, 0.0, 0.0]
    assert out["discounts"].flatten().tolist() == [0.25, 0.25, 0.5, 0.25, 0.5]
    assert out["next_observations"].flatten().tolist() == [2.0, 3.0, 3.0, 12.0, 12.0]
    assert out["terminated"].flatten().tolist() == [False, True, True, False, False]
    unmarked = dict(demo, terminated=torch.zeros(5, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="demo row 2 ends no episode"):
        n_step_rows(unmarked, 2, 0.5)


def test_with_n_step_the_memory_gets_one_folded_row_per_env_per_step_and_the_target_its_discount():
    learner = build_learner("sac", num_agents=2, n_step=3, normalize_observations=False)
    stored = []
    add = learner.memory.add_samples
    learner.memory.add_samples = lambda **values: (stored.append(values), add(**values))[1]
    fill_memory(learner, steps=6)
    assert len(stored) == 6 - 2 and all(v["rewards"].shape[0] == learner.num_envs for v in stored)
    assert "discounts" in learner._tensors_names
    sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=4)[0]))
    sampled["discounts"] = torch.full_like(sampled["discounts"], 0.25)
    inputs, next_inputs = {"observations": sampled["observations"]}, {"observations": sampled["next_observations"]}
    learner.cfg.backup_entropy = False
    torch.manual_seed(0)
    _, _, _, target, _, _ = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs, critic_next_inputs=next_inputs, rows=4,
    )
    sampled["rewards"] = torch.zeros_like(sampled["rewards"])
    torch.manual_seed(0)
    _, _, _, bootstrap, _, _ = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs, critic_next_inputs=next_inputs, rows=4,
    )
    sampled["discounts"] = torch.full_like(sampled["discounts"], 0.5)
    torch.manual_seed(0)
    _, _, _, doubled, _, _ = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs, critic_next_inputs=next_inputs, rows=4,
    )
    assert torch.allclose(doubled, 2 * bootstrap) and not torch.allclose(target, bootstrap)


def test_with_n_step_the_demo_rows_are_folded_when_loaded(tmp_path):
    rows = 6
    obs = torch.randn(rows + 1, OBS_DIM)
    path = write_demos(
        tmp_path / "demos.pt", rows=rows, observations=obs[:-1], next_observations=obs[1:],
        rewards=torch.ones(rows, 1),
    )
    learner = build_learner("sac", num_agents=1, n_step=3, demo_transitions=path, demo_sampling_start=0, demo_sampling_end=50,
                            discount_factor=0.5)
    assert learner._demo["rewards"].flatten().tolist() == [1.75, 1.75, 1.75, 1.75, 1.5, 1.0]
    assert learner._demo["discounts"].flatten().tolist() == [0.125, 0.125, 0.125, 0.125, 0.25, 0.5]


def test_a_delayed_progressive_exploration_holds_the_share_at_zero_until_its_start():
    learner = build_learner("sac", num_agents=1, progressive_exploration_steps=100, progressive_exploration_start=50)
    assert [learner._explore_share(t) for t in (0, 49, 50, 100, 150, 400)] == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
    actions = torch.ones(learner.num_envs, ACT_DIM)
    assert torch.equal(learner._explore_progressively(actions, 40), torch.zeros_like(actions))
    plain = build_learner("sac", num_agents=1, progressive_exploration_steps=100)
    assert [plain._explore_share(t) for t in (0, 50, 100)] == [0.0, 0.5, 1.0]  # start 0: unchanged


def test_a_start_without_a_ramp_is_refused():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="progressive_exploration_start"):
        SACCfg(progressive_exploration_start=10).validate(None)


def test_the_acting_std_is_logged_per_label_from_the_previous_steps_infos():
    learner = build_learner("sac", num_agents=2, envs_per_agent=3, std_by_label="eval_state/phase/id",
                            std_label_names=["corner", "grasp"])
    seen = []
    learner.on_log.append(lambda agent, metrics, step: seen.append((agent, metrics)))
    n = learner.num_envs
    obs = torch.randn(n, OBS_DIM)
    learner.act(obs, None, timestep=0, timesteps=10)  # no label yet: nothing
    assert not [m for _, m in seen if any(k.startswith("policy/std_") for k in m)]
    labels = torch.tensor([0, 0, 1, 1, 5, 0], dtype=torch.float32)  # 5: an end code, skipped
    learner.record_transition(
        observations=obs, states=None, actions=torch.zeros(n, ACT_DIM), rewards=torch.zeros(n, 1),
        next_observations=obs, next_states=None, terminated=torch.zeros(n, 1, dtype=torch.bool),
        truncated=torch.zeros(n, 1, dtype=torch.bool), infos={"eval_state": {"phase/id": labels}}, timestep=0, timesteps=10,
    )
    seen.clear()
    _, outputs = learner.act(obs, None, timestep=1, timesteps=10)
    std = outputs["log_std"].exp().mean(-1)
    got = {(agent, k): v for agent, m in seen for k, v in m.items() if k.startswith("policy/std_")}
    torch.testing.assert_close(got[(0, "policy/std_corner")], std[0:2])
    torch.testing.assert_close(got[(0, "policy/std_grasp")], std[2:3])
    torch.testing.assert_close(got[(1, "policy/std_grasp")], std[3:4])
    torch.testing.assert_close(got[(1, "policy/std_corner")], std[5:6])
    assert len(got) == 4  # env 4's end code is in no group


def test_std_by_label_needs_its_names():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="std_label_names"):
        SACCfg(std_by_label="eval_state/phase/id").validate(None)


def test_env_metrics_are_logged_per_label_over_the_envs_that_ran_the_policy():
    learner = build_learner("sac", num_agents=1, envs_per_agent=4, std_by_label="eval_state/phase/id",
                            std_label_names=["corner", "grasp"], metrics_by_label=["residual/position_mm"],
                            progressive_exploration_steps=100)
    seen = []
    learner.on_log.append(lambda agent, metrics, step: seen.append(metrics))
    n = learner.num_envs
    learner._std_labels = torch.tensor([0, 0, 1, 1])  # the observations' phases
    learner._explore_keep = torch.tensor([True, False, True, True])  # env 1 ran on zeros
    learner._emit_metrics_by_label({"metrics_to_log": {"residual/position_mm": torch.tensor([1.0, 2.0, 3.0, 4.0])}}, 0)
    got = {k: v for m in seen for k, v in m.items()}
    torch.testing.assert_close(got["residual/position_mm_corner"], torch.tensor([1.0]))
    torch.testing.assert_close(got["residual/position_mm_grasp"], torch.tensor([3.0, 4.0]))


def test_metrics_by_label_needs_the_labels():
    from robonuke_rl_core.learners.cfg import SACCfg

    with pytest.raises(ValueError, match="std_by_label"):
        SACCfg(metrics_by_label=["residual/position_mm"]).validate(None)

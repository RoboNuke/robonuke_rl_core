"""LearnerBase: env metrics, episode statistics, hooks, checkpoints, and the config rule."""

from __future__ import annotations

import copy

import pytest
import torch

from robonuke_rl_core.learners.base import run_dirs
from robonuke_rl_core.losses import LossContext

from helpers import ACT_DIM, OBS_DIM, build_learner, fill_memory


def _policy_head_bias(policy):
    """The actor's stacked output bias."""
    return getattr(policy.net, "trunk__fc_out__bias")


def collector():
    """An on_log hook plus the list it fills."""
    seen: list[tuple[int, dict, int]] = []
    return seen, lambda agent, metrics, step: seen.append((agent, dict(metrics), step))


def step_once(learner, metrics_to_log=None, rewards=None, terminated=None, step: int = 0):
    num_envs = learner.num_envs
    observations = torch.zeros(num_envs, OBS_DIM)
    actions = torch.zeros(num_envs, ACT_DIM)
    learner.record_transition(
        observations=observations,
        states=None,
        actions=actions,
        rewards=torch.zeros(num_envs, 1) if rewards is None else rewards,
        next_observations=observations,
        next_states=None,
        terminated=torch.zeros(num_envs, 1, dtype=torch.bool) if terminated is None else terminated,
        truncated=torch.zeros(num_envs, 1, dtype=torch.bool),
        infos={} if metrics_to_log is None else {"metrics_to_log": metrics_to_log},
        timestep=step,
        timesteps=100,
    )


# ------------------------------------------------------------------ 5. env metrics
def test_env_metrics_reach_each_hook_sliced_to_its_envs():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    seen, hook = collector()
    learner.on_log.append(hook)

    # one value per env, so each agent must see its own two
    values = torch.arange(learner.num_envs, dtype=torch.float32)
    step_once(learner, metrics_to_log={"env/force": values}, step=4)

    assert [agent for agent, _, _ in seen] == [0, 1, 2]
    for agent, metrics, step in seen:
        assert step == 4
        assert torch.equal(metrics["env/force"], values[agent * 2 : agent * 2 + 2])


def test_env_metrics_with_no_hooks_log_nothing():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    step_once(learner, metrics_to_log={"env/force": torch.zeros(learner.num_envs)})  # no raise


def test_a_wrong_metric_shape_raises():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    learner.on_log.append(lambda *_: None)
    with pytest.raises(TypeError) as err:
        step_once(learner, metrics_to_log={"env/force": torch.zeros(learner.num_envs + 1)})
    assert "env/force" in str(err.value)
    with pytest.raises(TypeError):
        step_once(learner, metrics_to_log={"env/force": 1.0})
    with pytest.raises(TypeError):
        step_once(learner, metrics_to_log=[1, 2, 3])


def test_a_missing_metrics_key_forwards_nothing():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    seen, hook = collector()
    learner.on_log.append(hook)
    step_once(learner, metrics_to_log=None)
    assert seen == []


# ------------------------------------------------------------------ 6. episode statistics
def test_episode_statistics_are_per_agent():
    learner = build_learner(
        "sac", num_agents=2, envs_per_agent=2, trainer_overrides={"write_interval": 3}
    )
    seen, hook = collector()
    learner.on_log.append(hook)

    # agent 0's envs collect 1 per step, agent 1's collect 10 per step
    rewards = torch.tensor([[1.0], [1.0], [10.0], [10.0]])
    none_done = torch.zeros(4, 1, dtype=torch.bool)
    all_done = torch.ones(4, 1, dtype=torch.bool)
    step_once(learner, rewards=rewards, terminated=none_done, step=0)
    step_once(learner, rewards=rewards, terminated=none_done, step=1)
    step_once(learner, rewards=rewards, terminated=all_done, step=2)
    learner._flush_episode_stats(step=2)

    stats = {agent: metrics for agent, metrics, _ in seen}
    assert float(stats[0]["episode/return"]) == pytest.approx(3.0)
    assert float(stats[1]["episode/return"]) == pytest.approx(30.0)
    assert float(stats[0]["episode/length"]) == pytest.approx(3.0)
    assert float(stats[0]["episode/count"]) == pytest.approx(2.0)  # two envs finished
    # the interval is cleared, so the next flush publishes nothing
    seen.clear()
    learner._flush_episode_stats(step=3)
    assert seen == []


def test_best_checkpoint_follows_the_highest_mean_return(tmp_path):
    dirs = [tmp_path / "a0", tmp_path / "a1"]
    learner = build_learner(
        "sac",
        num_agents=2,
        envs_per_agent=2,
        run_dirs_list=dirs,
        trainer_overrides={"write_interval": 1, "checkpoint_interval": 100},
    )
    all_done = torch.ones(4, 1, dtype=torch.bool)

    def finish_episode(value: float, step: int) -> None:
        step_once(learner, rewards=torch.full((4, 1), value), terminated=all_done, step=step)
        learner.post_interaction(timestep=step, timesteps=100)

    finish_episode(5.0, step=0)
    best = dirs[0] / "checkpoints" / "ckpt_best.pt"
    assert best.is_file()
    first_return = torch.load(best, weights_only=False)["mean_return"]
    assert first_return == pytest.approx(5.0)

    finish_episode(1.0, step=1)  # worse: the file must not change
    assert torch.load(best, weights_only=False)["mean_return"] == pytest.approx(5.0)

    finish_episode(9.0, step=2)  # better: rewritten
    assert torch.load(best, weights_only=False)["mean_return"] == pytest.approx(9.0)


# ------------------------------------------------------------------ 4. checkpoints
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_checkpoint_round_trip_into_another_slot(tmp_path, learner_name):
    """Save agent 2 of 3, load it into slot 0 of a fresh 1-agent learner, compare actions."""
    source = build_learner(
        learner_name,
        num_agents=3,
        envs_per_agent=2,
        run_dirs_list=[tmp_path / f"a{i}" for i in range(3)],
    )
    fill_memory(source)
    torch.manual_seed(5)
    source.update(timestep=10, timesteps=100)
    paths = source.save_checkpoints(step=10)
    assert all(p.is_file() for p in paths)

    target = build_learner(learner_name, num_agents=1, envs_per_agent=2)
    meta = target.load_agent(paths[2], slot=0, with_optimizer=True)
    assert meta["agent_idx"] == 2 and meta["step"] == 10

    observations = torch.randn(2, OBS_DIM)
    source.enable_models_training_mode(False)
    target.enable_models_training_mode(False)
    with torch.no_grad():
        # agent 2's rows of the source batch vs the single-agent target
        batch = torch.cat([torch.zeros(4, OBS_DIM), observations], dim=0)
        source_inputs = {"observations": source.normalize_observations(batch)}
        _, source_out = source.policy.act(source_inputs, role="policy")
        target_inputs = {"observations": target.normalize_observations(observations)}
        _, target_out = target.policy.act(target_inputs, role="policy")
    assert torch.allclose(
        source_out["mean_actions"][4:], target_out["mean_actions"], atol=1e-6
    )


@pytest.mark.parametrize("learner_name", ["sac"])
def test_loading_one_slot_leaves_every_other_agent_alone(tmp_path, learner_name):
    """Target critics are saved per agent; loading slot 1 touches slot 1 only."""
    source = build_learner(
        learner_name, num_agents=3, envs_per_agent=2, run_dirs_list=[tmp_path / f"a{i}" for i in range(3)]
    )
    fill_memory(source)
    torch.manual_seed(5)
    source.update(timestep=10, timesteps=100)  # targets now lag their critics
    path = source.save_checkpoints(step=10)[2]

    learner = build_learner(learner_name, num_agents=3, envs_per_agent=2)
    fill_memory(learner)
    torch.manual_seed(6)
    learner.update(timestep=10, timesteps=100)
    keys = learner._checkpoint_model_keys()
    assert "target_critic_1" in keys and "target_critic_2" in keys
    before = {a: {k: getattr(learner, k).agent_state_dict(a) for k in keys} for a in (0, 2)}

    learner.load_agent(path, slot=1)

    for agent in (0, 2):
        for key in keys:
            after = getattr(learner, key).agent_state_dict(agent)
            for name in after:
                assert torch.equal(after[name], before[agent][key][name]), (agent, key, name)
    # slot 1 holds the saved agent's lagging targets, not a copy of its critics
    for key in ("target_critic_1", "target_critic_2"):
        saved = getattr(source, key).agent_state_dict(2)
        loaded = getattr(learner, key).agent_state_dict(1)
        for name in saved:
            assert torch.equal(saved[name], loaded[name]), (key, name)


def test_ppo_every_agent_dropped_freezes_the_whole_update():
    """Over the KL limit, an agent is frozen policy AND critic (the keep mask on both
    optimizers). With every agent dropped the epoch breaks out and nothing moves at all."""
    # observation normalization on: the stats move between rollout and update, so every
    # agent's KL starts far above this threshold
    learner = build_learner("ppo", num_agents=3, envs_per_agent=2, kl_threshold=1.0e-6)
    fill_memory(learner)
    policy_before = copy.deepcopy(learner.policy.state_dict())
    value_before = copy.deepcopy(learner.value.state_dict())

    torch.manual_seed(7)
    learner.update(timestep=10, timesteps=100)

    for name, tensor in learner.policy.state_dict().items():
        assert torch.equal(tensor, policy_before[name]), f"policy {name} moved"
    for name, tensor in learner.value.state_dict().items():
        assert torch.equal(tensor, value_before[name]), f"value {name} moved"
    # no agent stepped, so no step counter advanced and no moment was written
    for optimizer in (learner.policy_optimizer, learner.value_optimizer):
        assert optimizer.t.tolist() == [0] * learner.num_agents
        for moment in optimizer.exp_avg:
            assert float(moment.abs().max()) == 0.0


def test_an_optimizer_slot_loads_at_any_agent_count(tmp_path):
    """BlockAdamW keeps its moments per agent, so one file fills exactly one slot."""
    source = build_learner(
        "sac", num_agents=3, envs_per_agent=2, run_dirs_list=[tmp_path / f"a{i}" for i in range(3)]
    )
    fill_memory(source)
    torch.manual_seed(5)
    source.update(timestep=10, timesteps=100)
    paths = source.save_checkpoints(step=10)

    target = build_learner("sac", num_agents=3, envs_per_agent=2)
    target.load_agent(paths[2], slot=1, with_optimizer=True)
    for index in range(len(target.policy_optimizer.params)):
        assert torch.equal(
            target.policy_optimizer.exp_avg[index][1], source.policy_optimizer.exp_avg[index][2]
        )
        # the other slots keep their fresh (zero) moments
        assert float(target.policy_optimizer.exp_avg[index][0].abs().max()) == 0.0
    assert int(target.policy_optimizer.t[1]) == int(source.policy_optimizer.t[2])
    assert int(target.policy_optimizer.t[0]) == 0


def test_checkpoints_need_run_dirs():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2)
    with pytest.raises(RuntimeError) as err:
        learner.save_checkpoints(step=0)
    assert "run_dirs" in str(err.value)


def test_run_dirs_follow_the_config_layout():
    class FakeCfg:
        def __init__(self):
            self.trainer = type("T", (), {"output_dir": "runs"})()
            self.wandb = type("W", (), {"project": "forge_pih", "group": "fgain_k100"})()
            self.derived = {"run_names": ["fgain_k100_a0", "fgain_k100_a1"]}

    dirs = run_dirs(FakeCfg())
    assert [str(d) for d in dirs] == [
        "runs/forge_pih/fgain_k100/fgain_k100_a0",
        "runs/forge_pih/fgain_k100/fgain_k100_a1",
    ]


# ------------------------------------------------------------------ 7. aux_loss hook
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
@pytest.mark.parametrize("target", ["policy", "critic"])
def test_aux_loss_hook_changes_the_loss_by_what_it_returns(learner_name, target):
    """A hook adding a constant shifts that loss; with no hook the loss is the ported one."""
    plain = build_learner(learner_name, num_agents=2, envs_per_agent=2)
    fill_memory(plain)
    hooked = copy.deepcopy(plain)

    seen_targets: list[str] = []

    def hook(ctx: LossContext):
        seen_targets.append(ctx.target)
        assert ctx.learner is hooked
        if ctx.target != target:
            return None
        # a constant times a policy parameter: a real gradient contribution
        return 100.0 * _policy_head_bias(hooked.policy).sum()

    hooked.aux_loss.append(hook)

    torch.manual_seed(3)
    plain.update(timestep=10, timesteps=100)
    torch.manual_seed(3)
    hooked.update(timestep=10, timesteps=100)

    assert set(seen_targets) == {"policy", "critic"}
    # the hook moved the policy's output bias (it is in both losses' graphs for PPO, and in the
    # policy loss for SAC); the two learners must differ somewhere
    plain_bias = _policy_head_bias(plain.policy)
    hooked_bias = _policy_head_bias(hooked.policy)
    if target == "policy" or learner_name == "ppo":
        assert not torch.equal(plain_bias, hooked_bias)
    else:
        assert torch.equal(plain_bias, hooked_bias)  # SAC's critic loss has no policy params


def test_no_aux_hook_means_no_extra_loss():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2)
    assert learner.compute_aux_loss(
        LossContext(learner=learner, target="policy", sampled={})
    ) is None


def test_an_aux_hook_returning_a_non_tensor_raises():
    learner = build_learner("sac", num_agents=2, envs_per_agent=2)
    learner.aux_loss.append(lambda ctx: 1.0)
    with pytest.raises(TypeError) as err:
        learner.compute_aux_loss(LossContext(learner=learner, target="policy", sampled={}))
    assert "aux_loss" in str(err.value)


# ------------------------------------------------------------------ env partition
def test_env_count_must_divide_by_agents():
    with pytest.raises(ValueError) as err:
        build_learner("sac", num_agents=3, envs_per_agent=2, trainer_overrides=None).__class__(
            models={}, memory=None, observation_space=None, action_space=None, cfg=None,
            trainer_cfg=None, num_agents=3, num_envs=7,
        )
    assert "divisible" in str(err.value)


def test_learner_metrics_must_be_one_value_per_agent():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    learner.on_log.append(lambda *_: None)
    with pytest.raises(TypeError) as err:
        learner.emit_per_agent({"loss/x": torch.zeros(2)}, step=0)
    assert "loss/x" in str(err.value)

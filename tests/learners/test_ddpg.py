"""``trainer.learner: ddpg`` (learners/ddpg.py): the deterministic actor and its exploration, the critic target (target
actor, target-policy smoothing, a random-subset min), the actor objective and its reductions, the target actor's
Polyak step, ResFiT's warm-up actions and critic warm-up (per learner, so per routed phase), checkpoints, eval and
the config's refusals."""

from __future__ import annotations

import math

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner, fill_memory
from robonuke_rl_core.learners.cfg import DDPGCfg


def ddpg(**overrides):
    return build_learner("ddpg", num_agents=2, envs_per_agent=3, architecture="mlp", **overrides)


def params(module):
    return torch.cat([p.detach().reshape(-1).clone() for p in module.parameters()])


def actor_params(learner):
    return torch.cat([p.detach().reshape(-1).clone() for p in learner._actor_parameters()])


def batch(learner):
    torch.manual_seed(5)
    sampled = learner._sample(learner.cfg.batch_size)
    inputs, next_inputs, critic_inputs, critic_next_inputs = learner._batch_inputs(sampled, train=False)
    return sampled, inputs, next_inputs, critic_inputs, critic_next_inputs


def test_the_action_is_the_scaled_squashed_mean_plus_scheduled_noise_clamped():
    learner = ddpg(actor_action_scale=0.1, exploration_std=0.4, exploration_std_end=0.0, exploration_std_steps=100)
    obs = torch.randn(6, OBS_DIM)
    with torch.no_grad():
        _, out = learner.policy.act({"observations": obs}, role="policy")
    mean = out["mean_actions"] * 0.1
    learner.enable_training_mode(False)
    eval_action, _ = learner.act(obs, None, timestep=0, timesteps=100)
    torch.testing.assert_close(eval_action, mean)  # eval: the scaled mean, no noise
    torch.testing.assert_close(learner.eval_actions(obs, deterministic=True), mean)
    learner.enable_training_mode(True)
    assert learner.exploration_std(0) == pytest.approx(0.4) and learner.exploration_std(50) == pytest.approx(0.2)
    assert learner.exploration_std(500) == pytest.approx(0.0)
    torch.manual_seed(0)
    noisy = torch.stack([learner.act(obs, None, timestep=50, timesteps=100)[0] for _ in range(400)])
    assert float((noisy - mean).std()) == pytest.approx(0.2, rel=0.1)  # the std at step 50
    clamped = ddpg(actor_action_scale=1.0, exploration_std=5.0, action_clip=1.0)
    assert float(clamped.act(obs, None, timestep=1, timesteps=100)[0].abs().max()) <= 1.0


def test_resfits_warm_up_acts_uniformly_until_this_learner_starts_learning():
    learner = ddpg(random_action_scale=0.2, learning_starts_rows=10**6, actor_action_scale=0.1, exploration_std=0.0)
    obs = torch.randn(6, OBS_DIM)
    torch.manual_seed(0)
    warm = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(200)])
    assert float(warm.abs().max()) <= 0.2 and float(warm.abs().max()) > 0.18  # uniform in [-0.2, 0.2]
    learner._agent_rows_ready = [True] * learner.num_agents  # its rows are there: the actor (here noise-free) from now on
    with torch.no_grad():
        _, out = learner.policy.act({"observations": obs}, role="policy")
    torch.testing.assert_close(learner.act(obs, None, timestep=300, timesteps=1000)[0], out["mean_actions"] * 0.1)


def test_the_actor_gaussian_warm_up_acts_about_the_actors_own_action():
    # random_action_mode: actor_gaussian with model.actor.output_bias: the actor starts at tanh(bias) for every state
    # (a zero output layer), and the warm-up draws N(0, random_action_scale) about it, clamped to action_clip
    learner = ddpg(random_action_scale=0.05, random_action_mode="actor_gaussian", learning_starts_rows=10**6,
                   actor_action_scale=1.0, action_clip=1.0, exploration_std=0.0,
                   model_overrides={"output_init_scale": 0.0, "output_bias": math.atanh(0.6)})
    obs = torch.randn(6, OBS_DIM)
    learner.enable_training_mode(False)
    torch.testing.assert_close(learner.act(obs, None, timestep=0, timesteps=100)[0], torch.full((6, ACT_DIM), 0.6))
    learner.enable_training_mode(True)
    torch.manual_seed(0)
    warm = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(400)])
    assert float(warm.mean()) == pytest.approx(0.6, abs=0.01)
    assert float(warm.std()) == pytest.approx(0.05, rel=0.1)
    assert float(warm.abs().max()) <= 1.0


def test_the_warm_up_mode_must_be_known():
    with pytest.raises(ValueError, match="random_action_mode"):
        DDPGCfg(random_action_mode="normal").validate(None)


def test_the_critic_target_is_r_plus_gamma_min_of_target_critics_at_the_target_actors_action():
    learner = ddpg(num_critics=2, critic_subset=2, discount_factor=0.9, actor_action_scale=0.5)
    fill_memory(learner)
    with torch.no_grad():  # the target actor differs from the actor, so the source of a' shows
        for p in learner.target_policy.parameters():
            p.add_(0.3)
    sampled, inputs, next_inputs, critic_inputs, critic_next_inputs = batch(learner)
    _, _, _, target, _, _ = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=next_inputs,
                                                         critic_inputs=critic_inputs, critic_next_inputs=critic_next_inputs,
                                                         rows=learner.cfg.batch_size)
    with torch.no_grad():
        _, out = learner.target_policy.act(next_inputs, role="policy")
        a_next = (out["mean_actions"] * 0.5).clamp(-1, 1)
        q, _ = learner.target_critics.act({**critic_next_inputs, "taken_actions": a_next}, role="target_critics")
        terminal = sampled["terminated"] & ~sampled["truncated"]
        expected = sampled["rewards"] + 0.9 * (~terminal) * q.min(dim=0).values
    torch.testing.assert_close(target, expected)
    # without a target actor, a' comes from the actor itself
    learner.cfg.target_actor = False
    _, _, _, own, _, _ = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=next_inputs,
                                                      critic_inputs=critic_inputs, critic_next_inputs=critic_next_inputs,
                                                      rows=learner.cfg.batch_size)
    assert not torch.allclose(own, target)


def test_target_policy_smoothing_stays_within_its_clip():
    learner = ddpg(num_critics=1, target_policy_noise=True, exploration_std=10.0, target_noise_clip=1e-7)
    fill_memory(learner)
    learner._explore_std.fill_(10.0)
    sampled, inputs, next_inputs, critic_inputs, critic_next_inputs = batch(learner)
    rows = learner.cfg.batch_size
    smoothed = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=critic_inputs,
                                            critic_next_inputs=critic_next_inputs, rows=rows)[3]
    learner.cfg.target_policy_noise = False
    plain = learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=critic_inputs,
                                         critic_next_inputs=critic_next_inputs, rows=rows)[3]
    torch.testing.assert_close(smoothed, plain, atol=1e-5, rtol=0)  # noise of std 10 clipped to 1e-7


def test_the_critic_loss_forms():
    learner = ddpg(num_critics=3, critic_subset=3)
    fill_memory(learner)
    sampled, inputs, next_inputs, critic_inputs, critic_next_inputs = batch(learner)
    rows = learner.cfg.batch_size
    kw = dict(sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=critic_inputs,
              critic_next_inputs=critic_next_inputs, rows=rows)
    torch.manual_seed(1)
    mse, _, _, target, _, _ = learner._compute_critic_loss(**kw)
    with torch.no_grad():
        values, _ = learner.critics.act({**critic_inputs, "taken_actions": sampled["actions"]}, role="critics")
    torch.testing.assert_close(mse, (values - target).pow(2).mean())
    learner.cfg.critic_loss = "mean_abs_squared"
    torch.manual_seed(1)
    resfit = learner._compute_critic_loss(**kw)[0]
    torch.testing.assert_close(resfit, (values - target).abs().mean(dim=0).pow(2).mean())  # (mean_k |Q_k - y|)^2


@pytest.mark.parametrize("reduction", ["mean", "min", "q1"])
def test_the_actor_maximizes_the_reduced_q_at_its_noise_free_action(reduction):
    learner = ddpg(num_critics=3, critic_subset=2, actor_q_reduction=reduction, actor_action_scale=0.3,
                   action_l2_reg_weight=0.5)
    fill_memory(learner)
    sampled, inputs, _, critic_inputs, _ = batch(learner)
    rows = learner.cfg.batch_size
    loss, actions, _, q, _ = learner._compute_actor_loss(sampled=sampled, inputs=inputs, critic_inputs=critic_inputs, rows=rows)
    with torch.no_grad():
        _, out = learner.policy.act(inputs, role="policy")
        a = out["mean_actions"] * 0.3
        values, _ = learner.critics.act({**critic_inputs, "taken_actions": a}, role="critics")
        expected_q = {"mean": values.mean(dim=0), "min": values.min(dim=0).values, "q1": values[0]}[reduction]
    torch.testing.assert_close(actions.detach(), a)
    torch.testing.assert_close(q.detach(), expected_q)
    torch.testing.assert_close(loss.detach(), (-expected_q + 0.5 * a.pow(2).sum(-1, keepdim=True)).mean())


def test_the_target_actor_follows_the_actor_by_polyak_after_each_actor_step():
    learner = ddpg(polyak=0.25)
    fill_memory(learner)
    with torch.no_grad():
        for p in learner.target_policy.parameters():
            p.add_(1.0)
    target_before = params(learner.target_policy)
    learner.update(timestep=0, timesteps=10)
    policy_after = params(learner.policy)
    torch.testing.assert_close(params(learner.target_policy), target_before + 0.25 * (policy_after - target_before))
    # the Gaussian std the deterministic actor never uses takes no part (and is never optimized)
    assert all("log_std" not in name for name, p in learner.policy.named_parameters() if any(p is q for q in learner._actor_parameters()))


def test_the_critic_warm_up_trains_the_critics_alone_at_once_then_the_actor_joins():
    learner = ddpg(critic_warmup_updates=6, gradient_steps=2)
    fill_memory(learner)
    actor0, critics0 = actor_params(learner), params(learner.critics)
    learner._update_if_ready(timestep=0, timesteps=100)
    assert learner._updates == 6  # 6 critic gradient steps in one go
    torch.testing.assert_close(actor_params(learner), actor0)  # the actor did not move
    assert not torch.equal(params(learner.critics), critics0)
    learner._update_if_ready(timestep=1, timesteps=100)
    assert not torch.equal(actor_params(learner), actor0)  # from the next update on, it learns


def test_routed_phases_keep_their_own_warm_up_and_learning_starts():
    from robonuke_rl_core.learners.routed import RoutedSAC

    names = ["corner", "grasp", "insert", "screw", "success"]
    kw = dict(num_agents=1, envs_per_agent=2, rollout=64, architecture="mlp", std_by_label="eval_state/phase/id",
              std_label_names=names, route_clock=True, learning_starts_rows=6, critic_warmup_updates=2, random_action_scale=0.2,
              exploration_std=0.0)
    routed = RoutedSAC({r: build_learner("ddpg", route_label=r, **kw) for r in ("insert", "screw")}, names)
    routed.enable_training_mode(True)
    obs = torch.randn(2, OBS_DIM)
    for t in range(12):  # every env in insert: only insert's rows and clock advance
        actions, _ = routed.act(obs, None, timestep=t, timesteps=100)
        routed.record_transition(observations=obs, states=None, actions=actions, rewards=torch.zeros(2, 1),
                                 next_observations=obs, next_states=None, terminated=torch.zeros(2, 1, dtype=torch.bool),
                                 truncated=torch.zeros(2, 1, dtype=torch.bool),
                                 infos={"eval_state": {"phase/id": torch.full((2,), 2.0)}}, timestep=t, timesteps=100)
        routed.post_interaction(timestep=t, timesteps=100)
    insert, screw = routed.learners["insert"], routed.learners["screw"]
    assert insert._critic_warmed and insert._updates > 0 and not insert._random_phase(12)
    assert not screw._critic_warmed and screw._updates == 0 and screw._random_phase(12)  # screw has not started


def test_checkpoints_carry_the_target_actor(tmp_path):
    source = ddpg(run_dirs_list=[tmp_path / "a0", tmp_path / "a1"])
    fill_memory(source)
    source.update(timestep=0, timesteps=10)
    with torch.no_grad():
        for p in source.target_policy.parameters():
            p.add_(0.5)
    path = source.save_checkpoints(step=3)[1]
    target = ddpg()
    target.load_agent(path, slot=0)
    for (_, a), (_, b) in zip(source.target_policy.named_parameters(), target.target_policy.named_parameters()):
        torch.testing.assert_close(b[0], a[1])


def test_the_config_refuses_sac_only_options_and_bad_values():
    DDPGCfg().validate(None)
    for bad in ({"learn_entropy": True}, {"critic_distribution": "categorical"}, {"noise_repeat": True},
                {"cross_batch_value": True}, {"action_l2_weight": 1.0}, {"critic_loss": "huber"},
                {"q_target_min": 0.0}, {"actor_action_scale": 0.0}, {"actor_q_reduction": "min_random_pair"},
                {"exploration_std": -1.0}):
        with pytest.raises(ValueError):
            DDPGCfg(**bad).validate(None)


def test_building_without_a_target_actor_fails_loud():
    learner = ddpg()
    models = dict(learner.models)
    models.pop("target_policy")
    with pytest.raises(ValueError, match="target_policy"):
        type(learner)(models=models, memory=None, observation_space=learner.observation_space, state_space=None,
                      action_space=learner.action_space, device="cpu", cfg=learner.cfg, trainer_cfg=learner.trainer_cfg,
                      num_agents=2, num_envs=6)


def test_the_replay_memory_is_saved_beside_each_checkpoint_only_when_asked(tmp_path):
    """sac.save_memory_with_checkpoints (DDPG inherits it): memory_<step>.pt beside each numbered checkpoint, each
    agent's rows oldest first (the ring unrolled once it wrapped); off by default, and never announced for upload."""
    off = ddpg(run_dirs_list=[tmp_path / "off0", tmp_path / "off1"])
    fill_memory(off)
    off.save_checkpoints(step=5)
    assert not list((tmp_path / "off0" / "checkpoints").glob("memory_*.pt"))
    on = ddpg(run_dirs_list=[tmp_path / "a0", tmp_path / "a1"], save_memory_with_checkpoints=True)
    announced = []
    on.on_checkpoint.append(lambda agent, step, path: announced.append(path.name))
    fill_memory(on)
    memory = on.memory
    paths = on.save_checkpoints(step=7)
    for agent in range(2):
        saved = torch.load(tmp_path / f"a{agent}" / "checkpoints" / "memory_7.pt", weights_only=False)
        size = memory._agent_size[agent] if memory._agent_size is not None else memory.size
        assert saved["meta"]["rows"] == size and saved["meta"]["step"] == 7
        for name, tensor in memory.tensors.items():
            assert saved["rows"][name].shape == (size, tensor.shape[-1])
        if not saved["meta"]["wrapped"]:
            torch.testing.assert_close(saved["rows"]["observations"], memory.tensors["observations"][agent, :size].cpu())
    assert all(name.startswith("ckpt_") for name in announced) and len(paths) == 2
    # a wrapped ring: the saved rows run oldest to newest
    memory.size, memory.pointer = memory.capacity, 3
    if memory._agent_size is not None:
        memory._agent_size = [memory.capacity] * 2
        memory._agent_pointer = [3, 3]
    marks = torch.arange(memory.capacity, dtype=torch.float32)
    memory.tensors["rewards"][0, :, 0] = marks
    rows = torch.load(on.save_memory(0, 9), weights_only=False)["rows"]["rewards"][:, 0]
    torch.testing.assert_close(rows, torch.cat((marks[3:], marks[:3])))

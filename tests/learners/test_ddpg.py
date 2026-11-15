"""``trainer.learner: ddpg`` (learners/ddpg.py): the deterministic actor and its exploration, the critic target (target
actor, target-policy smoothing, a random-subset min), the actor objective and its reductions, the target actor's
Polyak step, ResFiT's warm-up actions and critic warm-up (per learner, so per routed phase), checkpoints, eval and
the config's refusals."""

from __future__ import annotations

import math

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner, fill_memory
from robonuke_rl_core.learners.cfg import DDPGCfg, NoiseCfg


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


def N(**kw):
    return NoiseCfg(**kw)


def started(learner):
    learner._agent_rows_ready = [True] * learner.num_agents
    return learner


def test_the_action_is_the_squashed_mean_on_the_range_plus_scheduled_noise_clamped():
    learner = ddpg(action_range=[-0.1, 0.1], train_noise=N(scale=0.4, scale_end=0.0, scale_steps=100), clip_actions=False)
    obs = torch.randn(6, OBS_DIM)
    with torch.no_grad():
        _, out = learner.policy.act({"observations": obs}, role="policy")
    mean = out["mean_actions"] * 0.1
    learner.enable_training_mode(False)
    eval_action, _ = learner.act(obs, None, timestep=0, timesteps=100)
    torch.testing.assert_close(eval_action, mean)  # eval: the scaled mean, no noise
    torch.testing.assert_close(learner.eval_actions(obs, deterministic=True), mean)
    with pytest.raises(ValueError, match="no noise"):
        learner.eval_actions(obs, deterministic=False)
    learner.enable_training_mode(True)
    assert learner.noise.scale("train", 0) == pytest.approx(0.4) and learner.noise.scale("train", 50) == pytest.approx(0.2)
    assert learner.noise.scale("train", 500) == pytest.approx(0.0)
    torch.manual_seed(0)
    noisy = torch.stack([learner.act(obs, None, timestep=50, timesteps=100)[0] for _ in range(400)])
    assert float((noisy - mean).std()) == pytest.approx(0.2, rel=0.1)  # the std at step 50
    clamped = ddpg(train_noise=N(scale=5.0))  # clip_actions (the default): clamped to [-1, 1]
    assert float(clamped.act(obs, None, timestep=1, timesteps=100)[0].abs().max()) <= 1.0


def test_correlated_noise_holds_each_agents_draw_for_its_drawn_length():
    learner = ddpg(train_noise=N(scale=0.3, repeat=True, repeat_max=8), clip_actions=False)
    learner.enable_training_mode(True)
    obs = torch.randn(6, OBS_DIM)
    with torch.no_grad():
        mean = learner.policy.act({"observations": obs}, role="policy")[1]["mean_actions"]
    torch.manual_seed(0)
    hold = learner.noise.held["train"]
    draws, lengths = [], []
    for step in range(200):
        actions, _ = learner.act(obs, None, timestep=step, timesteps=1000)
        torch.testing.assert_close(actions, mean + 0.3 * hold.held)  # the held draw is the noise
        draws.append(hold.held.clone())
        lengths.append(hold.length.clone())
    draws, lengths = torch.stack(draws), torch.stack(lengths)
    assert int(lengths.max()) <= 8
    for agent in range(2):
        block = draws[:, agent * 3:(agent + 1) * 3]
        changed = (block[1:] != block[:-1]).any(dim=(1, 2))
        starts = [0] + [t + 1 for t in range(len(changed)) if changed[t]]
        for start, end in zip(starts, starts[1:]):
            assert end - start == int(lengths[start, agent])  # each hold lasts its drawn length
    assert ddpg().noise.held["train"] is None  # off: a fresh draw every step


def test_resfits_warm_up_acts_uniformly_about_the_midpoint_until_this_learner_starts_learning():
    learner = ddpg(warmup_noise=N(center="mid", distribution="uniform", scale=0.2), train_noise=N(scale=0.0),
                   learning_starts_rows=10**6, action_range=[-0.1, 0.1], clip_actions=False)
    obs = torch.randn(6, OBS_DIM)
    torch.manual_seed(0)
    warm = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(200)])
    assert float(warm.abs().max()) <= 0.2 and float(warm.abs().max()) > 0.18  # uniform in [-0.2, 0.2]
    started(learner)  # its rows are there: the actor (here noise-free) from now on
    with torch.no_grad():
        _, out = learner.policy.act({"observations": obs}, role="policy")
    torch.testing.assert_close(learner.act(obs, None, timestep=300, timesteps=1000)[0], out["mean_actions"] * 0.1)


def test_a_gaussian_warm_up_acts_about_the_actors_own_action():
    # with model.actor.output_bias the actor starts at tanh(bias) for every state (a zero output layer); the warm-up draws
    # N(0, 0.05) about it, clamped to the range
    learner = ddpg(warmup_noise=N(scale=0.05), train_noise=N(scale=0.0), learning_starts_rows=10**6,
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


def test_a_uniform_phase_with_no_scale_covers_the_whole_range():
    learner = ddpg(action_range=[-1.5, 3.5], warmup_noise=N(center="mid", distribution="uniform", scale=None),
                   learning_starts_rows=10**6)
    assert learner.noise.scale("warmup", 0) == pytest.approx(2.5)  # the range's half width
    torch.manual_seed(0)
    warm = torch.stack([learner.act(torch.randn(6, OBS_DIM), None, timestep=t + 1, timesteps=100)[0] for t in range(300)])
    assert float(warm.min()) >= -1.5 and float(warm.max()) <= 3.5
    assert float(warm.min()) < -1.4 and float(warm.max()) > 3.4 and float(warm.mean()) == pytest.approx(1.0, abs=0.05)


def test_an_off_centre_range_starts_at_its_midpoint_and_clamps_to_its_ends():
    # a multiplier-style range: tanh(mu) = 0 is the midpoint 1, the actor's start with a zero output layer
    learner = started(ddpg(action_range=[-1.5, 3.5], train_noise=N(scale=10.0), model_overrides={"output_init_scale": 0.0}))
    obs = torch.randn(6, OBS_DIM)
    learner.enable_training_mode(False)
    torch.testing.assert_close(learner.act(obs, None, timestep=0, timesteps=100)[0], torch.full((6, ACT_DIM), 1.0))
    learner.enable_training_mode(True)
    torch.manual_seed(0)
    acted = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(50)])
    assert float(acted.min()) == pytest.approx(-1.5) and float(acted.max()) == pytest.approx(3.5)  # clamped to the range
    loose = started(ddpg(action_range=[-1.5, 3.5], train_noise=N(scale=10.0), clip_actions=False,
                         model_overrides={"output_init_scale": 0.0}))
    loose.enable_training_mode(True)
    wide = torch.stack([loose.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(50)])
    assert float(wide.max()) > 3.5 and float(wide.min()) < -1.5  # clip_actions: false leaves the noise unclamped


def test_the_uniform_warm_up_is_about_the_ranges_midpoint():
    learner = ddpg(action_range=[-1.5, 3.5], warmup_noise=N(center="mid", distribution="uniform", scale=0.5),
                   learning_starts_rows=10**6)
    obs = torch.randn(6, OBS_DIM)
    torch.manual_seed(0)
    warm = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(300)])
    assert float(warm.min()) >= 0.5 and float(warm.max()) <= 1.5  # 1 +- 0.5
    assert float(warm.mean()) == pytest.approx(1.0, abs=0.02)


def test_adaptive_noise_scales_each_rows_and_dims_draw_by_the_factor_of_its_observation():
    learner = started(ddpg(train_noise=N(scale=0.2, adaptive=True), clip_actions=False))
    learner.enable_training_mode(True)
    obs = torch.randn(6, OBS_DIM)
    with pytest.raises(RuntimeError, match="register_exploration_scale"):
        learner.act(obs, None, timestep=1, timesteps=100)  # adaptive with no factor: refused, never a silent constant
    factor = torch.ones(6, ACT_DIM)
    factor[0] = 0.0  # env 0 still: no noise
    factor[1, 0] = 5.0  # env 1, dim 0: five times the std
    seen = []
    learner.exploration_scale = lambda o: seen.append(o) or factor
    with torch.no_grad():
        mean = learner.policy.act({"observations": obs}, role="policy")[1]["mean_actions"]
    torch.manual_seed(0)
    noise = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(400)]) - mean
    assert seen[0] is obs  # the factor reads the raw observation the learner acts on
    assert float(noise[:, 0].abs().max()) == 0.0
    assert float(noise[:, 1, 0].std()) == pytest.approx(1.0, rel=0.1)
    assert float(noise[:, 2:].std()) == pytest.approx(0.2, rel=0.1)


def test_the_clip_bounds_the_scaled_draw_before_the_adaptive_factor():
    learner = started(ddpg(train_noise=N(scale=10.0, clip=0.1, adaptive=True), clip_actions=False))
    learner.exploration_scale = lambda o: torch.full((o.shape[0], ACT_DIM), 3.0)
    obs = torch.randn(6, OBS_DIM)
    with torch.no_grad():
        mean = learner.policy.act({"observations": obs}, role="policy")[1]["mean_actions"]
    noise = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(20)]) - mean
    assert float(noise.abs().max()) == pytest.approx(0.3)  # clipped to 0.1, then 3 x


def test_each_agent_acts_with_its_own_phase_and_each_phase_with_its_own_options():
    learner = ddpg(warmup_noise=N(scale=10.0, clip=0.1), train_noise=N(scale=10.0, clip=0.5), clip_actions=False,
                   learning_starts_rows=10**6)
    learner._agent_rows_ready = [False, True]  # agent 0 still warming up, agent 1 learning
    obs = torch.randn(6, OBS_DIM)
    with torch.no_grad():
        mean = learner.policy.act({"observations": obs}, role="policy")[1]["mean_actions"]
    noise = torch.stack([learner.act(obs, None, timestep=t + 1, timesteps=100)[0] for t in range(20)]) - mean
    assert float(noise[:, :3].abs().max()) == pytest.approx(0.1) and float(noise[:, 3:].abs().max()) == pytest.approx(0.5)


def test_base_only_episodes_act_with_the_neutral_action_and_no_noise():
    learner = ddpg(base_only_episodes=1, train_noise=N(scale=1.0))
    learner.neutral_action = 1.0
    actions, _ = learner.act(torch.randn(6, OBS_DIM), None, timestep=0, timesteps=100)
    assert torch.equal(actions, torch.ones_like(actions))


def test_the_noise_options_must_be_known():
    for bad in ({"center": "base"}, {"distribution": "normal"}, {"scale": None}, {"scale": -1.0}, {"clip": 0.0},
                {"repeat": True, "repeat_max": 0}, {"target_clip": 0.0},
                {"distribution": "uniform", "scale": None, "scale_end": 0.1}):
        with pytest.raises(ValueError):
            NoiseCfg(**bad).validate("ddpg.train_noise")
    with pytest.raises(ValueError, match="warmup_noise.distribution"):
        DDPGCfg(warmup_noise=N(distribution="normal")).validate(None)


def test_the_critic_target_is_r_plus_gamma_min_of_target_critics_at_the_target_actors_action():
    learner = ddpg(num_critics=2, critic_subset=2, discount_factor=0.9, action_range=[-0.5, 0.5])
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
        a_next = (out["mean_actions"] * 0.5).clamp(-0.5, 0.5)
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
    learner = ddpg(num_critics=1, train_noise=N(scale=10.0, target_smoothing=True, target_clip=1e-7))
    fill_memory(learner)
    sampled, inputs, next_inputs, critic_inputs, critic_next_inputs = batch(learner)
    rows = learner.cfg.batch_size

    def target():
        return learner._compute_critic_loss(sampled=sampled, inputs=inputs, next_inputs=next_inputs,
                                            critic_inputs=critic_inputs, critic_next_inputs=critic_next_inputs, rows=rows)[3]

    learner.noise.load_smoothing("train", 0)
    smoothed = target()
    learner.cfg.train_noise.target_smoothing = False
    learner.noise.load_smoothing("train", 0)
    torch.testing.assert_close(smoothed, target(), atol=1e-5, rtol=0)  # noise of std 10 clipped to 1e-7


def test_each_phase_loads_its_own_target_smoothing_and_the_adaptive_one_reads_the_next_observation():
    learner = ddpg(warmup_noise=N(scale=0.2, target_smoothing=True, target_scale=0.3, target_clip=0.5, target_adaptive=True),
                   train_noise=N(scale=0.4, scale_end=0.0, scale_steps=100, target_smoothing=True, target_clip=0.7))
    smooth = learner.noise._smooth
    learner.noise.load_smoothing("warmup", 0)  # the critic warm-up burst
    assert [float(smooth[k]) for k in ("on", "std", "clip", "adaptive")] == pytest.approx([1.0, 0.3, 0.5, 1.0])
    learner.noise.load_smoothing("train", 50)  # target_scale null: the phase's scale at that step
    assert [float(smooth[k]) for k in ("on", "std", "clip", "adaptive")] == pytest.approx([1.0, 0.2, 0.7, 0.0])
    learner.noise.load_smoothing("warmup", 0)
    next_obs = torch.randn(4, OBS_DIM)
    seen = []
    learner.exploration_scale = lambda o: seen.append(o) or torch.zeros(o.shape[0], ACT_DIM)
    out = learner.noise.smooth(torch.zeros(4, ACT_DIM), next_obs)
    assert seen[0] is next_obs and float(out.abs().max()) == 0.0  # a zero factor: no smoothing noise


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
    learner = ddpg(num_critics=3, critic_subset=2, actor_q_reduction=reduction, action_range=[-0.3, 0.3],
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


def test_the_critics_right_after_their_warm_up_are_saved_once_per_agent_whatever_the_step(tmp_path):
    learner = ddpg(critic_warmup_updates=4, learning_starts_rows=10**6, run_dirs_list=[tmp_path / "a0", tmp_path / "a1"],
                   trainer_overrides={"checkpoint_interval": 1000})
    fill_memory(learner)
    learner._agent_rows_ready = [False, True]  # agent 1 starts first
    learner._update_if_ready(timestep=37, timesteps=100)
    path1 = tmp_path / "a1" / "checkpoints" / "ckpt_critic_warmup.ckpt"
    assert path1.is_file() and not (tmp_path / "a0" / "checkpoints" / "ckpt_critic_warmup.ckpt").exists()
    saved = torch.load(path1, weights_only=False)
    assert saved["step"] == 37
    loaded = ddpg()
    loaded.load_agent(path1, slot=0)
    for (_, a), (_, b) in zip(learner.critics.named_parameters(), loaded.critics.named_parameters()):
        torch.testing.assert_close(b[0], a[1])  # exactly agent 1's critics after its burst
    learner._update_if_ready(timestep=38, timesteps=100)  # a normal update: the saved file stays as it was
    assert torch.load(path1, weights_only=False)["step"] == 37
    learner._agent_rows_ready = [True, True]
    learner._update_if_ready(timestep=90, timesteps=100)  # agent 0's own burst, at its own step
    assert torch.load(tmp_path / "a0" / "checkpoints" / "ckpt_critic_warmup.ckpt", weights_only=False)["step"] == 90
    assert torch.load(path1, weights_only=False)["step"] == 37


def test_the_memory_the_critic_warm_up_trained_on_is_saved_beside_its_checkpoint(tmp_path):
    learner = ddpg(critic_warmup_updates=4, learning_starts_rows=10**6, save_memory_with_critic_warmup=True,
                   run_dirs_list=[tmp_path / "a0", tmp_path / "a1"], trainer_overrides={"checkpoint_interval": 1000})
    fill_memory(learner)
    learner._agent_rows_ready = [False, True]  # agent 1 warms up first
    memory = learner.memory
    size = memory._agent_size[1] if memory._agent_size is not None else int(memory.size)
    expected = {name: tensor[1, :size].detach().cpu().clone() for name, tensor in memory.tensors.items()}
    learner._update_if_ready(timestep=37, timesteps=100)
    path = tmp_path / "a1" / "checkpoints" / "memory_critic_warmup.pt"
    assert path.is_file() and not (tmp_path / "a0" / "checkpoints" / "memory_critic_warmup.pt").exists()
    saved = torch.load(path, weights_only=False)
    assert saved["meta"]["step"] == 37 and saved["meta"]["agent"] == 1 and saved["meta"]["rows"] == size
    assert set(saved["rows"]) == set(expected)
    for name, rows in expected.items():
        torch.testing.assert_close(saved["rows"][name], rows)  # exactly agent 1's rows when its burst ran


def test_without_the_option_no_memory_is_saved_with_the_warm_up_checkpoint(tmp_path):
    learner = ddpg(critic_warmup_updates=4, run_dirs_list=[tmp_path / "a0", tmp_path / "a1"],
                   trainer_overrides={"checkpoint_interval": 1000})
    fill_memory(learner)
    learner._update_if_ready(timestep=5, timesteps=100)
    assert (tmp_path / "a0" / "checkpoints" / "ckpt_critic_warmup.ckpt").is_file()
    assert not list(tmp_path.rglob("memory_*.pt"))


def test_saving_the_warm_up_memory_needs_a_warm_up():
    with pytest.raises(ValueError, match="save_memory_with_critic_warmup needs critic_warmup_updates"):
        DDPGCfg(save_memory_with_critic_warmup=True).validate(None)
    DDPGCfg(save_memory_with_critic_warmup=True, critic_warmup_updates=10).validate(None)


def test_routed_phases_keep_their_own_warm_up_and_learning_starts():
    from robonuke_rl_core.learners.routed import RoutedSAC

    names = ["corner", "grasp", "insert", "screw", "success"]
    kw = dict(num_agents=1, envs_per_agent=2, rollout=64, architecture="mlp", std_by_label="eval_state/phase/id",
              std_label_names=names, route_clock=True, learning_starts_rows=6, critic_warmup_updates=2,
              warmup_noise=N(center="mid", distribution="uniform", scale=0.2), train_noise=N(scale=0.0))
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


def test_shared_warmup_pools_every_envs_rows_until_the_agents_start_together_then_each_keeps_its_own():
    from robonuke_rl_core.learners.routed import RoutedSAC

    names = ["corner", "grasp", "insert", "screw", "success"]
    kw = dict(num_agents=2, envs_per_agent=1, rollout=64, architecture="mlp", std_by_label="eval_state/phase/id",
              std_label_names=names, route_clock=True, learning_starts_rows=6, critic_warmup_updates=2,
              warmup_noise=N(center="mid", distribution="uniform", scale=0.2), train_noise=N(scale=0.0), shared_warmup=True)
    routed = RoutedSAC({r: build_learner("ddpg", route_label=r, **kw) for r in ("insert", "screw")}, names)
    routed.enable_training_mode(True)
    insert = routed.learners["insert"]

    def step(t, phases):
        obs = torch.full((2, OBS_DIM), float(t))
        actions, _ = routed.act(obs, None, timestep=t, timesteps=100)
        routed.record_transition(observations=obs, states=None, actions=actions, rewards=torch.zeros(2, 1),
                                 next_observations=obs, next_states=None, terminated=torch.zeros(2, 1, dtype=torch.bool),
                                 truncated=torch.zeros(2, 1, dtype=torch.bool),
                                 infos={"eval_state": {"phase/id": torch.tensor(phases, dtype=torch.float32)}},
                                 timestep=t, timesteps=100)
        routed.post_interaction(timestep=t, timesteps=100)

    step(0, [2.0, 1.0])  # labels known from here: agent 0's env in insert, agent 1's in grasp
    for t in range(1, 7):
        step(t, [2.0, 1.0])  # only agent 0's env is in insert; its rows go to both agents' insert memories
    assert insert.memory._agent_size == [6, 6]
    assert insert._agent_route_steps == [6, 6]  # the pooled steps count on every agent's clock
    first = insert.memory.get_tensor_by_name("observations")[:, :6, 0]
    torch.testing.assert_close(first[0], first[1])  # the same pooled rows
    assert insert._agent_warmed == [True, True]  # one burst, both agents, though agent 1 never was in insert
    assert insert._agent_updates[0] == insert._agent_updates[1] > 0
    for t in range(7, 10):
        step(t, [2.0, 1.0])  # learning: each agent keeps only its own envs' rows
    assert insert.memory._agent_size == [9, 6]
    assert insert._agent_route_steps == [9, 6]  # and counts only its own steps again


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
    DDPGCfg(train_noise=N(repeat=True)).validate(None)  # held noise, as SAC's noise_repeat
    DDPGCfg(shared_warmup=True, learning_starts_rows=10).validate(None)
    with pytest.raises(ValueError, match="learning_starts_rows"):
        DDPGCfg(shared_warmup=True).validate(None)
    for bad in ({"learn_entropy": True}, {"critic_distribution": "categorical"}, {"noise_repeat": True},
                {"cross_batch_value": True}, {"action_l2_weight": 1.0}, {"critic_loss": "huber"},
                {"q_target_min": 0.0}, {"action_range": [0.5, 0.5]}, {"action_range": [-1.0]},
                {"actor_q_reduction": "min_random_pair"}, {"train_noise": N(scale=-1.0)}):
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



def test_the_critic_warm_up_publishes_only_its_last_updates():
    learner = ddpg(critic_warmup_updates=40, gradient_steps=2)
    learner.WARMUP_LOGGED_UPDATES = 8  # the last 8 gradient steps: 4 update points of 2
    fill_memory(learner)
    points = []
    learner.on_log.append(lambda agent, metrics, step: points.append(dict(metrics)) if "loss/critic" in metrics else None)
    learner._update_if_ready(timestep=0, timesteps=100)
    assert learner._updates == 40
    assert len(points) == 4 * learner.num_agents  # one point per agent for each of the burst's last 4 updates
    points.clear()
    learner._update_if_ready(timestep=1, timesteps=100)
    assert len(points) == learner.num_agents  # an ordinary update publishes as before


def test_the_critic_health_metrics():
    learner = ddpg()
    rows = 4
    y = torch.tensor([0.0, 1.0, 2.0, 3.0] * learner.num_agents).reshape(-1, 1)
    perfect = learner.critic_health_metrics(y.clone(), y, rows)
    torch.testing.assert_close(perfect["q/explained_variance"], torch.ones(learner.num_agents))
    torch.testing.assert_close(perfect["q/td_abs_max"], torch.zeros(learner.num_agents))
    flat = learner.critic_health_metrics(torch.full_like(y, 1.5), y, rows)
    torch.testing.assert_close(flat["q/explained_variance"], torch.zeros(learner.num_agents))  # no better than the mean
    torch.testing.assert_close(flat["q/td_abs_mean"], torch.full((learner.num_agents,), 1.0))
    # rows 0-1 online (fitted exactly), rows 2-3 demo (each off by 1): every metric on all rows and on each part
    pred = y.clone().reshape(learner.num_agents, rows)
    pred[:, 2:] += torch.tensor([1.0, -1.0])
    split = learner.critic_health_metrics(pred.reshape(-1, 1), y, rows, demo_rows=2)
    a = learner.num_agents
    torch.testing.assert_close(split["q/explained_variance_online"], torch.ones(a))
    torch.testing.assert_close(split["q/td_abs_max_online"], torch.zeros(a))
    torch.testing.assert_close(split["q/td_abs_mean_demo"], torch.ones(a))
    torch.testing.assert_close(split["q/explained_variance_demo"], torch.full((a,), -3.0))  # Var(err) 1 vs Var(y) 0.25
    torch.testing.assert_close(split["q/td_abs_mean"], torch.full((a,), 0.5))
    torch.testing.assert_close(split["q/mean_online"], torch.full((a,), 0.5))
    torch.testing.assert_close(split["q/mean_demo"], torch.full((a,), 2.5))
    assert "q/mean_online" not in learner.critic_health_metrics(y.clone(), y, rows)  # no demo rows: all rows only


def test_an_update_publishes_the_critic_health_and_the_action_slope():
    learner = ddpg(policy_delay=1, log_action_grad=True)
    fill_memory(learner)
    seen = {}
    learner.on_log.append(lambda agent, metrics, step: seen.update(metrics))
    learner._update_if_ready(timestep=0, timesteps=100)
    for key in ("q/explained_variance", "q/td_abs_mean", "q/td_abs_max", "q/action_grad_norm"):
        assert key in seen, key
        assert torch.isfinite(torch.as_tensor(seen[key])).all()


def test_the_action_slope_is_off_by_default():
    learner = ddpg(policy_delay=1)
    fill_memory(learner)
    seen = {}
    learner.on_log.append(lambda agent, metrics, step: seen.update(metrics))
    learner._update_if_ready(timestep=0, timesteps=100)
    assert "q/explained_variance" in seen and "q/action_grad_norm" not in seen


def test_critic_values_scores_given_transitions_per_slot_as_the_update_does():
    learner = ddpg(num_critics=2, critic_subset=2)  # no target smoothing, the target min over both: not random
    fill_memory(learner)
    names = learner._demo_names() + (["discounts"] if learner._n_step() > 1 else [])
    size = int(learner.memory.size)
    sets = []
    for agent, rows in enumerate((7, 3)):  # uneven sets per slot, chunked below their size
        sets.append({n: learner.memory.get_tensor_by_name(n)[agent, :rows].clone() for n in names})
    out = learner.critic_values(sets, chunk=2)
    assert out[0][0].shape == (7,) and out[1][0].shape == (3,)
    # the same rows through the update's own critic computation, slot by slot
    for agent, rows in enumerate((7, 3)):
        batch = {n: torch.cat([sets[agent][n]] * learner.num_agents) for n in names}
        inputs, next_inputs, critic_inputs, critic_next_inputs = learner._batch_inputs(batch, train=False)
        with torch.no_grad():
            _, q1, q2, target, _, _ = learner._compute_critic_loss(sampled=batch, inputs=inputs, next_inputs=next_inputs,
                                                                  critic_inputs=critic_inputs,
                                                                  critic_next_inputs=critic_next_inputs, rows=rows)
        q = learner._critic_prediction(q1, q2).reshape(learner.num_agents, rows)[agent]
        torch.testing.assert_close(out[agent][0], q)
        torch.testing.assert_close(out[agent][1], target.reshape(learner.num_agents, rows)[agent])
    assert learner.critic_values([None, sets[1]])[0] is None
    assert size > 0

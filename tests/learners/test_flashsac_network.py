"""``model.architecture: flashsac`` (``models/flashsac.py``): FlashSAC's BatchNorm network under the
package's per-agent vmap, its running statistics, and SAC's per-call modes."""

from __future__ import annotations

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner, fill_memory

FLASH = dict(
    architecture="flashsac", critic_distribution="categorical", critic_num_bins=11, adaptive_reward_scaling=True,
    weight_normalization=True, target_entropy_sigma=0.15, initial_entropy_value=0.01, entropy_loss_form="alpha",
    cross_batch_value=True, noise_repeat=True, normalize_observations=False, gradient_steps=2, policy_delay=2,
)


def running(model):
    _, buffers = model.net._stacked()
    return {k: v.clone() for k, v in buffers.items() if "running" in k}


def test_every_flashsac_piece_trains_together():
    learner = build_learner("sac", num_agents=2, **FLASH)
    fill_memory(learner)
    before = [p.detach().clone() for p in learner.critics.parameters()] + [p.detach().clone() for p in learner.policy.parameters()]
    for step in range(3):
        learner.update(timestep=step, timesteps=100)
    after = list(learner.critics.parameters()) + list(learner.policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)


def test_training_calls_update_the_running_statistics_and_acting_does_not():
    learner = build_learner("sac", num_agents=2, **FLASH)
    fill_memory(learner)
    actor0, critic0, target0 = running(learner.policy), running(learner.critics), running(learner.target_critics)
    learner.enable_training_mode(False)
    learner.act(torch.randn(learner.num_envs, OBS_DIM), None, timestep=0, timesteps=10)  # eval: running stats
    assert all(torch.equal(v, running(learner.policy)[k]) for k, v in actor0.items())
    learner.enable_training_mode(True)
    learner.update(timestep=1, timesteps=100)
    for start, model in ((actor0, learner.policy), (critic0, learner.critics), (target0, learner.target_critics)):
        now = running(model)
        assert any(not torch.equal(v, now[k]) for k, v in start.items())


def test_an_agents_batch_statistics_come_from_its_own_rows():
    learner = build_learner("sac", num_agents=2, **FLASH)
    learner.policy.enable_training_mode(True)
    obs = torch.randn(2 * 6, OBS_DIM)
    other = obs.clone()
    other[6:] = other[6:] * 50 + 10  # only agent 1's rows change
    a, _ = learner.policy.compute({"observations": obs}, "policy")
    b, _ = learner.policy.compute({"observations": other}, "policy")
    torch.testing.assert_close(a[:6], b[:6])
    assert not torch.allclose(a[6:], b[6:])


def test_the_mode_switch_is_per_call_and_restored():
    learner = build_learner("sac", num_agents=1, **FLASH)
    learner.policy.enable_training_mode(True)
    with learner._batch_mode(learner.policy, False):
        assert learner.policy.training is False
    assert learner.policy.training is True


def test_the_heads_and_blocks_are_weight_normalized():
    learner = build_learner("sac", num_agents=1, **FLASH)
    params, _ = learner.policy.net._stacked()
    for name in ("trunk.embed.weight", "trunk.blocks.0.w1.weight", "mean.weight", "log_std.weight"):
        norms = params[name].norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))
    scale, shift = params["trunk.embed_norm.weight"], params["trunk.embed_norm.bias"]
    torch.testing.assert_close((scale**2 + shift**2).sum(-1).sqrt(), torch.full((1,), OBS_DIM**0.5))


def test_the_architecture_is_sac_only(tmp_path):
    from robonuke_rl_core.models.cfg import FlashSACModelCfg
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="SAC network"):
        FlashSACModelCfg().validate(SimpleNamespace(trainer=SimpleNamespace(learner="ppo")))


def test_a_small_mean_init_starts_the_residual_near_zero_and_survives_weight_normalization():
    learner = build_learner("sac", num_agents=2, model_overrides={"mean_init_scale": 0.01}, **FLASH)
    params, _ = learner.policy.net._stacked()
    norms = params["mean.weight"].norm(dim=-1)
    torch.testing.assert_close(norms, torch.full_like(norms, 0.01))  # orthogonal gain 0.01, not reset to 1
    learner.enable_training_mode(False)
    with torch.no_grad():
        mean, _ = learner.policy.compute({"observations": torch.randn(64, OBS_DIM)}, "policy")
    assert mean.abs().max() < 0.1
    learner.enable_training_mode(True)
    fill_memory(learner)
    learner.update(timestep=1, timesteps=100)
    params, _ = learner.policy.net._stacked()
    assert params["mean.weight"].norm(dim=-1).max() < 0.5  # still out of weight normalization
    other = params["log_std.weight"].norm(dim=-1)
    torch.testing.assert_close(other, torch.ones_like(other))  # the rest keeps FlashSAC's unit rows


def test_init_std_starts_every_state_at_that_std_and_it_can_still_learn_state_dependence():
    learner = build_learner("sac", num_agents=2, model_overrides={"mean_init_scale": 0.01, "init_std": 0.15}, **FLASH)
    learner.enable_training_mode(False)
    with torch.no_grad():
        _, out = learner.policy.compute({"observations": torch.randn(64, OBS_DIM) * 3}, "policy")
    std = out["log_std"].exp()
    assert ((std - 0.15).abs() / 0.15).max() < 0.15  # every state, every dim, about 0.15 (gain-0.01 weights)
    learner.enable_training_mode(True)
    fill_memory(learner)
    for step in range(3):
        learner.update(timestep=step, timesteps=100)
    params, _ = learner.policy.net._stacked()
    assert params["log_std.weight"].norm(dim=-1).max() < 0.5  # out of weight normalization, learning from small


def test_init_std_must_lie_inside_the_bound():
    from types import SimpleNamespace

    from robonuke_rl_core.models.cfg import FlashSACActorCfg, FlashSACModelCfg

    with pytest.raises(ValueError, match="init_std"):
        FlashSACModelCfg(actor=FlashSACActorCfg(init_std=10.0)).validate(SimpleNamespace(trainer=SimpleNamespace(learner="sac")))

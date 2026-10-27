"""FlashSAC's pieces in SAC (``sac.critic_distribution`` .. ``sac.noise_repeat``): each checked
against FlashSAC's own code (github.com/Holiday-Robot/FlashSAC, ``flash_rl/agents``),
transcribed here as the references, and checked to be off by default."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.learners.cfg import SACCfg, sac_critic_outputs
from robonuke_rl_core.learners.sac import categorical_projection
from robonuke_rl_core.models.factory import build_models
from robonuke_rl_core.models.normalizer import BlockReturnScaler

from helpers import ACT_DIM, OBS_DIM, box, build_learner, fill_memory, tiny_model_cfg


# ---------------------------------------------------------------------- FlashSAC's references
def flashsac_projection(target_log_probs, reward, done, actor_entropy, gamma, num_bins, min_v, max_v):
    """``flashSAC/update.py::_compute_categorical_td_target``, verbatim but for the decorator."""
    batch_size = reward.shape[0]
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(min_v, max_v, num_bins, dtype=target_log_probs.dtype).view(1, -1)
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    frac = b - lower.float()
    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac
    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype)
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)
    return target_probs


class FlashSACRewardNormalizer:
    """``utils/reward_normalization.py``'s ``RewardNormalizer`` + ``RunningMeanStd``, one agent."""

    def __init__(self, gamma, g_max, epsilon=1e-8):
        self.gamma, self.g_max, self.epsilon = gamma, g_max, epsilon
        self.G_r = torch.zeros(1)
        self.G_r_max = torch.zeros(1)
        self.mean, self.var, self.count, self.rms_eps = torch.zeros(1), torch.ones(1), torch.tensor(0.0), 1e-4

    def update(self, reward, terminated, truncated):
        done = torch.logical_or(terminated, truncated).float()
        self.G_r = self.gamma * (1.0 - done) * self.G_r + reward
        self.G_r_max = torch.maximum(self.G_r_max, torch.max(torch.abs(self.G_r)))
        x = self.G_r
        sample_mean = torch.mean(x, dim=0)
        sample_var = torch.var(x, dim=0, unbiased=False)
        sample_count = float(x.shape[0])
        delta = sample_mean - self.mean
        total = self.count + sample_count
        ratio = sample_count / total
        new_mean = self.mean + delta * ratio
        m2 = self.var * (self.count + self.rms_eps) + sample_var * sample_count + torch.square(delta) * self.count * ratio
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def denominator(self):
        return torch.maximum(torch.sqrt(self.var + self.epsilon), self.G_r_max / self.g_max)


def flashsac_zeta_cdf(mu, max_n):
    """``flashSAC/agent.py::_build_truncated_zeta_cdf``."""
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


# ---------------------------------------------------------------------- defaults
def test_every_flashsac_option_is_off_by_default():
    cfg = SACCfg()
    assert cfg.critic_distribution == "scalar"
    assert not cfg.adaptive_reward_scaling
    assert not cfg.weight_normalization
    assert cfg.target_entropy_sigma is None
    assert not cfg.cross_batch_value
    assert not cfg.noise_repeat
    assert sac_critic_outputs(cfg) == 1


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"critic_distribution": "quantile"}, "critic_distribution"),
        ({"critic_distribution": "categorical", "critic_num_bins": 1}, "critic_num_bins"),
        ({"critic_distribution": "categorical", "critic_min_value": 5.0}, "critic_min_value"),
        ({"critic_distribution": "categorical", "num_critics": 4, "critic_subset": 2}, "critic_subset"),
        ({"adaptive_reward_scaling": True, "adaptive_reward_g_max": 0.0}, "adaptive_reward_g_max"),
        ({"adaptive_reward_scaling": True, "normalize_rewards": True}, "use one"),
        ({"target_entropy_sigma": 0.15, "target_entropy": -2.0}, "set one"),
        ({"target_entropy_sigma": 0.0}, "target_entropy_sigma"),
        ({"noise_repeat": True, "noise_repeat_max": 0}, "noise_repeat_max"),
        ({"noise_repeat": True, "noise_repeat_zeta_s": 0.0}, "noise_repeat_zeta_s"),
    ],
)
def test_the_flashsac_fields_fail_loud_on_bad_values(overrides, message):
    cfg = SACCfg(**overrides)
    with pytest.raises(ValueError, match=message):
        cfg.validate(None)


# ---------------------------------------------------------------------- categorical critic
def test_the_projection_matches_flashsacs_and_keeps_the_mass():
    torch.manual_seed(0)
    rows, bins = 64, 101
    log_probs = torch.log_softmax(torch.randn(rows, bins), dim=-1)
    reward = torch.randn(rows) * 3.0
    done = (torch.rand(rows) < 0.2).float()
    entropy = torch.randn(rows) * 0.1
    gamma = 0.97
    expected = flashsac_projection(log_probs, reward, done, entropy, gamma, bins, -5.0, 5.0)
    atoms = torch.linspace(-5.0, 5.0, bins)
    shifted = reward[:, None] + gamma * (1.0 - done[:, None]) * (atoms - entropy[:, None])
    got = categorical_projection(log_probs.exp(), shifted, -5.0, 5.0)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got.sum(dim=-1), torch.ones(rows))


def test_a_terminal_target_is_the_reward_on_the_support():
    bins = 11
    atoms = torch.linspace(-5.0, 5.0, bins)
    probs = torch.softmax(torch.randn(3, bins), dim=-1)
    rewards = torch.tensor([[2.0], [2.5], [9.0]])  # on an atom, between two, beyond the support
    got = categorical_projection(probs, rewards.expand(-1, bins), -5.0, 5.0)
    assert got[0, 7] == pytest.approx(1.0)
    assert got[1, 7] == pytest.approx(0.5) and got[1, 8] == pytest.approx(0.5)
    assert got[2, -1] == pytest.approx(1.0)
    torch.testing.assert_close((got * atoms).sum(-1), torch.tensor([2.0, 2.5, 5.0]))


@pytest.mark.parametrize("architecture", ["simba", "mlp"])
def test_a_categorical_critic_emits_its_bins_and_trains(architecture):
    learner = build_learner(
        "sac", architecture=architecture, critic_distribution="categorical", critic_num_bins=21
    )
    fill_memory(learner)
    observations = torch.randn(learner.num_agents * 4, OBS_DIM)
    actions = torch.rand(learner.num_agents * 4, ACT_DIM) * 2 - 1
    out, _ = learner.critics.act({"observations": observations, "taken_actions": actions}, role="critics")
    assert out.shape == (2, learner.num_agents * 4, 21)
    before = [p.detach().clone() for p in learner.critics.parameters()]
    learner.update(timestep=1, timesteps=100)
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.critics.parameters()))


def test_the_categorical_target_is_flashsacs_from_the_lowest_q_critic():
    learner = build_learner(
        "sac", num_agents=1, critic_distribution="categorical", critic_num_bins=11,
        normalize_observations=False, initial_entropy_value=0.3, discount_factor=0.9,
    )
    fill_memory(learner)
    rows = 4
    sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=rows)[0]))
    sampled["terminated"][0] = True  # one terminal row
    inputs = {"observations": sampled["observations"]}
    next_inputs = {"observations": sampled["next_observations"]}
    torch.manual_seed(0)
    loss, _, _, target_values, next_log_prob, ce = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs,
        critic_next_inputs=next_inputs, rows=rows,
    )
    torch.manual_seed(0)
    next_actions, outputs = learner.policy.act(next_inputs, role="policy")
    logits, _ = learner.target_critics.act({**next_inputs, "taken_actions": next_actions}, role="target_critics")
    log_probs = torch.log_softmax(logits, -1)
    atoms = torch.linspace(-5.0, 5.0, 11)
    qs = (log_probs.exp() * atoms).sum(-1)
    chosen = log_probs[qs.argmin(0), torch.arange(rows)]
    expected = flashsac_projection(
        chosen, sampled["rewards"].view(-1), sampled["terminated"].float().view(-1),
        0.3 * outputs["log_prob"].view(-1), 0.9, 11, -5.0, 5.0,
    )
    torch.testing.assert_close(target_values, (expected * atoms).sum(-1, keepdim=True))
    pred, _ = learner.critics.act({**inputs, "taken_actions": sampled["actions"]}, role="critics")
    expected_loss = -(expected.unsqueeze(0) * torch.log_softmax(pred, -1)).sum(-1).mean()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(ce, expected_loss.view(1))


def test_a_critic_built_with_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="critic_outputs"):
        learner = build_learner("sac", num_agents=1)
        type(learner)(
            models=learner.models, memory=None, observation_space=box(OBS_DIM), state_space=None,
            action_space=box(ACT_DIM), device="cpu",
            cfg=SACCfg(critic_distribution="categorical"), trainer_cfg=learner.trainer_cfg,
            num_agents=1, num_envs=2,
        )


def test_a_categorical_critic_must_start_at_zero_logits():
    model = tiny_model_cfg()
    model.critic.critic_output_init_mean = 1.0
    with pytest.raises(ValueError, match="critic_output_init_mean"):
        build_models("sac", model, box(OBS_DIM), None, box(ACT_DIM), 1, "cpu", critic_outputs=11)


def test_ppo_refuses_a_categorical_width():
    with pytest.raises(ValueError, match="critic_outputs"):
        build_models("ppo", tiny_model_cfg(), box(OBS_DIM), None, box(ACT_DIM), 1, "cpu", critic_outputs=11)


# ---------------------------------------------------------------------- adaptive reward scaling
def test_the_return_scaler_is_flashsacs_per_agent():
    torch.manual_seed(0)
    agents, envs, gamma, g_max = 2, 3, 0.97, 5.0
    scaler = BlockReturnScaler(agents, gamma=gamma, g_max=g_max)
    references = [FlashSACRewardNormalizer(gamma, g_max) for _ in range(agents)]
    for step in range(200):
        rewards = torch.randn(agents * envs, 1) - 0.5
        done = torch.rand(agents * envs, 1) < 0.05
        scaler.update(rewards, done)
        for agent, reference in enumerate(references):
            block = slice(agent * envs, (agent + 1) * envs)
            reference.update(rewards[block, 0], done[block, 0], torch.zeros(envs, dtype=torch.bool))
    for agent, reference in enumerate(references):
        torch.testing.assert_close(scaler.denominator[agent], reference.denominator())


def test_an_agents_scale_depends_on_its_own_envs_only():
    a = BlockReturnScaler(2, gamma=0.9, g_max=5.0)
    b = BlockReturnScaler(2, gamma=0.9, g_max=5.0)
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    other = rewards.clone()
    other[2:] *= 100.0
    done = torch.zeros(4, 1, dtype=torch.bool)
    for _ in range(5):
        a.update(rewards, done)
        b.update(other, done)
    torch.testing.assert_close(a.denominator[0], b.denominator[0])
    assert not torch.equal(a.denominator[1], b.denominator[1])


def test_the_critics_reward_is_divided_and_the_memory_keeps_it_raw():
    learner = build_learner(
        "sac", num_agents=2, adaptive_reward_scaling=True, normalize_observations=False, backup_entropy=False
    )
    fill_memory(learner)
    assert torch.all(learner.return_scaler.denominator != 1.0)
    rows = 4
    sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=rows)[0]))
    sampled["terminated"][:] = True  # the target is the reward alone
    sampled["truncated"][:] = False
    inputs = {"observations": sampled["observations"]}
    next_inputs = {"observations": sampled["next_observations"]}
    _, _, _, target, _, _ = learner._compute_critic_loss(
        sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=inputs,
        critic_next_inputs=next_inputs, rows=rows,
    )
    denominator = learner.return_scaler.denominator.repeat_interleave(rows, dim=0)
    torch.testing.assert_close(target, sampled["rewards"] / denominator)


def test_the_return_scaler_round_trips_through_a_checkpoint(tmp_path):
    learner = build_learner("sac", num_agents=2, adaptive_reward_scaling=True)
    fill_memory(learner)
    state = learner.return_scaler.state_dict_for(1)
    fresh = BlockReturnScaler(2, gamma=0.99, g_max=5.0)
    fresh.load_state_dict_into(0, state)
    torch.testing.assert_close(fresh.denominator[0], learner.return_scaler.denominator[1])
    torch.testing.assert_close(fresh.returns[0], learner.return_scaler.returns[1])
    assert "returns" in learner._checkpoint_normalizers()


# ---------------------------------------------------------------------- weight normalization
def _check_normalized(model):
    net = model.net
    params, _ = net._stacked()
    for name, module in net._meta.named_modules():
        prefix = f"{name}." if name else ""
        if isinstance(module, nn.Linear):
            norms = params[f"{prefix}weight"].norm(dim=-1)
            torch.testing.assert_close(norms, torch.ones_like(norms))
        elif isinstance(module, nn.LayerNorm):
            scale, shift = params[f"{prefix}weight"], params[f"{prefix}bias"]
            norms = (scale.square() + shift.square()).sum(-1).sqrt()
            torch.testing.assert_close(norms, torch.full_like(norms, math.sqrt(scale.shape[-1])))


@pytest.mark.parametrize("architecture", ["simba", "mlp"])
def test_weight_normalization_holds_at_build_and_after_updates(architecture):
    learner = build_learner("sac", architecture=architecture, weight_normalization=True)
    for model in (learner.policy, learner.critics, learner.target_critics):
        _check_normalized(model)
    fill_memory(learner)
    for step in range(3):
        learner.update(timestep=step, timesteps=100)
    _check_normalized(learner.policy)
    _check_normalized(learner.critics)


def test_weight_normalization_leaves_the_biases():
    learner = build_learner("sac", num_agents=1)
    learner.policy.net.normalize_weights()
    params, _ = learner.policy.net._stacked()
    _check_normalized(learner.policy)
    plain = build_learner("sac", num_agents=1)
    plain_params, _ = plain.policy.net._stacked()
    torch.testing.assert_close(params["trunk.fc_in.bias"], plain_params["trunk.fc_in.bias"])


# ---------------------------------------------------------------------- unified entropy target
def test_the_unified_entropy_target_is_a_gaussians_entropy():
    learner = build_learner("sac", num_agents=1, action_dim=6, target_entropy_sigma=0.15)
    expected = 0.5 * 6 * math.log(2 * math.pi * math.e * 0.15**2)
    assert learner._target_entropy == pytest.approx(expected)
    torch.manual_seed(0)
    gaussian = torch.distributions.Normal(torch.zeros(6), torch.full((6,), 0.15)).entropy().sum()
    assert learner._target_entropy == pytest.approx(float(gaussian), rel=1e-6)


# ---------------------------------------------------------------------- cross-batch value
@pytest.mark.parametrize("categorical", [False, True])
def test_cross_batch_changes_nothing_without_batch_statistics(categorical):
    extra = {"critic_distribution": "categorical", "critic_num_bins": 11} if categorical else {}
    results = []
    for cross in (False, True):
        learner = build_learner("sac", num_agents=2, cross_batch_value=cross, **extra)
        fill_memory(learner)
        rows = 4
        sampled = dict(zip(learner._tensors_names, learner.memory.sample(names=learner._tensors_names, batch_size=rows)[0]))
        inputs, next_inputs, critic_inputs, critic_next_inputs = learner._batch_inputs(sampled, train=False)
        torch.manual_seed(3)
        critic = learner._compute_critic_loss(
            sampled=sampled, inputs=inputs, next_inputs=next_inputs, critic_inputs=critic_inputs,
            critic_next_inputs=critic_next_inputs, rows=rows,
        )
        torch.manual_seed(4)
        # the joined actor call samples s ++ s' at once: hold its noise fixed to compare
        noise = torch.randn(2 * 2 * rows, ACT_DIM)
        if cross:
            actor_inputs = {**inputs, "noise": noise}
            actor_next = {**next_inputs, "noise": noise}
        else:
            current = noise.view(2, 2, rows, ACT_DIM)[:, 0].reshape(-1, ACT_DIM)
            actor_inputs, actor_next = {**inputs, "noise": current}, next_inputs
        actor = learner._compute_actor_loss(
            sampled=sampled, inputs=actor_inputs, critic_inputs=critic_inputs, rows=rows, next_inputs=actor_next
        )
        results.append((critic, actor))
    (critic_a, actor_a), (critic_b, actor_b) = results
    for x, y in zip(critic_a, critic_b):
        if x is not None:
            torch.testing.assert_close(x, y)
    for x, y in zip(actor_a[:4], actor_b[:4]):
        torch.testing.assert_close(x, y)


def test_joined_halves_keep_each_agents_rows_together():
    learner = build_learner("sac", num_agents=2)
    current = torch.arange(8.0).view(8, 1)
    following = current + 100.0
    joined = learner._join_halves(current, following, rows=4)
    assert joined.view(-1).tolist() == [0, 1, 2, 3, 100, 101, 102, 103, 4, 5, 6, 7, 104, 105, 106, 107]
    out = joined.view(1, 16, 1)
    torch.testing.assert_close(learner._critic_half(out, 4, 0)[0], current)
    torch.testing.assert_close(learner._critic_half(out, 4, 1)[0], following)
    torch.testing.assert_close(learner._actor_half(joined, 4), current)


# ---------------------------------------------------------------------- noise repetition
def test_the_zeta_lengths_are_flashsacs():
    learner = build_learner("sac", num_agents=1, noise_repeat=True)
    torch.testing.assert_close(learner._zeta_cdf, flashsac_zeta_cdf(2.0, 16))


def test_held_noise_is_piecewise_constant_per_agent_and_drives_the_action():
    learner = build_learner("sac", num_agents=2, envs_per_agent=3, noise_repeat=True, normalize_observations=False)
    torch.manual_seed(0)
    observations = torch.randn(6, OBS_DIM)
    draws, lengths = [], []
    for step in range(300):
        actions, outputs = learner.act(observations, None, timestep=step, timesteps=1000)
        draws.append(learner._held_noise.clone())
        lengths.append(learner._noise_length.clone())
        mean, log_std = learner.policy.net.forward_flat(observations)
        torch.testing.assert_close(actions, torch.tanh(mean + log_std.exp() * learner._held_noise))
    draws = torch.stack(draws)
    lengths = torch.stack(lengths)
    assert int(lengths.min()) >= 1 and int(lengths.max()) <= 16
    for agent in range(2):
        block = draws[:, agent * 3:(agent + 1) * 3]
        changed = (block[1:] != block[:-1]).any(dim=(1, 2))
        # one agent's envs switch together; each hold lasts exactly its drawn length
        starts = [0] + [t + 1 for t in range(len(changed)) if changed[t]]
        for start, end in zip(starts, starts[1:]):
            assert end - start == int(lengths[start, agent])
    switches = [(draws[1:, a * 3:(a + 1) * 3] != draws[:-1, a * 3:(a + 1) * 3]).any(dim=(1, 2)) for a in range(2)]
    assert not torch.equal(switches[0], switches[1])  # the agents hold independently


def test_without_training_or_the_option_no_noise_is_held():
    learner = build_learner("sac", num_agents=1, noise_repeat=True)
    learner.enable_training_mode(False)
    learner.act(torch.randn(2, OBS_DIM), None, timestep=0, timesteps=10)
    assert learner._held_noise is None
    plain = build_learner("sac", num_agents=1)
    plain.act(torch.randn(2, OBS_DIM), None, timestep=0, timesteps=10)
    assert plain._held_noise is None


def test_the_random_warm_up_is_not_held():
    learner = build_learner("sac", num_agents=1, noise_repeat=True, random_timesteps=5)
    learner.act(torch.randn(2, OBS_DIM), None, timestep=0, timesteps=10)
    assert learner._held_noise is None

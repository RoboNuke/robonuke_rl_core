"""The independence rule: nothing computed from agent j's data may change agent i's update.

The template for every learner: fill the memory, copy the learner, make agent 1's data
extreme in the copy, run one update on both, and require agents 0 and 2 to come out
bit-identical. Every per-agent mechanism (normalizers, gradient clipping, KL early stop,
advantage normalization, the SAC temperature, memory sampling) is exercised by this.
"""

from __future__ import annotations

import copy

import pytest
import torch


from helpers import build_learner, fill_memory

AGENT_UNDER_TEST = 1  # the agent whose data is made extreme
UNTOUCHED = (0, 2)


def snapshot(learner, agent: int) -> dict:
    """Everything that belongs to one agent: weights, optimizer moments, stats, extras."""
    snap = {}
    for key in learner._checkpoint_model_keys():
        snap[f"model/{key}"] = getattr(learner, key).agent_state_dict(agent)
    for key in learner._checkpoint_optimizer_keys():
        snap[f"optimizer/{key}"] = getattr(learner, key).agent_state_dict(agent)
    for name, norm in learner._checkpoint_normalizers().items():
        snap[f"normalizer/{name}"] = norm.state_dict_for(agent)
    snap["extras"] = learner._checkpoint_extras(agent)
    return snap


def assert_same(left, right, path: str = "") -> None:
    if torch.is_tensor(left):
        assert torch.is_tensor(right), path
        assert torch.equal(left, right), f"{path}: {left} != {right}"
    elif isinstance(left, dict):
        assert set(left) == set(right), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{i}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def make_extreme(learner, agent: int) -> None:
    """Scale one agent's stored data far out of range; its envs only."""
    low = agent * learner.envs_per_agent
    high = low + learner.envs_per_agent
    memory = learner.memory
    memory.get_tensor_by_name("rewards")[:, low:high] *= 1.0e6
    for name in ("observations", "next_observations"):
        if name in memory.tensors:
            memory.get_tensor_by_name(name)[:, low:high] *= 1.0e3
    if "values" in memory.tensors:
        memory.get_tensor_by_name("values")[:, low:high] *= 1.0e6


def run_updates(learner, count: int = 3, seed: int = 7) -> None:
    """A few updates from a fixed RNG state, so both learners draw the same batches.

    More than one: Adam's first step is sign-only (``m_hat/sqrt(v_hat)`` is +-1), so a single
    step cannot show that a gradient's magnitude changed.
    """
    torch.manual_seed(seed)
    for _ in range(count):
        learner.update(timestep=10, timesteps=100)


def _model_weights(snap: dict) -> dict:
    return {key: value for key, value in snap.items() if key.startswith("model/")}


def _kept(learner) -> dict:
    """Collect each agent's logged ``ppo/kept`` fractions."""
    kept: dict = {}
    learner.on_log.append(
        lambda agent, metrics, step: kept.setdefault(agent, []).append(float(metrics["ppo/kept"]))
        if "ppo/kept" in metrics
        else None
    )
    return kept


@pytest.mark.parametrize(
    "learner_name, overrides",
    [
        ("sac", {}),
        ("sac", {"grad_norm_clip": 0.5}),
        ("ppo", {}),
        ("ppo", {"grad_norm_clip": 0.5}),
        # KL cases run without observation normalization: then the untouched agents start at
        # KL 0 and keep updating, while agent 1 (whose stored log_prob came from its unscaled
        # observations) is dropped in the extreme copy only. The test asserts that this
        # happened, so the masking path is really exercised.
        ("ppo", {"grad_norm_clip": 0.5, "kl_threshold": 0.1, "normalize_observations": False}),
        ("ppo", {"kl_threshold": 0.1, "value_update_ratio": 3, "normalize_observations": False}),
        # a Bernoulli dim exercises the straight-through and Bernoulli log-prob paths
        ("ppo", {"model_overrides": {"bernoulli_action_dims": [1]}}),
        ("sac", {"model_overrides": {"bernoulli_action_dims": [1]}}),
        ("ppo", {"entropy_loss_scale": 0.01, "value_update_ratio": 3}),
    ],
)
def test_one_agents_data_cannot_change_another_agents_update(learner_name, overrides):
    control = build_learner(learner_name, num_agents=3, envs_per_agent=2, **overrides)
    fill_memory(control)
    before = {agent: snapshot(control, agent) for agent in UNTOUCHED}

    extreme = copy.deepcopy(control)
    make_extreme(extreme, AGENT_UNDER_TEST)

    masked = overrides.get("kl_threshold", 0) > 0
    if masked:
        control_kept, extreme_kept = _kept(control), _kept(extreme)

    run_updates(control)
    run_updates(extreme)

    for agent in UNTOUCHED:
        assert_same(snapshot(control, agent), snapshot(extreme, agent), f"agent{agent}")
        # and the networks really changed (not just normalizer stats), so the comparison
        # above is not vacuous
        with pytest.raises(AssertionError):
            assert_same(
                _model_weights(before[agent]),
                _model_weights(snapshot(control, agent)),
                f"agent{agent}",
            )

    if masked:
        # agent 1 was dropped in the extreme copy but not in the control, while the untouched
        # agents kept updating: the masked-loss and frozen-step paths really ran
        assert extreme_kept[AGENT_UNDER_TEST] != control_kept[AGENT_UNDER_TEST]
        assert min(extreme_kept[AGENT_UNDER_TEST]) < 1.0
        for agent in UNTOUCHED:
            assert max(extreme_kept[agent]) == 1.0

    # the extreme agent itself did diverge
    with pytest.raises(AssertionError):
        assert_same(
            snapshot(control, AGENT_UNDER_TEST),
            snapshot(extreme, AGENT_UNDER_TEST),
            "agent1",
        )


def test_ppo_kl_early_stop_drops_only_the_offending_agent():
    """An agent over the KL threshold stops; the others keep updating."""
    # normalization off so the stored log_prob matches the recomputed one exactly: then an
    # untouched agent's KL starts at 0 while agent 1's is huge (its stored log_prob came from
    # its pre-scaled observations). With normalization on, the stats move between the rollout
    # and the update and every agent starts with a large KL.
    learner = build_learner(
        "ppo", num_agents=3, envs_per_agent=2, kl_threshold=0.1, normalize_observations=False
    )
    fill_memory(learner)
    make_extreme(learner, AGENT_UNDER_TEST)

    logged: list[tuple[int, dict]] = []
    learner.on_log.append(lambda agent, metrics, step: logged.append((agent, dict(metrics))))
    run_updates(learner, count=1)

    kept = {agent: float(metrics["ppo/kept"]) for agent, metrics in logged if "ppo/kept" in metrics}
    assert kept[AGENT_UNDER_TEST] < 1.0  # dropped partway through the epoch
    for agent in UNTOUCHED:
        assert kept[agent] > kept[AGENT_UNDER_TEST]
    # the other agents' advantage scale is unaffected by agent 1's huge rewards
    advantages = learner.memory.get_tensor_by_name("advantages")
    for agent in UNTOUCHED:
        low = agent * learner.envs_per_agent
        block = advantages[:, low : low + learner.envs_per_agent]
        assert abs(float(block.mean())) < 1.0e-5
        assert abs(float(block.std()) - 1.0) < 0.2


def test_sac_temperature_is_per_agent():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2, learn_entropy=True)
    fill_memory(learner)
    extreme = copy.deepcopy(learner)
    make_extreme(extreme, AGENT_UNDER_TEST)

    run_updates(learner)
    run_updates(extreme)

    for agent in UNTOUCHED:
        assert torch.equal(
            learner._entropy_coefficient[agent], extreme._entropy_coefficient[agent]
        )
    assert not torch.equal(
        learner._entropy_coefficient[AGENT_UNDER_TEST],
        extreme._entropy_coefficient[AGENT_UNDER_TEST],
    )


def test_normalizer_stats_are_per_agent():
    learner = build_learner("sac", num_agents=3, envs_per_agent=2)
    fill_memory(learner)
    extreme = copy.deepcopy(learner)
    make_extreme(extreme, AGENT_UNDER_TEST)

    run_updates(learner)
    run_updates(extreme)

    norm_a = learner.observation_normalizer
    norm_b = extreme.observation_normalizer
    for agent in UNTOUCHED:
        assert torch.equal(norm_a.running_mean[agent], norm_b.running_mean[agent])
        assert torch.equal(norm_a.running_variance[agent], norm_b.running_variance[agent])
    assert not torch.equal(
        norm_a.running_mean[AGENT_UNDER_TEST], norm_b.running_mean[AGENT_UNDER_TEST]
    )

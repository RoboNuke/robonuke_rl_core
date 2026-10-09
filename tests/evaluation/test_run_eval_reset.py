"""run_eval resets a stateful policy at the start of every round. Pure torch, a fake env."""

from __future__ import annotations

import torch

from robonuke_rl_core.evaluation import run_eval

NUM_ENVS, LENGTH = 2, 3


class FakeEnv:
    """Every env times out after LENGTH steps; reset() really resets the clock."""

    num_envs = NUM_ENVS

    def __init__(self):
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.unwrapped = self

    def reset(self):
        self.episode_length_buf.zero_()
        return torch.zeros(NUM_ENVS, 1), {}

    def step(self, actions):
        self.episode_length_buf += 1
        done = (self.episode_length_buf >= LENGTH).unsqueeze(-1)
        return torch.zeros(NUM_ENVS, 1), torch.zeros(NUM_ENVS, 1), done, done, {}


class ChunkPolicy:
    """Records, per call, how many steps it has acted since its last reset."""

    def __init__(self):
        self.since_reset = 0
        self.resets = 0
        self.seen = []

    def reset(self):
        self.resets += 1
        self.since_reset = 0

    def __call__(self, observations, states):
        self.seen.append(self.since_reset)
        self.since_reset += 1
        return torch.zeros(NUM_ENVS, 1)


def test_a_stateful_policy_is_reset_before_every_round():
    policy = ChunkPolicy()
    run_eval(env=FakeEnv(), policy=policy, num_rollouts=3 * NUM_ENVS, max_episode_length=LENGTH,
             seed=0, save_state=False)
    assert policy.resets == 3
    assert policy.seen == [0, 1, 2] * 3  # each round starts from a fresh policy state


def test_a_plain_callable_still_works():
    calls = []
    result = run_eval(env=FakeEnv(), policy=lambda o, s: calls.append(1) or torch.zeros(NUM_ENVS, 1),
                      num_rollouts=NUM_ENVS, max_episode_length=LENGTH, seed=0, save_state=False)
    assert len(calls) == LENGTH and result.summary["episodes"] == NUM_ENVS


# ------------------------------------------------------------------ project eval policies
def test_a_registered_project_learner_builds_its_own_eval_policy():
    import pytest

    from robonuke_rl_core import evaluation

    class Cfg:
        class trainer:
            learner = "bc_test"

    calls = []
    evaluation.register_eval_policy("bc_test", lambda cfg, env, ckpt, det: calls.append((env, ckpt, det)) or ("p", None, {"step": 1}))
    try:
        assert evaluation.build_eval_policy(Cfg, "env", "ckpt", deterministic=False) == ("p", None, {"step": 1})
        assert calls == [("env", "ckpt", False)]
        with pytest.raises(ValueError, match="already registered"):
            evaluation.register_eval_policy("bc_test", lambda *a: None)
        with pytest.raises(ValueError, match="built in"):
            evaluation.register_eval_policy("sac", lambda *a: None)
    finally:
        evaluation.EVAL_POLICY_BUILDERS.pop("bc_test", None)

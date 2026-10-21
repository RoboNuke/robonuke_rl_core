"""run_eval's progress bar: the total is every episode run to its time limit, and an episode that
closes early adds its unused steps, so the bar ends exactly at the total. Pure torch, a fake env."""

from __future__ import annotations

import pytest
import torch
import tqdm

from robonuke_rl_core.evaluation import run_eval

NUM_ENVS, LENGTH = 3, 5
#: the step each env's episode ends at (it auto-resets after); None: never, the budget runs out
ENDS = (2, 3, None)


class FakeEnv:
    num_envs = NUM_ENVS

    def __init__(self):
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.ends = torch.tensor([e if e is not None else 10**9 for e in ENDS])
        self.unwrapped = self

    def reset(self):
        self.episode_length_buf.zero_()
        return torch.zeros(NUM_ENVS, 1), {}

    def step(self, actions):
        self.episode_length_buf += 1
        done = self.episode_length_buf >= self.ends
        self.episode_length_buf[done] = 0
        zeros = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
        return torch.zeros(NUM_ENVS, 1), torch.zeros(NUM_ENVS, 1), done.unsqueeze(-1), zeros, {}


class FakeBar:
    made = []

    def __init__(self, total, **kwargs):
        self.total, self.updates, self.closed = total, [], False
        FakeBar.made.append(self)

    def update(self, n):
        self.updates.append(n)

    def close(self):
        self.closed = True


@pytest.fixture
def bars(monkeypatch):
    FakeBar.made = []
    monkeypatch.setattr(tqdm, "tqdm", FakeBar)
    return FakeBar.made


def test_the_bar_counts_every_episode_to_its_time_limit(bars):
    # rounds of 3 and 1 envs: 4 episodes x 5 steps
    run_eval(env=FakeEnv(), policy=lambda o, s: torch.zeros(NUM_ENVS, 1), num_rollouts=4,
             max_episode_length=LENGTH, seed=0, save_state=False, progress=True)
    (bar,) = bars
    assert bar.total == 4 * LENGTH and bar.closed
    # round 1: 3 stepping; env 0 closes at step 2 (+3 unused), env 1 at 3 (+2), env 2 runs out;
    # round 2: env 0 alone, closes at step 2 (+3)
    assert bar.updates == [3, 3 + 3, 2 + 2, 1, 1, 1, 1 + 3]
    assert sum(bar.updates) == bar.total


def test_no_bar_unless_asked(bars):
    run_eval(env=FakeEnv(), policy=lambda o, s: torch.zeros(NUM_ENVS, 1), num_rollouts=3,
             max_episode_length=LENGTH, seed=0, save_state=False)
    assert bars == []

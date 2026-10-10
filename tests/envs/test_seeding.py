"""The run's seed reaches the simulator: skrl's first reset must not reseed everything with 0.

skrl's IsaacLabWrapper resets the env the first time with ``env.reset(seed=config.torch.key)``
(default 0), and Isaac Lab's reset then reseeds random / NumPy / torch / Warp / Replicator with
it. ``wrap_for_skrl`` sets the key to the run's seed first; ``reseed_env`` reseeds through the
env's own ``seed`` (Isaac Sim's set_seed), not torch alone.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from robonuke_rl_core.envs import build


def test_wrap_for_skrl_sets_skrl_s_reset_seed_before_wrapping(monkeypatch):
    import skrl
    import skrl.envs.wrappers.torch as wrappers

    seen = {}
    monkeypatch.setattr(wrappers, "wrap_env", lambda env, wrapper: seen.update(key=skrl.config.torch.key, wrapper=wrapper) or "wrapped")
    monkeypatch.setattr(skrl.config.torch, "key", 0)
    assert build.wrap_for_skrl(object(), 42) == "wrapped"
    assert seen == {"key": 42, "wrapper": "isaaclab"}


def test_skrl_s_isaac_lab_wrapper_still_takes_its_first_reset_seed_from_the_key():
    """If skrl changes where the first reset's seed comes from, wrap_for_skrl stops working."""
    from skrl.envs.wrappers.torch import isaaclab_envs

    source = inspect.getsource(isaaclab_envs.IsaacLabWrapper)
    assert "self._seed = config.torch.key" in source
    assert "self._env.reset(seed=self._seed)" in source


def test_reseed_env_uses_the_env_s_own_seed():
    calls = []

    class Unwrapped:
        def seed(self, value):
            calls.append(value)

    class Wrapped:
        unwrapped = Unwrapped()

    build.reseed_env(Wrapped(), 7)
    assert calls == [7]


def test_reseed_env_falls_back_to_torch():
    build.reseed_env(object(), 5)
    a = torch.rand(3)
    torch.manual_seed(5)
    assert torch.equal(a, torch.rand(3))

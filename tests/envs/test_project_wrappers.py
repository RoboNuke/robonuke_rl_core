"""build_env applies project wrappers outside the package's, in registration order."""

from __future__ import annotations

import pytest

from robonuke_rl_core.envs import build


class Cfg:
    controller = None
    wrappers = None


@pytest.fixture(autouse=True)
def clean_registry():
    saved = list(build.PROJECT_WRAPPERS)
    build.PROJECT_WRAPPERS.clear()
    yield
    build.PROJECT_WRAPPERS[:] = saved


def test_registered_wrappers_wrap_in_order_and_see_the_config():
    seen = []

    def first(cfg, env, task):
        seen.append(("first", cfg, task))
        return ("first", env)

    def second(cfg, env, task):
        seen.append(("second", cfg, task))
        return ("second", env)

    build.register_env_wrapper(first)
    build.register_env_wrapper(second)
    cfg = Cfg()
    assert build.build_env(cfg, "env", "Some-Task") == ("second", ("first", "env"))
    assert seen == [("first", cfg, "Some-Task"), ("second", cfg, "Some-Task")]


def test_bad_registrations_raise():
    with pytest.raises(TypeError):
        build.register_env_wrapper("not callable")
    fn = lambda cfg, env, task: env
    build.register_env_wrapper(fn)
    with pytest.raises(ValueError, match="already registered"):
        build.register_env_wrapper(fn)

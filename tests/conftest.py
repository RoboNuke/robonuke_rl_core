"""Shared test setup: the `gpu` marker, the one Isaac Sim startup, and the env factory.

Two things can only happen once per process:

* **The sim app.** Kit can only be launched once, so `isaac_sim` is session-scoped.
* **The env.** Isaac Lab hangs when a second env is created after one was closed (a closed
  env does not release its USD stage). So the whole GPU suite shares one env, built from
  `tests/gpu_forge.yaml` and never closed (the process exits in `pytest_unconfigure`). The
  factory **raises** on a second env rather than hanging: a GPU test that needs a different
  env means splitting the suite into one process per directory.

A GPU test that cannot start Isaac Sim fails; nothing is skipped silently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_EXIT_STATUS = 0
_STARTED = False


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs Isaac Sim and a GPU; run with `pytest -m gpu`"
    )


@pytest.fixture(scope="session")
def isaac_sim():
    """Start the sim app once for the whole session."""
    global _STARTED
    from isaaclab.app import AppLauncher

    # Omniverse Kit parses sys.argv itself and dies on pytest's own flags ("Ill formed
    # parameter: -m", then a segfault), so it must not see them.
    argv = sys.argv[:]
    sys.argv = argv[:1]
    try:
        launcher = AppLauncher(headless=True)
    finally:
        sys.argv = argv
    _STARTED = True
    # No app.close(): Kit's shutdown hangs under pytest. pytest_unconfigure ends the process
    # once the report is written.
    yield launcher.app


GPU_CONFIG = Path(__file__).resolve().parent / "gpu_forge.yaml"


@pytest.fixture(scope="session")
def gpu_env_factory(isaac_sim):
    """``make(cfg) -> wrapped env``, at most once per process; a second call raises."""
    built: list[str] = []

    def make(cfg):
        if built:
            raise RuntimeError(
                f"an env for '{built[0]}' was already created in this process, and Isaac Lab "
                "hangs when a second one is created. Share the session env (the gpu_env "
                "fixture), or split the GPU suite into one pytest process per directory."
            )
        built.append(cfg.task_name)
        import gymnasium as gym
        from skrl.envs.wrappers.torch import wrap_env

        return wrap_env(gym.make(cfg.task_name, cfg=cfg.task_cfg), wrapper="isaaclab")

    return make


@pytest.fixture(scope="session")
def gpu_cfg(isaac_sim, tmp_path_factory):
    """The resolved config every GPU test shares (Isaac Lab must be running first)."""
    from robonuke_rl_core.config import load_config

    output_dir = tmp_path_factory.mktemp("runs")
    # a task-cfg override on the CLI as well as a section one, so the GPU tests prove both
    # layers reach the live env (the file sets num_envs: 8)
    return load_config(
        GPU_CONFIG, [f"trainer.output_dir={output_dir}", "task.cfg.scene.num_envs=4"]
    )


@pytest.fixture(scope="session")
def gpu_env(gpu_cfg, gpu_env_factory):
    """One wrapped Isaac Lab env for the whole session. Never closed, on purpose."""
    return gpu_env_factory(gpu_cfg)


@pytest.fixture(autouse=True)
def _start_isaac_for_gpu_tests(request):
    """Every gpu-marked test gets the session's sim app; CPU tests never start it."""
    if request.node.get_closest_marker("gpu"):
        request.getfixturevalue("isaac_sim")


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Only when Isaac Sim ran: its shutdown would otherwise hang the session."""
    if not _STARTED:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)

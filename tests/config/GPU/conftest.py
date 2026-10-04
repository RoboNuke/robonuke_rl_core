"""Starts Isaac Sim once per session, before any Isaac Lab import.

If the sim app cannot start, this fixture raises and the GPU tests fail. They are never
skipped silently.
"""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def isaac_sim():
    from isaaclab.app import AppLauncher

    # Omniverse Kit parses sys.argv itself and dies on pytest's own flags ("Ill formed
    # parameter: -m", then a segfault), so it must not see them.
    argv = sys.argv[:]
    sys.argv = argv[:1]
    try:
        launcher = AppLauncher(headless=True)
    finally:
        sys.argv = argv

    # No app.close() here: Kit's shutdown hangs forever under pytest. pytest_sessionfinish
    # below ends the process once the report has been written.
    yield launcher.app


_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """End the process after the report is printed; Kit's own shutdown would hang."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)

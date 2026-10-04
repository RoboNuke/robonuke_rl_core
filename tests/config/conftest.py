"""These tests cover config.py itself, not the sections other areas register.

Every area that registers a section would otherwise make each config fixture incomplete
(`trainer.learner` and friends are required). So the config tests run with only the two
sections config.py owns; the learner and model tests cover theirs.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core import config as cfgmod


@pytest.fixture(autouse=True)
def only_config_sections(request):
    # the GPU test loads the shared config, which sets every registered section
    if request.node.get_closest_marker("gpu"):
        yield
        return
    known = dict(cfgmod.SECTIONS)
    cfgmod.SECTIONS.clear()
    cfgmod.SECTIONS.update({"experiment": cfgmod.ExperimentCfg, "wandb": cfgmod.WandbCfg})
    yield
    cfgmod.SECTIONS.clear()
    cfgmod.SECTIONS.update(known)

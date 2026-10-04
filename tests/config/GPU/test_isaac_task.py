"""Real Isaac Lab task: the config manager must resolve and round trip a Forge env cfg.

Run on the GPU machine: `pytest -m gpu`
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.config import dump, load_config, load_from_run

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent


def test_forge_task_resolves_and_round_trips(tmp_path):
    cfg = load_config(HERE / "forge_exp.yaml", ["task.cfg.scene.num_envs=16"])

    assert cfg.task_name == "Isaac-Forge-PegInsert-Direct-v0"
    assert cfg.derived["run_names"] == [f"fgain_k100_a{i}" for i in range(4)]
    # the CLI override and the file layers reached the real env cfg object
    assert cfg.task_cfg.scene.num_envs == 16
    assert cfg.task_cfg.decimation == 4  # Forge's own default is 8
    assert cfg.task_cfg.episode_length_s == 7.5  # Forge's own default is 10.0

    first = dump(cfg, tmp_path / "run_a")
    second = dump(load_from_run(tmp_path / "run_a"), tmp_path / "run_b")

    a = OmegaConf.to_container(OmegaConf.load(first))
    b = OmegaConf.to_container(OmegaConf.load(second))
    assert a.pop("meta") and b.pop("meta")
    assert a == b

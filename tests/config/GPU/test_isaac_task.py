"""Real Isaac Lab task: the config manager must resolve, seed and round trip a Forge env.

Run on the GPU machine: `pytest -m gpu`. The config and the env come from the session
fixtures in `tests/conftest.py` — one env per process, see that file for why.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.config import check_env_kept_overrides, dump, load_config, load_from_run

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent


def test_the_config_reached_the_real_env_cfg(gpu_cfg, gpu_env):
    cfg, live = gpu_cfg, gpu_env.unwrapped.cfg

    assert cfg.task_name == "Isaac-Forge-PegInsert-Direct-v0"
    assert cfg.derived["run_names"] == [
        f"{cfg.wandb.group}_a{i}" for i in range(cfg.experiment.num_agents)
    ]
    # the file layer reached the env cfg object (Forge's own defaults are 8 and 10.0)
    assert cfg.task_cfg.decimation == 4
    assert cfg.task_cfg.episode_length_s == 7.5
    # and a task-cfg CLI override beat the file (which sets 8) all the way to the live env
    assert cfg.task_cfg.scene.num_envs == 4
    assert live.scene.num_envs == 4
    assert gpu_env.num_envs == 4
    # one seed: experiment.seed is the env's seed, and the live env kept it
    assert cfg.task_cfg.seed == cfg.experiment.seed
    assert live.seed == cfg.experiment.seed


def test_a_discarded_override_raises(gpu_env):
    """Forge's __init__ recomputes observation_space, so an override of it must be caught."""
    discarded = load_config(HERE / "forge_discarded.yaml")
    with pytest.raises(ValueError) as err:
        check_env_kept_overrides(discarded, gpu_env.unwrapped.cfg)
    assert "task.cfg.observation_space" in str(err.value)


def test_round_trip_from_the_live_env(tmp_path, gpu_cfg, gpu_env):
    live = gpu_env.unwrapped.cfg
    first = dump(gpu_cfg, tmp_path / "run_a", live)
    second = dump(load_from_run(tmp_path / "run_a"), tmp_path / "run_b", live)

    a = OmegaConf.to_container(OmegaConf.load(first))
    b = OmegaConf.to_container(OmegaConf.load(second))
    assert a.pop("meta") and b.pop("meta")
    assert a == b

    # the file records what the env computed, not the pre-__init__ default
    # (spaces are written in Isaac Lab's serialized form)
    from isaaclab.envs.utils.spaces import serialize_space

    assert a["task"]["cfg"]["observation_space"] == serialize_space(live.observation_space)
    assert live.observation_space != 999

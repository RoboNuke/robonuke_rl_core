"""Real Isaac Lab task: the config manager must resolve, seed and round trip a Forge env.

Run on the GPU machine: `pytest -m gpu`
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.config import check_env_kept_overrides, dump, load_config, load_from_run

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent


def test_forge_task_resolves_seeds_and_round_trips(tmp_path):
    import gymnasium as gym

    cfg = load_config(HERE / "forge_exp.yaml", ["task.cfg.scene.num_envs=16", "experiment.seed=7"])

    assert cfg.task_name == "Isaac-Forge-PegInsert-Direct-v0"
    assert cfg.derived["run_names"] == [f"fgain_k100_a{i}" for i in range(4)]
    # the CLI override and the file layers reached the real env cfg object
    assert cfg.task_cfg.scene.num_envs == 16
    assert cfg.task_cfg.decimation == 4  # Forge's own default is 8
    assert cfg.task_cfg.episode_length_s == 7.5  # Forge's own default is 10.0
    assert cfg.task_cfg.seed == 7  # experiment.seed is the env seed

    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    try:
        live = env.unwrapped.cfg
        assert live.seed == 7

        # Forge's __init__ recomputes observation_space, so an override of it must be caught
        discarded = load_config(HERE / "forge_discarded.yaml")
        with pytest.raises(ValueError) as err:
            check_env_kept_overrides(discarded, live)
        assert "task.cfg.observation_space" in str(err.value)

        # dump from the live env cfg, reload, dump again: identical except meta
        first = dump(cfg, tmp_path / "run_a", live)
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
    finally:
        env.close()

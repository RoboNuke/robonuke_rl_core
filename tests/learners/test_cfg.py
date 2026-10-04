"""The learner, model and memory sections: defaults load, every validate rule bites.

These go through the real config pipeline with a faked task, so the cross-section rules
(`num_envs` divisible by `num_agents`, PPO's minibatch split, FlashSAC's reward scaling)
are checked the way a run would hit them.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import dump, load_config, load_from_run


@dataclass
class FakeScene:
    num_envs: int = 8
    env_spacing: float = 1.5


@dataclass
class FakeEnvCfg:
    decimation: int = 8
    episode_length_s: float = 5.0
    seed: Optional[int] = None
    scene: FakeScene = field(default_factory=FakeScene)


@pytest.fixture(autouse=True)
def fake_task(monkeypatch):
    monkeypatch.setattr(cfgmod, "load_task_cfg", lambda name: FakeEnvCfg())
    monkeypatch.setattr(cfgmod, "task_cfg_to_dict", lambda cfg: dataclasses.asdict(copy.deepcopy(cfg)))

    def apply(env_cfg, data):
        for key, value in data.items():
            current = getattr(env_cfg, key)
            if dataclasses.is_dataclass(current):
                apply(current, value)
            else:
                setattr(env_cfg, key, value)
        return env_cfg

    monkeypatch.setattr(cfgmod, "apply_task_cfg", apply)


BASE = """
task:
  name: Fake-Task-v0
experiment:
  seed: 1
  num_agents: 2
wandb:
  entity: hur
  project: p
  group: g
trainer:
  learner: sac
  total_timesteps: 100
"""


def write(tmp_path, extra: str = "", name: str = "exp.yaml"):
    path = tmp_path / name
    path.write_text(BASE + extra)
    return path


def test_every_section_loads_with_its_defaults(tmp_path):
    cfg = load_config(write(tmp_path))
    assert set(cfg.sections) == {
        "experiment", "wandb", "trainer", "sac", "ppo", "flash_sac", "model", "memory", "losses",
    }
    assert cfg.trainer.learner == "sac"
    assert cfg.sac.batch_size == 64  # a class default, not set by the file
    assert cfg.model.actor.actor_latent == 512
    assert cfg.memory.memory_size == 1_000_000
    assert cfg.losses.terms == []


def test_round_trip(tmp_path):
    cfg = load_config(write(tmp_path), ["sac.batch_size=8", "model.critic.critic_n=3"])
    first = dump(cfg, tmp_path / "run_a", cfg.task_cfg)
    second = dump(load_from_run(tmp_path / "run_a"), tmp_path / "run_b", cfg.task_cfg)
    a = OmegaConf.to_container(OmegaConf.load(first))
    b = OmegaConf.to_container(OmegaConf.load(second))
    a.pop("meta"), b.pop("meta")
    assert a == b
    assert a["sac"]["batch_size"] == 8
    assert a["model"]["critic"]["critic_n"] == 3
    # every learner's section is dumped, not only the one in use
    assert {"sac", "ppo", "flash_sac"} <= set(a)


# ------------------------------------------------------------------ 8. the env partition
def test_num_envs_must_divide_by_num_agents(tmp_path):
    with pytest.raises(ValueError) as err:
        load_config(write(tmp_path), ["experiment.num_agents=3", "task.cfg.scene.num_envs=8"])
    message = str(err.value)
    assert "num_envs" in message and "num_agents" in message

    cfg = load_config(write(tmp_path), ["experiment.num_agents=4", "task.cfg.scene.num_envs=8"])
    assert cfg.experiment.num_agents == 4


def test_unknown_learner_raises(tmp_path):
    with pytest.raises(ValueError) as err:
        load_config(write(tmp_path), ["trainer.learner=dqn"])
    assert "trainer.learner" in str(err.value)


@pytest.mark.parametrize(
    "override, needle",
    [
        ("trainer.total_timesteps=0", "total_timesteps"),
        ("trainer.write_interval=-1", "write_interval"),
        ("sac.lr_schedule=linear", "lr_schedule"),
        ("sac.entropy_loss_form=huber", "entropy_loss_form"),
        ("sac.batch_size=0", "batch_size"),
        ("sac.polyak=0.0", "polyak"),
        ("sac.periodic_reset_enabled=true", "periodic_reset_frequency"),
        ("memory.memory_size=0", "memory.memory_size"),
        ("model.actor.reduction=median", "reduction"),
        ("model.critic.n_atoms=1", "n_atoms"),
        ("model.critic.v_min=9.0", "v_min"),
        ("model.actor.min_log_std=5.0", "min_log_std"),
    ],
)
def test_each_validate_rule_fails_on_a_bad_value(tmp_path, override, needle):
    with pytest.raises(ValueError) as err:
        load_config(write(tmp_path), [override])
    assert needle in str(err.value)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        (["ppo.mini_batches=3"], "mini_batches"),
        (["ppo.value_update_ratio=0"], "value_update_ratio"),
        (["ppo.rollouts=0"], "rollouts"),
        (["ppo.lr_schedule=step"], "lr_schedule"),
    ],
)
def test_ppo_rules(tmp_path, overrides, needle):
    with pytest.raises(ValueError) as err:
        load_config(write(tmp_path), ["trainer.learner=ppo"] + overrides)
    assert needle in str(err.value)


def test_flash_sac_reward_scaling_must_match_the_critic_support(tmp_path):
    with pytest.raises(ValueError) as err:
        load_config(
            write(tmp_path),
            [
                "trainer.learner=flash_sac",
                "flash_sac.reward_scaling_enabled=true",
                "flash_sac.reward_scaling_g_max=3.0",
                "model.critic.v_max=5.0",
            ],
        )
    assert "reward_scaling_g_max" in str(err.value)

    cfg = load_config(
        write(tmp_path),
        [
            "trainer.learner=flash_sac",
            "flash_sac.reward_scaling_enabled=true",
            "flash_sac.reward_scaling_g_max=5.0",
            "model.critic.v_max=5.0",
        ],
    )
    assert cfg.flash_sac.reward_scaling_enabled is True


def test_flash_sac_target_entropy_mode(tmp_path):
    with pytest.raises(ValueError) as err:
        load_config(write(tmp_path), ["flash_sac.target_entropy_mode=sigma"])
    assert "target_entropy_mode" in str(err.value)


def test_a_callable_is_a_module_name_string(tmp_path):
    """No callable fields: a shaper is a "module:name" string, resolved where it is used."""
    cfg = load_config(write(tmp_path), ["sac.rewards_shaper=robonuke_rl_core.config:resolve_callable"])
    assert cfg.sac.rewards_shaper == "robonuke_rl_core.config:resolve_callable"
    assert cfgmod.resolve_callable(cfg.sac.rewards_shaper) is cfgmod.resolve_callable


def test_ppo_minibatches_must_divide_the_per_agent_rows(tmp_path):
    # 1 rollout x 2 envs per agent = 2 rows per agent: 4 minibatches would be empty
    with pytest.raises(ValueError) as err:
        load_config(
            write(tmp_path),
            ["trainer.learner=ppo", "ppo.rollouts=1", "ppo.mini_batches=4",
             "task.cfg.scene.num_envs=4", "experiment.num_agents=2"],
        )
    assert "envs_per_agent" in str(err.value)

    cfg = load_config(
        write(tmp_path),
        ["trainer.learner=ppo", "ppo.rollouts=4", "ppo.mini_batches=4",
         "task.cfg.scene.num_envs=4", "experiment.num_agents=2"],
    )
    assert cfg.ppo.mini_batches == 4

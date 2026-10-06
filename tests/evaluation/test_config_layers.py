"""The config layers an eval or a debug run adds on top of the trained run's own.

The layering itself is `config.py`'s (`load_from_run`); what is pinned here is the one thing
eval and debug inject, and why. **Eval runs one agent whatever the run trained**: it loads
one checkpoint slot into a one-agent shell and gives it every env. So
``experiment.num_agents`` from the trained run is not merely unused — left in place it makes
the divisibility rule reject a perfectly good eval env count, and it makes the eval's own
``resolved_config.yaml`` claim an agent count that never ran.

No Isaac Lab: the three functions that touch it are faked, as in `tests/config`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import RESOLVED_NAME, load_from_run
from robonuke_rl_core.evaluation import SINGLE_AGENT_OVERRIDE

TASK = "Fake-Task-v0"
TRAINED_AGENTS = 3
#: not divisible by TRAINED_AGENTS: that is the whole point
EVAL_ENVS = 64


@dataclass
class FakeScene:
    num_envs: int = 4


@dataclass
class FakeEnvCfg:
    seed: Optional[int] = None
    observation_space: Any = 21
    scene: FakeScene = field(default_factory=FakeScene)


@pytest.fixture(autouse=True)
def fake_task(monkeypatch):
    def load(name):
        if name != TASK:
            raise ValueError(f"unknown fake task {name!r}")
        return FakeEnvCfg()

    def to_dict(env_cfg):
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg.observation_space = json.dumps({"space": "Box", "value": env_cfg.observation_space})
        return dataclasses.asdict(env_cfg)

    def apply(env_cfg, data):
        """Walk the dict onto the object, as Isaac Lab's update_class_from_dict does."""
        for key, value in data.items():
            if isinstance(value, dict):
                apply(getattr(env_cfg, key), value)
            else:
                setattr(env_cfg, key, value)
        return env_cfg

    monkeypatch.setattr(cfgmod, "load_task_cfg", load)
    monkeypatch.setattr(cfgmod, "task_cfg_to_dict", to_dict)
    monkeypatch.setattr(cfgmod, "apply_task_cfg", apply)


def trained_run(tmp_path: Path, num_agents: int = TRAINED_AGENTS) -> Path:
    """A run directory holding what training wrote: several agents, their own env count."""
    run_dir = tmp_path / "runs" / "proj" / "group"
    run_dir.mkdir(parents=True)
    (run_dir / RESOLVED_NAME).write_text(
        "meta:\n"
        "  pkg_commit: abc123\n"
        "derived:\n"
        "  run_names: [group_a0, group_a1, group_a2]\n"
        "experiment:\n"
        f"  num_agents: {num_agents}\n"
        "  seed: 42\n"
        "wandb:\n"
        "  entity: hur\n"
        "  project: proj\n"
        "  group: group\n"
        "trainer:\n"
        "  learner: sac\n"
        "  total_timesteps: 10000\n"
        "task:\n"
        f"  name: {TASK}\n"
        "  cfg:\n"
        "    scene:\n"
        f"      num_envs: {num_agents * 8}\n"
    )
    return run_dir


def eval_config(tmp_path: Path, num_envs: int = EVAL_ENVS) -> Path:
    """What an eval config sets: the test conditions and its own env count."""
    path = tmp_path / "e2e.yaml"
    path.write_text(f"task:\n  cfg:\n    scene:\n      num_envs: {num_envs}\n")
    return path


# ------------------------------------------------------------------ the bug this fixes
def test_an_eval_env_count_need_not_divide_by_the_trained_agent_count(tmp_path):
    """64 envs over a run that trained 3 agents: fine, because eval runs one."""
    cfg = load_from_run(
        trained_run(tmp_path),
        [SINGLE_AGENT_OVERRIDE],
        extra_files=[eval_config(tmp_path)],
    )
    assert cfg.experiment.num_agents == 1
    assert cfg.task_cfg.scene.num_envs == EVAL_ENVS
    # and the eval's own record says one agent, which is what will actually run
    assert cfg.derived["run_names"] == ["group_a0"]


def test_without_the_override_the_divisibility_rule_rejects_it(tmp_path):
    """The failure the injection exists to prevent, kept as the reason it is there."""
    with pytest.raises(ValueError) as err:
        load_from_run(trained_run(tmp_path), [], extra_files=[eval_config(tmp_path)])
    message = str(err.value)
    assert "num_envs (64)" in message and f"num_agents ({TRAINED_AGENTS})" in message


def test_the_injection_wins_over_an_explicit_cli_agent_count(tmp_path):
    """Eval cannot run several agents, so asking for them is overridden, not obeyed."""
    cfg = load_from_run(
        trained_run(tmp_path),
        ["experiment.num_agents=3", SINGLE_AGENT_OVERRIDE],
        extra_files=[eval_config(tmp_path)],
    )
    assert cfg.experiment.num_agents == 1


def test_everything_else_from_the_trained_run_still_carries_over(tmp_path):
    """The override is surgical: only the agent count changes."""
    cfg = load_from_run(
        trained_run(tmp_path),
        [SINGLE_AGENT_OVERRIDE],
        extra_files=[eval_config(tmp_path)],
    )
    assert cfg.experiment.seed == 42
    assert cfg.trainer.learner == "sac"
    assert cfg.task_name == TASK
    assert cfg.task_cfg.seed == 42  # written from experiment.seed, as always


@pytest.mark.parametrize("num_agents", [1, 2, 3, 5])
def test_it_holds_whatever_the_run_trained(tmp_path, num_agents):
    cfg = load_from_run(
        trained_run(tmp_path, num_agents),
        [SINGLE_AGENT_OVERRIDE],
        extra_files=[eval_config(tmp_path)],
    )
    assert cfg.experiment.num_agents == 1


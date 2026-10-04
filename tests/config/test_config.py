"""CPU tests for robonuke_rl_core/config.py. No Isaac Lab: the task functions are faked."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pytest
from omegaconf import MISSING, OmegaConf

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import (
    add_config_args,
    dump,
    load_config,
    load_from_run,
    register_section,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------ the fake task
@dataclass
class FakeScene:
    num_envs: int = 4
    env_spacing: float = 1.5


@dataclass
class FakeEnvCfg:
    """Stands in for an Isaac Lab env cfg object."""

    decimation: int = 8
    episode_length_s: float = 5.0
    device: str = "cuda:0"
    joint_ids: str = "slice(None,None,None)"  # Isaac Lab stores slices as strings
    scene: FakeScene = field(default_factory=FakeScene)


def _fake_apply(env_cfg: Any, data: dict) -> Any:
    for key, value in data.items():
        current = getattr(env_cfg, key)
        if dataclasses.is_dataclass(current):
            _fake_apply(current, value)
        else:
            setattr(env_cfg, key, value)
    return env_cfg


@pytest.fixture(autouse=True)
def fake_task(monkeypatch):
    """Replace the three Isaac Lab functions, and undo any section a test registers."""

    def load(name):
        if name != "Fake-Task-v0":
            raise ValueError(f"unknown fake task {name!r}")
        return FakeEnvCfg()

    monkeypatch.setattr(cfgmod, "load_task_cfg", load)
    monkeypatch.setattr(cfgmod, "task_cfg_to_dict", dataclasses.asdict)
    monkeypatch.setattr(cfgmod, "apply_task_cfg", _fake_apply)

    known = dict(cfgmod.SECTIONS)
    yield
    cfgmod.SECTIONS.clear()
    cfgmod.SECTIONS.update(known)


# ------------------------------------------------------------------ 1. load order
def test_load_order_across_chain_and_cli():
    cfg = load_config(FIXTURES / "nested/deep/leaf.yaml")
    # class / task defaults, untouched by any file
    assert cfg.task_cfg.episode_length_s == 5.0
    assert cfg.task_cfg.device == "cuda:0"
    # most-base file
    assert cfg.task_cfg.decimation == 4
    assert cfg.task_cfg.scene.env_spacing == 2.0
    assert cfg.experiment.num_agents == 2
    assert cfg.wandb.entity == "hur"
    # middle file beats the most-base file
    assert cfg.experiment.seed == 200
    assert cfg.wandb.project == "mid_project"
    assert cfg.wandb.tags == ["mid", "baseline"]
    # the passed file beats both
    assert cfg.wandb.group == "leaf_group"
    assert cfg.task_cfg.scene.num_envs == 256

    # the CLI beats every file
    cfg = load_config(
        FIXTURES / "nested/deep/leaf.yaml",
        ["task.cfg.scene.num_envs=128", "experiment.seed=3", "wandb.tags=[fgain,debug]"],
    )
    assert cfg.task_cfg.scene.num_envs == 128
    assert cfg.experiment.seed == 3
    assert cfg.wandb.tags == ["fgain", "debug"]
    assert cfg.wandb.project == "mid_project"


# ------------------------------------------------------------------ 2. chain errors
def test_missing_base_file_raises_and_names_the_file_that_asked_for_it():
    with pytest.raises(FileNotFoundError) as err:
        load_config(FIXTURES / "missing_base.yaml")
    message = str(err.value)
    assert "does_not_exist.yaml" in message
    assert "missing_base.yaml" in message


def test_missing_passed_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(FIXTURES / "nope.yaml")


def test_cycle_raises_and_prints_the_chain():
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "cycle_a.yaml")
    message = str(err.value)
    assert "cycle" in message
    assert "cycle_a.yaml" in message and "cycle_b.yaml" in message


# ------------------------------------------------------------------ 3. unknown keys
@pytest.mark.parametrize(
    "fixture, needle",
    [
        ("unknown_section.yaml", "controller"),
        ("unknown_field.yaml", "num_agent"),
        ("unknown_task_key.yaml", "num_env"),
    ],
)
def test_unknown_key_from_a_file_raises_and_names_the_layer(fixture, needle):
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / fixture)
    message = str(err.value)
    assert needle in message
    assert fixture in message  # the layer that caused it


@pytest.mark.parametrize(
    "override, needle",
    [
        ("controller.stiffness=100.0", "controller"),  # unknown top-level section
        ("experiment.num_agent=4", "num_agent"),  # unknown field in a section
        ("task.cfg.scene.num_env=128", "num_env"),  # unknown key in the task cfg
    ],
)
def test_unknown_key_from_the_cli_raises_and_names_the_layer(override, needle):
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "minimal.yaml", [override])
    message = str(err.value)
    assert "from CLI" in message
    assert needle in message


# ------------------------------------------------------------------ 4. required fields
def test_missing_required_fields_are_all_listed():
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "missing_required.yaml")
    message = str(err.value)
    for path in ("experiment.seed", "wandb.entity", "wandb.project", "wandb.group"):
        assert path in message


def test_no_layer_sets_task_name(tmp_path):
    path = tmp_path / "no_task.yaml"
    path.write_text("experiment:\n  seed: 1\n")
    with pytest.raises(ValueError) as err:
        load_config(path)
    assert "task.name" in str(err.value)


# ------------------------------------------------------------------ 5. section types
def test_bool_for_int_raises():
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "bool_for_int.yaml")
    message = str(err.value)
    assert "num_agents" in message
    assert "bool_for_int.yaml" in message


def test_scientific_notation_loads_as_a_float():
    cfg = load_config(FIXTURES / "float_sci.yaml")
    assert cfg.task_cfg.episode_length_s == pytest.approx(1e-4)


# ------------------------------------------------------------------ 6. task types
def test_wrong_task_type_raises_with_the_dotted_path():
    with pytest.raises(TypeError) as err:
        load_config(FIXTURES / "bad_task_type.yaml")
    message = str(err.value)
    assert "task.cfg.decimation" in message
    assert "int" in message


def test_int_for_float_is_allowed_in_the_task_cfg():
    cfg = load_config(FIXTURES / "int_for_float.yaml")
    assert cfg.task_cfg.episode_length_s == 3


def test_bool_for_int_in_the_task_cfg_raises():
    with pytest.raises(TypeError) as err:
        load_config(FIXTURES / "minimal.yaml", ["task.cfg.decimation=true"])
    assert "task.cfg.decimation" in str(err.value)


# ------------------------------------------------------------------ 7. malformed CLI
@pytest.mark.parametrize("bad", ["--headless", "experiment.seed", "-x", "=3", "a..b=1"])
def test_malformed_override_raises(bad):
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "minimal.yaml", [bad])
    assert "override" in str(err.value)


def test_add_config_args_requires_exactly_one_source():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--headless", action="store_true")  # stands in for AppLauncher's args
    args, unknown = parser.parse_known_args(
        ["--config", "a.yaml", "--headless", "experiment.seed=3"]
    )
    assert args.config == "a.yaml" and args.headless is True
    assert unknown == ["experiment.seed=3"]
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--headless"])
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--config", "a.yaml", "--from_run", "runs/x"])


# ------------------------------------------------------------------ 8. derived
def test_derived_run_names():
    assert load_config(FIXTURES / "minimal.yaml").derived["run_names"] == ["fgain_k100_a0"]
    assert load_config(FIXTURES / "four_agents.yaml").derived["run_names"] == [
        f"fgain_k100_a{i}" for i in range(4)
    ]


# ------------------------------------------------------------------ 9. round trip
def test_round_trip(tmp_path):
    first = load_config(FIXTURES / "task_override.yaml", ["experiment.num_agents=3"])
    first_path = dump(first, tmp_path / "run_a")

    second = load_from_run(tmp_path / "run_a")
    second_path = dump(second, tmp_path / "run_b")

    a = OmegaConf.to_container(OmegaConf.load(first_path))
    b = OmegaConf.to_container(OmegaConf.load(second_path))
    assert a.pop("meta") and b.pop("meta")
    assert a == b

    # the reload really rebuilt the objects
    assert second.task_cfg.decimation == 16
    assert second.task_cfg.scene.num_envs == 32
    assert second.derived["run_names"] == a["derived"]["run_names"]


def test_dump_writes_every_value_including_defaults(tmp_path):
    cfg = load_config(FIXTURES / "minimal.yaml")
    data = OmegaConf.to_container(OmegaConf.load(dump(cfg, tmp_path / "run")))
    assert list(data) == ["meta", "derived", "task", "experiment", "wandb"]
    assert data["task"]["cfg"]["device"] == "cuda:0"  # a default no layer set
    assert data["task"]["cfg"]["scene"]["env_spacing"] == 1.5
    assert data["wandb"]["tags"] == []
    assert set(data["meta"]) == {"pkg_commit", "project_commit", "created"}
    assert len(data["meta"]["pkg_commit"]) == 40


def test_from_run_takes_overrides_on_top(tmp_path):
    dump(load_config(FIXTURES / "minimal.yaml"), tmp_path / "run")
    cfg = load_from_run(tmp_path / "run", ["experiment.num_agents=2", "task.cfg.scene.num_envs=512"])
    assert cfg.task_cfg.scene.num_envs == 512
    assert cfg.derived["run_names"] == ["fgain_k100_a0", "fgain_k100_a1"]


# ------------------------------------------------------------------ 10. validate
@dataclass
class ControllerCfg:
    kind: str = "hybrid"
    max_agents: int = MISSING
    gains: List[float] = field(default_factory=lambda: [100.0, 10.0])

    def validate(self, cfg: Any) -> None:
        if cfg.experiment.num_agents > self.max_agents:
            raise ValueError(
                f"controller.max_agents ({self.max_agents}) is below experiment.num_agents "
                f"({cfg.experiment.num_agents})"
            )


def test_validate_error_surfaces():
    register_section("controller", ControllerCfg)
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "minimal.yaml", ["controller.max_agents=1", "experiment.num_agents=4"])
    message = str(err.value)
    assert "controller.max_agents (1)" in message
    assert "num_agents (4)" in message


def test_builtin_validate_rules():
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "minimal.yaml", ["experiment.num_agents=0"])
    assert "num_agents" in str(err.value)
    with pytest.raises(ValueError) as err:
        load_config(FIXTURES / "minimal.yaml", ["wandb.group='bad group'"])
    assert "whitespace" in str(err.value)


# ------------------------------------------------------------------ 11. register_section
def test_registered_section_loads():
    register_section("controller", ControllerCfg)
    cfg = load_config(FIXTURES / "minimal.yaml", ["controller.max_agents=4", "controller.kind=vic"])
    assert cfg.controller.kind == "vic"
    assert cfg.controller.gains == [100.0, 10.0]
    assert cfg["controller"] is cfg.controller


def test_duplicate_section_raises():
    with pytest.raises(ValueError) as err:
        register_section("experiment", ControllerCfg)
    assert "already registered" in str(err.value)


@pytest.mark.parametrize("name", ["base", "task", "meta", "derived"])
def test_reserved_section_names_raise(name):
    with pytest.raises(ValueError) as err:
        register_section(name, ControllerCfg)
    assert "reserved" in str(err.value)


def test_section_must_be_a_dataclass():
    class Plain:
        pass

    with pytest.raises(TypeError):
        register_section("plain", Plain)


# ------------------------------------------------------------------ extras
def test_resolve_callable():
    assert cfgmod.resolve_callable("pathlib:Path") is Path
    with pytest.raises(ValueError):
        cfgmod.resolve_callable("pathlib.Path")
    with pytest.raises(ValueError):
        cfgmod.resolve_callable("pathlib:NoSuchThing")
    with pytest.raises(ValueError):
        cfgmod.resolve_callable("no_such_module:thing")


def test_config_py_imports_without_isaac_lab():
    import sys

    assert not [m for m in sys.modules if m.split(".")[0] in ("isaaclab", "isaaclab_tasks", "omni")]

"""model.architecture picks the dataclasses behind model.actor / model.critic.

Same pattern as task.name: the last layer that sets it wins, the selected class builds the
structured node, and struct mode then rejects another architecture's fields. These tests run
with the real model section added back (the conftest fixture strips it) and config.py's
fake task in place.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core import config as cfgmod
from robonuke_rl_core.config import load_config
from robonuke_rl_core.models.cfg import (
    MODEL_ARCHITECTURES,
    SimbaModelCfg,
    register_architecture,
)

# the fake task and the section-stripping fixture
from test_config import FIXTURES, fake_task  # noqa: F401

MINIMAL = FIXTURES / "minimal.yaml"


@pytest.fixture(autouse=True)
def with_model_section():
    cfgmod.SECTIONS["model"] = SimbaModelCfg
    yield  # conftest's only_config_sections fixture restores the registry afterwards


# module level: OmegaConf resolves the (PEP 563) string annotations against module globals
import dataclasses


@dataclasses.dataclass
class Fake2Actor:
    width: int = 4


@dataclasses.dataclass
class Fake2Critic:
    depth: int = 1


@dataclasses.dataclass
class Fake2ModelCfg:
    architecture: str = "fake2"
    actor: Fake2Actor = dataclasses.field(default_factory=Fake2Actor)
    critic: Fake2Critic = dataclasses.field(default_factory=Fake2Critic)


@pytest.fixture()
def fake2_architecture():
    """A second architecture with its own actor/critic field names."""
    register_architecture("fake2", Fake2ModelCfg)
    yield Fake2ModelCfg
    del MODEL_ARCHITECTURES["fake2"]


def test_the_default_architecture_is_simba():
    cfg = load_config(MINIMAL)
    assert isinstance(cfg.model, SimbaModelCfg)
    assert cfg.model.architecture == "simba"
    assert cfg.model.actor.actor_n == 2  # SimBa fields are there


def test_an_unknown_architecture_raises_naming_the_layer():
    with pytest.raises(ValueError) as err:
        load_config(MINIMAL, ["model.architecture=resnet"])
    message = str(err.value)
    assert "resnet" in message and "simba" in message and "CLI" in message


def test_selecting_an_architecture_swaps_the_schema(fake2_architecture):
    cfg = load_config(MINIMAL, ["model.architecture=fake2", "model.actor.width=7"])
    assert isinstance(cfg.model, fake2_architecture)
    assert cfg.model.actor.width == 7

    # the other architecture's fields are rejected, not silently accepted
    with pytest.raises(ValueError) as err:
        load_config(MINIMAL, ["model.architecture=fake2", "model.actor.actor_n=1"])
    assert "actor_n" in str(err.value)

    with pytest.raises(ValueError) as err:
        load_config(MINIMAL, ["model.actor.width=7"])  # simba is the default
    assert "width" in str(err.value)


def test_the_last_layer_that_sets_the_architecture_wins(fake2_architecture, tmp_path):
    base = tmp_path / "exp.yaml"
    base.write_text(
        f"base: {MINIMAL}\nmodel:\n  architecture: fake2\n  actor:\n    width: 9\n"
    )
    cfg = load_config(base)
    assert isinstance(cfg.model, fake2_architecture)

    # the CLI switches back to simba; fake2's fields then fail the merge loudly
    with pytest.raises(ValueError) as err:
        load_config(base, ["model.architecture=simba"])
    assert "width" in str(err.value)


def test_registering_a_bad_architecture_raises():
    import dataclasses

    @dataclasses.dataclass
    class WrongDefault:
        architecture: str = "other-name"

    with pytest.raises(ValueError) as err:
        register_architecture("bad", WrongDefault)
    assert "architecture" in str(err.value)

    with pytest.raises(ValueError):
        register_architecture("simba", SimbaModelCfg)  # duplicate

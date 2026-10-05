"""The Forge boundary: the task-name rule that gates every wrapper under envs/forge/."""

from __future__ import annotations

import pytest

from robonuke_rl_core.envs.forge.compat import FORGE_MARKER, is_forge_task, require_forge_env


@pytest.mark.parametrize(
    "task",
    [
        "Isaac-Forge-PegInsert-Direct-v0",
        "isaac-forge-gearmesh-direct-v0",
        "My-FORGE-Variant",
        "project_forge_task",
    ],
)
def test_a_forge_task_passes(task):
    assert is_forge_task(task)
    require_forge_env(task, "ForgeControllerWrapper", ("cfg.ctrl",))


@pytest.mark.parametrize(
    "task",
    ["Isaac-Factory-PegInsert-Direct-v0", "Isaac-Lift-Cube-Franka-v0", "Isaac-Cartpole-v0", ""],
)
def test_any_other_task_is_refused_with_the_rule_and_the_contract(task):
    assert not is_forge_task(task)
    with pytest.raises(ValueError) as err:
        require_forge_env(task, "ForgeControllerWrapper", ("cfg.ctrl", "force_sensor_smooth"))
    message = str(err.value)
    assert "ForgeControllerWrapper" in message
    assert repr(task) in message
    assert FORGE_MARKER in message  # the rule itself
    assert "force_sensor_smooth" in message  # the attributes it would have needed
    assert "own package" in message  # where another family's wrapper belongs


def test_factory_is_not_forge_even_though_it_is_the_parent_class():
    """The rule is the name, not the class tree: Factory lacks the force sensor."""
    assert not is_forge_task("Isaac-Factory-PegInsert-Direct-v0")

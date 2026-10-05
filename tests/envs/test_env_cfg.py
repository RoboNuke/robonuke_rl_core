"""The `controller` and `wrappers` config sections: defaults, rules, and the Bernoulli check.

The cross-check between the controller's selection block and the actor's Bernoulli dims is
the one that matters most: get it wrong and a continuous action silently becomes a 0/1
switch at runtime, with no crash and no obvious symptom. It fails at config load instead.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from omegaconf import OmegaConf

from robonuke_rl_core.envs.cfg import ControllerCfg, WrappersCfg


class FakeActor:
    def __init__(self, bernoulli=None, force_zero=None):
        self.bernoulli_action_dims = bernoulli
        self.force_zero_action_dims = force_zero


class FakeCfg:
    """Just enough of a resolved config for `validate`: the model's actor."""

    def __init__(self, actor=None):
        self.model = type("Model", (), {"actor": actor})()


def controller(**overrides) -> ControllerCfg:
    fields = dict(enabled=True)
    fields.update(overrides)
    return ControllerCfg(**fields)


# ------------------------------------------------------------------ defaults and round trip
def test_the_defaults_are_inert_and_valid():
    cfg = ControllerCfg()
    assert cfg.enabled is False and cfg.use_pose is True and cfg.use_force is False
    assert cfg.gain_mapping == "constant" and cfg.native_action_dim == 7
    cfg.validate(FakeCfg(FakeActor()))  # a disabled section checks nothing

    wrappers = WrappersCfg()
    assert not any(
        group.enabled
        for group in (wrappers.fragile, wrappers.efficient_reset, wrappers.contact)
    )
    assert wrappers.orientation.mode == "quat"
    wrappers.validate(None)


def test_both_sections_round_trip_through_omegaconf():
    for section in (ControllerCfg(enabled=True, use_force=True), WrappersCfg()):
        node = OmegaConf.structured(section)
        back = OmegaConf.to_object(OmegaConf.create(OmegaConf.to_yaml(node)))
        assert back == dataclasses.asdict(section)


# ------------------------------------------------------------------ controller rules
def test_enabled_with_no_branch_raises():
    with pytest.raises(ValueError) as err:
        controller(use_pose=False, use_force=False).validate(None)
    assert "controller.use_pose" in str(err.value) and "controller.use_force" in str(err.value)


def test_a_disabled_controller_never_complains():
    ControllerCfg(enabled=False, use_pose=False, use_force=False, gain_mapping="nonsense").validate(None)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"gain_mapping": "rotated"}, "gain_mapping"),
        ({"gain_min": [1.0] * 5}, "length 6"),
        ({"gain_min": [1e4] * 6}, "gain_min <= gain_max"),
        ({"damping_ratio": 0.0}, "damping_ratio"),
        ({"native_action_dim": 0}, "native_action_dim"),
        ({"use_force": True, "force_gain_min": [1e4] * 6}, "force_gain_min <= force_gain_max"),
        ({"use_force": True, "force_target_bounds": [0.0] * 6}, "force_target_bounds"),
    ],
)
def test_each_range_rule_fails_on_a_bad_value(overrides, needle):
    with pytest.raises(ValueError) as err:
        controller(**overrides).validate(None)
    assert needle in str(err.value)


def test_force_settings_without_the_force_branch_raise():
    """Silently ignoring them is how a force experiment runs as a pose experiment."""
    with pytest.raises(ValueError) as err:
        controller(use_force=False, force_gain_max=[500.0] * 6).validate(None)
    assert "force_gain_max" in str(err.value) and "use_force" in str(err.value)

    # the same values with the branch on are fine
    controller(use_force=True, force_gain_max=[500.0] * 6).validate(
        FakeCfg(FakeActor([7, 8, 9, 10, 11, 12]))
    )


# ------------------------------------------------------------------ the Bernoulli cross-check
def test_both_branches_need_the_selection_block_as_the_bernoulli_dims():
    cfg = FakeCfg(FakeActor(bernoulli=[7, 8, 9, 10, 11, 12]))
    controller(use_force=True).validate(cfg)  # the selection block is [6..11]


@pytest.mark.parametrize(
    "declared",
    [None, [], [7, 8, 9, 10, 11], [7, 8, 9, 10, 11, 12, 13], [8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5]],
)
def test_missing_extra_or_shifted_bernoulli_dims_raise(declared):
    with pytest.raises(ValueError) as err:
        controller(use_force=True).validate(FakeCfg(FakeActor(bernoulli=declared)))
    message = str(err.value)
    assert "bernoulli_action_dims" in message
    assert "[7, 8, 9, 10, 11, 12]" in message  # it names the indices it expects


def test_a_single_branch_must_declare_no_bernoulli_dims():
    controller().validate(FakeCfg(FakeActor(bernoulli=None)))
    with pytest.raises(ValueError) as err:
        controller().validate(FakeCfg(FakeActor(bernoulli=[0])))
    assert "no selection block" in str(err.value)


def test_force_zero_dims_may_not_collide_with_controller_blocks():
    with pytest.raises(ValueError) as err:
        controller(use_force=True).validate(
            FakeCfg(FakeActor(bernoulli=[7, 8, 9, 10, 11, 12], force_zero=[3, 4]))
        )
    assert "force_zero_action_dims" in str(err.value) and "[3, 4]" in str(err.value)


def test_the_check_is_skipped_when_there_is_no_model_section():
    controller(use_force=True).validate(None)
    controller(use_force=True).validate(FakeCfg(actor=None))


# ------------------------------------------------------------------ wrappers rules
def test_fragile_break_force_shape_follows_the_mode():
    wrappers = WrappersCfg()
    wrappers.fragile.enabled = True
    wrappers.fragile.break_force = [50.0]
    wrappers.validate(None)

    wrappers.fragile.direction_break_force = True
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "[shear, normal]" in str(err.value)

    wrappers.fragile.break_force = [30.0, 60.0]
    wrappers.validate(None)
    wrappers.fragile.break_force = [0.0, 60.0]
    with pytest.raises(ValueError):
        wrappers.validate(None)


def test_loss_of_contact_needs_the_contact_sensor():
    wrappers = WrappersCfg()
    wrappers.fragile.enabled = True
    wrappers.fragile.require_contact = True
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "wrappers.contact.enabled" in str(err.value)

    wrappers.contact.enabled = True
    wrappers.validate(None)


def test_appending_contact_to_the_obs_needs_the_sensor():
    wrappers = WrappersCfg()
    wrappers.contact.append_to_policy_obs = True
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "wrappers.contact.enabled" in str(err.value)


def test_a_bad_orientation_mode_raises():
    wrappers = WrappersCfg()
    wrappers.orientation.mode = "euler"
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "quat" in str(err.value) and "6d_rot_mat" in str(err.value)


def test_a_bad_contact_threshold_raises():
    wrappers = WrappersCfg()
    wrappers.contact.enabled = True
    wrappers.contact.force_threshold = 0.0
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "force_threshold" in str(err.value)


def test_the_debounce_floor_is_one_step():
    wrappers = WrappersCfg()
    wrappers.fragile.enabled = True
    wrappers.fragile.require_contact_debounce_steps = 0
    with pytest.raises(ValueError) as err:
        wrappers.validate(None)
    assert "debounce" in str(err.value)


# ------------------------------------------------------------------ the hybrid axis mask
def test_the_mask_sets_how_many_axes_are_hybrid():
    cfg = controller(use_force=True, force_axes=[1, 1, 1, 0, 0, 0])
    cfg.validate(FakeCfg(FakeActor(bernoulli=[7, 8, 9])))  # 3-D hybrid: 3 selection dims

    with pytest.raises(ValueError) as err:
        cfg.validate(FakeCfg(FakeActor(bernoulli=[7, 8, 9, 10, 11, 12])))
    assert "[7, 8, 9]" in str(err.value)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"use_force": True, "force_axes": [0] * 6}, "selects no axis"),
        ({"use_force": True, "force_axes": [1, 1, 1]}, "length 6"),
        ({"use_force": True, "force_axes": [2, 0, 0, 0, 0, 0]}, "binary"),
        (
            {"use_pose": False, "use_force": True, "force_axes": [1, 1, 1, 0, 0, 0]},
            "no controller at all",
        ),
        ({"force_axes": [1, 1, 1, 0, 0, 0]}, "use_force"),  # set while the branch is off
    ],
)
def test_the_mask_rules_fail_on_a_bad_value(overrides, needle):
    with pytest.raises(ValueError) as err:
        controller(**overrides).validate(None)
    assert needle in str(err.value)

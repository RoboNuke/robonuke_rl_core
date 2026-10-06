"""Composition and the env-agnostic math: wrapper order, rot6d, the fragile projection.

No Isaac Lab here — `build_env` imports each wrapper only when its flag is on, so a config
with everything off is testable on CPU, and the pure tensor math (the 6-D rotation
representation, the axial/shear split) is testable on hand-computed cases.
"""

from __future__ import annotations

import math

import pytest
import torch

from robonuke_rl_core.envs.build import WRAPPER_ORDER, build_env, describe
from robonuke_rl_core.envs.cfg import ControllerCfg, WrappersCfg
from robonuke_rl_core.envs.forge.fragile import split_axial_shear
from robonuke_rl_core.envs.orientation import MODES, QUAT, ROT6D, quat_to_rot6d, rot6d_suffix


class FakeCfg:
    def __init__(self, controller=None, wrappers=None):
        self.controller = controller or ControllerCfg()
        self.wrappers = wrappers or WrappersCfg()


# ------------------------------------------------------------------ composition
def test_nothing_enabled_returns_the_env_untouched():
    env = object()
    cfg = FakeCfg()
    cfg.wrappers.task_metrics.enabled = False  # the one group that defaults to on
    assert build_env(cfg, env, "Isaac-Forge-PegInsert-Direct-v0") is env
    assert describe(cfg) == []


def test_task_metrics_are_on_by_default_and_skipped_off_forge():
    """They change no behaviour, so they ride along — but only where there is something to read."""
    cfg = FakeCfg()
    assert cfg.wrappers.task_metrics.enabled is True
    assert describe(cfg) == ["task_metrics"]

    env = object()  # a non-Forge task skips the wrapper instead of failing the run
    assert build_env(cfg, env, "Isaac-Lift-Cube-Franka-v0") is env


def test_the_wrapper_order_is_fixed_and_contact_is_last():
    """Contact appends to the observation, so nothing that edits obs may wrap after it."""
    assert WRAPPER_ORDER == (
        "controller", "efficient_reset", "fragile", "contact", "task_metrics",
    )
    assert WRAPPER_ORDER.index("contact") > WRAPPER_ORDER.index("fragile")
    assert WRAPPER_ORDER[0] == "controller"  # innermost: it owns the action space
    assert WRAPPER_ORDER[-1] == "task_metrics"  # outermost: it only observes


def test_describe_lists_only_what_is_enabled_in_order():
    wrappers = WrappersCfg()
    wrappers.contact.enabled = True
    wrappers.fragile.enabled = True
    wrappers.task_metrics.enabled = False
    cfg = FakeCfg(ControllerCfg(enabled=True), wrappers)
    assert describe(cfg) == ["controller", "fragile", "contact"]

    wrappers.efficient_reset.enabled = True
    wrappers.task_metrics.enabled = True
    assert describe(cfg) == [
        "controller", "efficient_reset", "fragile", "contact", "task_metrics",
    ]


def test_a_config_without_the_sections_is_a_no_op():
    env = object()
    assert build_env(object(), env, "Isaac-Forge-PegInsert-Direct-v0") is env


# ------------------------------------------------------------------ the 6-D rotation
def test_rot6d_is_the_first_two_columns_of_the_rotation_matrix():
    # 90 degrees about z: R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    half = math.sqrt(0.5)
    quat = torch.tensor([[half, 0.0, 0.0, half]])  # (w, x, y, z)
    assert quat_to_rot6d(quat)[0].tolist() == pytest.approx([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], abs=1e-6)

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert quat_to_rot6d(identity)[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    # 180 degrees about x: R = diag(1, -1, -1)
    about_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert quat_to_rot6d(about_x)[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])


def test_q_and_minus_q_give_the_same_answer():
    """The double cover is the whole reason this representation exists."""
    torch.manual_seed(0)
    quat = torch.randn(16, 4)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    assert torch.allclose(quat_to_rot6d(quat), quat_to_rot6d(-quat), atol=1e-6)


def test_the_columns_stay_orthonormal():
    torch.manual_seed(1)
    quat = torch.randn(32, 4)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    six = quat_to_rot6d(quat)
    first, second = six[:, :3], six[:, 3:]
    assert torch.allclose(torch.linalg.norm(first, dim=-1), torch.ones(32), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(second, dim=-1), torch.ones(32), atol=1e-5)
    assert torch.allclose((first * second).sum(-1), torch.zeros(32), atol=1e-5)


def test_an_unnormalized_quaternion_is_normalized_first():
    quat = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    assert quat_to_rot6d(quat)[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        quat_to_rot6d(torch.zeros(1, 3))


def test_the_mode_names_and_channel_suffix():
    assert MODES == (QUAT, ROT6D)
    assert rot6d_suffix("fingertip_quat") == "fingertip_quat_rot6d"


# ------------------------------------------------------------------ the fragile projection
def test_the_axial_shear_split_on_hand_computed_cases():
    axis = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    force = torch.tensor(
        [
            [0.0, 0.0, 10.0],  # pure axial
            [3.0, 4.0, 0.0],  # pure shear, magnitude 5
            [3.0, 4.0, 12.0],  # both
            [3.0, 4.0, 12.0],  # same force, axis along x
        ]
    )
    axial, shear = split_axial_shear(force, axis)
    assert axial.tolist() == pytest.approx([10.0, 0.0, 12.0, 3.0])
    assert shear.tolist() == pytest.approx([0.0, 5.0, 5.0, math.sqrt(4**2 + 12**2)])


def test_the_axial_component_is_a_magnitude_not_a_sign():
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    pressing = split_axial_shear(torch.tensor([[0.0, 0.0, -8.0]]), axis)
    pulling = split_axial_shear(torch.tensor([[0.0, 0.0, 8.0]]), axis)
    assert pressing[0].item() == pytest.approx(8.0)
    assert pulling[0].item() == pytest.approx(8.0)  # a peg breaks either way


def test_a_tilted_axis_is_renormalized():
    axis = torch.tensor([[0.0, 0.0, 5.0]])  # not a unit vector
    axial, shear = split_axial_shear(torch.tensor([[0.0, 3.0, 4.0]]), axis)
    assert axial.item() == pytest.approx(4.0)
    assert shear.item() == pytest.approx(3.0)


def test_the_split_adds_back_up_to_the_force_magnitude():
    torch.manual_seed(2)
    force = torch.randn(64, 3) * 10
    axis = torch.randn(64, 3)
    axial, shear = split_axial_shear(force, axis)
    assert torch.allclose(
        torch.sqrt(axial**2 + shear**2), torch.linalg.norm(force, dim=1), atol=1e-4
    )


# ------------------------------------------------------------------ the orientation rewrite
class FakeTaskCfg:
    def __init__(self, obs_order, state_order):
        self.obs_order = list(obs_order)
        self.state_order = list(state_order)


def test_the_order_rewrite_swaps_only_the_quaternion_channels():
    from robonuke_rl_core.envs.build import rewrite_orientation_order

    task_cfg = FakeTaskCfg(
        obs_order=["fingertip_pos", "fingertip_quat", "ee_linvel"],
        state_order=["fingertip_quat", "held_quat", "joint_pos"],
    )
    obs_dims = {"fingertip_pos": 3, "fingertip_quat": 4, "ee_linvel": 3, "joint_pos": 7}
    state_dims = dict(obs_dims)

    swapped = rewrite_orientation_order(task_cfg, obs_dims, state_dims)

    assert task_cfg.obs_order == ["fingertip_pos", "fingertip_quat_rot6d", "ee_linvel"]
    assert task_cfg.state_order == ["fingertip_quat_rot6d", "held_quat_rot6d", "joint_pos"]
    assert sorted(set(swapped)) == ["fingertip_quat", "held_quat"]
    # the new channels are registered at 6 dims, so the env sizes its spaces correctly
    assert obs_dims["fingertip_quat_rot6d"] == 6 and state_dims["held_quat_rot6d"] == 6
    # the observation grows by exactly 2 per swapped channel (6 - 4)
    assert sum(obs_dims[name] for name in task_cfg.obs_order) == 3 + 6 + 3


def test_the_rewrite_leaves_an_order_without_quaternions_alone():
    from robonuke_rl_core.envs.build import rewrite_orientation_order

    task_cfg = FakeTaskCfg(obs_order=["fingertip_pos"], state_order=["joint_pos"])
    assert rewrite_orientation_order(task_cfg, {}, {}) == []
    assert task_cfg.obs_order == ["fingertip_pos"] and task_cfg.state_order == ["joint_pos"]

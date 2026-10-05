"""The action interface: the layout table, the degenerate fills, and the gain maps.

Pure math, no env. This is the gold of the controller area — if the layout here and the
controller's reading of it ever disagree, a policy's gains become its force targets and
nothing crashes, so the table below is written out by hand rather than derived from the code
under test.
"""

from __future__ import annotations

import math

import pytest
import torch

from robonuke_rl_core.envs.cfg import ControllerCfg
from robonuke_rl_core.envs.interface import ActionInterface, ActionLayout, geometric_scale

ROWS = 4


def controller(**overrides) -> ControllerCfg:
    fields = dict(enabled=True, use_pose=True, use_force=False, gain_mapping="constant")
    fields.update(overrides)
    return ControllerCfg(**fields)


# ------------------------------------------------------------------ 1. the layout table
#  (use_pose, use_force, gain_mapping) -> (action_dim, pose, selection, force, gains)
#  the pose block is the env's own action vector: 7 on Forge (the default), 6 on Factory
LAYOUTS = {
    (True, False, "constant"): (7, (0, 7), None, None, None),
    (True, False, "variable_diagonal"): (13, (0, 7), None, None, (7, 13)),
    (False, True, "constant"): (6, None, None, (0, 6), None),
    (False, True, "variable_diagonal"): (12, None, None, (0, 6), (6, 12)),
    (True, True, "constant"): (19, (0, 7), (7, 13), (13, 19), None),
    (True, True, "variable_diagonal"): (31, (0, 7), (7, 13), (13, 19), (19, 31)),
}


@pytest.mark.parametrize("key, expected", sorted(LAYOUTS.items(), key=lambda kv: str(kv[0])))
def test_the_layout_matches_the_hand_written_table(key, expected):
    use_pose, use_force, mapping = key
    dim, pose, selection, force, gains = expected
    layout = ActionLayout(
        controller(use_pose=use_pose, use_force=use_force, gain_mapping=mapping)
    )

    assert layout.action_dim == dim
    for block, want in (
        (layout.pose_slice, pose),
        (layout.selection_slice, selection),
        (layout.force_slice, force),
        (layout.gain_slice, gains),
    ):
        if want is None:
            assert block.stop == block.start  # the block does not exist
        else:
            assert (block.start, block.stop) == want

    # the blocks tile the action vector with no gap and no overlap
    spans = [
        (b.start, b.stop)
        for b in (layout.pose_slice, layout.selection_slice, layout.force_slice, layout.gain_slice)
        if b.stop > b.start
    ]
    assert spans == sorted(spans)
    assert [s for s, _ in spans[1:]] == [e for _, e in spans[:-1]]
    assert (spans[0][0], spans[-1][1]) == (0, dim)


def test_the_selection_block_exists_exactly_when_both_branches_do():
    for use_pose, use_force in ((True, False), (False, True)):
        layout = ActionLayout(controller(use_pose=use_pose, use_force=use_force))
        assert layout.selection_indices == []
    assert ActionLayout(controller(use_force=True)).selection_indices == [7, 8, 9, 10, 11, 12]


def test_an_unknown_gain_mapping_raises():
    with pytest.raises(ValueError) as err:
        ActionLayout(controller(gain_mapping="cholesky"))
    assert "cholesky" in str(err.value)


# ------------------------------------------------------------------ 2. the degenerate fills
@pytest.mark.parametrize("mapping", ["constant", "variable_diagonal"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pose_only_is_exactly_impedance_for_any_action(mapping, seed):
    """S = I and f_d = 0, whatever the policy emits — that is what makes pose-only exact."""
    interface = ActionInterface(controller(gain_mapping=mapping))
    torch.manual_seed(seed)
    actions = torch.randn(ROWS, interface.action_dim) * 3.0  # well outside [-1, 1]

    _, selection, force_target, _, _ = interface.split(actions)
    assert torch.equal(selection, torch.ones(ROWS, 6))
    assert torch.equal(force_target, torch.zeros(ROWS, 6))


@pytest.mark.parametrize("mapping", ["constant", "variable_diagonal"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_force_only_hands_every_axis_to_the_force_law(mapping, seed):
    interface = ActionInterface(
        controller(use_pose=False, use_force=True, gain_mapping=mapping)
    )
    torch.manual_seed(seed)
    actions = torch.randn(ROWS, interface.action_dim) * 3.0

    pose_target, selection, _, _, _ = interface.split(actions)
    assert torch.equal(selection, torch.zeros(ROWS, 6))
    # nothing commands a pose: the env still gets an action of its own width, all zeros
    assert torch.equal(pose_target, torch.zeros(ROWS, interface.cfg.native_action_dim))


def test_selection_actions_are_read_as_the_bernoulli_sign():
    """+1 keeps an axis on position (S=1), -1 hands it to the force law (S=0)."""
    interface = ActionInterface(controller(use_force=True))
    actions = torch.zeros(1, interface.action_dim)
    actions[0, interface.selection_indices] = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    _, selection, _, _, _ = interface.split(actions)
    assert selection[0].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_the_force_target_scales_by_its_bounds():
    bounds = [50.0, 40.0, 30.0, 5.0, 4.0, 3.0]
    interface = ActionInterface(
        controller(use_pose=False, use_force=True, force_target_bounds=bounds)
    )
    actions = torch.tensor([[1.0, -1.0, 0.0, 0.5, -2.0, 1.0]])  # -2 clamps to -1

    _, _, force_target, _, _ = interface.split(actions)
    assert force_target[0].tolist() == pytest.approx([50.0, -40.0, 0.0, 2.5, -4.0, 3.0])


# ------------------------------------------------------------------ 3. constant gains
@pytest.mark.parametrize("seed", [0, 1])
def test_constant_gains_ignore_the_actions(seed):
    forces = [0.5, 0.4, 0.3, 0.02, 0.01, 0.03]
    interface = ActionInterface(
        controller(use_force=True, gain_mapping="constant", default_force_gains=forces)
    )
    torch.manual_seed(seed)
    first = interface.split(torch.randn(ROWS, interface.action_dim))
    second = interface.split(torch.randn(ROWS, interface.action_dim))

    assert torch.equal(first[3], second[3])  # K
    assert torch.equal(first[4], second[4])  # K_f
    assert first[4][0].tolist() == pytest.approx(forces)


# ------------------------------------------------------------------ 4. variable diagonal
def test_variable_diagonal_hits_min_mid_and_max_per_axis():
    low = [100.0, 200.0, 300.0, 5.0, 6.0, 7.0]
    high = [2000.0, 400.0, 3000.0, 50.0, 60.0, 70.0]
    interface = ActionInterface(
        controller(gain_mapping="variable_diagonal", gain_min=low, gain_max=high)
    )
    layout = interface.layout
    actions = torch.zeros(3, interface.action_dim)
    actions[0, layout.pose_gain_slice] = -1.0
    actions[1, layout.pose_gain_slice] = 0.0
    actions[2, layout.pose_gain_slice] = 1.0

    _, _, _, pose_gains, _ = interface.split(actions)
    assert pose_gains[0].tolist() == pytest.approx(low)
    assert pose_gains[2].tolist() == pytest.approx(high)
    # the midpoint is the geometric mean, hand-computed
    assert pose_gains[1].tolist() == pytest.approx(
        [math.sqrt(a * b) for a, b in zip(low, high)], rel=1e-6
    )


def test_variable_diagonal_scales_the_force_gains_from_their_own_bounds():
    interface = ActionInterface(
        controller(
            use_pose=False,
            use_force=True,
            gain_mapping="variable_diagonal",
            force_gain_min=[1.0] * 6,
            force_gain_max=[100.0] * 6,
        )
    )
    layout = interface.layout
    actions = torch.zeros(2, interface.action_dim)
    actions[1, layout.force_gain_slice] = 1.0

    _, _, _, _, force_gains = interface.split(actions)
    assert force_gains[0].tolist() == pytest.approx([10.0] * 6)  # sqrt(1 * 100)
    assert force_gains[1].tolist() == pytest.approx([100.0] * 6)


def test_a_zero_lower_bound_disables_that_axis_instead_of_exploding():
    actions = torch.tensor([[-1.0, 0.0, 1.0]])
    scaled = geometric_scale(actions, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
    assert scaled[0].tolist() == [0.0, 0.0, 0.0]
    assert torch.isfinite(scaled).all()


def test_both_branches_give_each_branch_its_own_gain_block():
    interface = ActionInterface(
        controller(
            use_force=True,
            gain_mapping="variable_diagonal",
            gain_min=[100.0] * 6,
            gain_max=[1000.0] * 6,
            force_gain_min=[1.0] * 6,
            force_gain_max=[10.0] * 6,
        )
    )
    layout = interface.layout
    actions = torch.zeros(1, interface.action_dim)
    actions[0, layout.pose_gain_slice] = 1.0  # max pose stiffness
    actions[0, layout.force_gain_slice] = -1.0  # min force stiffness

    _, _, _, pose_gains, force_gains = interface.split(actions)
    assert pose_gains[0].tolist() == pytest.approx([1000.0] * 6)
    assert force_gains[0].tolist() == pytest.approx([1.0] * 6)


# ------------------------------------------------------------------ 5. derived damping
@pytest.mark.parametrize("ratio", [1.0, 0.5, 2.0])
def test_damping_follows_the_configured_relation(ratio):
    interface = ActionInterface(controller(damping_ratio=ratio))
    gains = torch.tensor([[100.0, 400.0, 900.0, 0.0, 25.0, 1.0]])

    damping = interface.damping(gains)
    assert damping[0].tolist() == pytest.approx(
        [2.0 * ratio * math.sqrt(k) for k in gains[0].tolist()]
    )
    # damping is never commanded: there is no action block for it in any layout
    assert interface.action_dim == ActionLayout(interface.cfg).action_dim


def test_a_wrong_action_width_raises_with_the_layout():
    interface = ActionInterface(controller(use_force=True, gain_mapping="variable_diagonal"))
    with pytest.raises(ValueError) as err:
        interface.split(torch.zeros(ROWS, interface.action_dim - 1))
    assert "action_dim=31" in str(err.value)
    with pytest.raises(ValueError):
        interface.split(torch.zeros(interface.action_dim))  # not batched


# ------------------------------------------------------------------ 6. the hybrid axis mask
HYBRID_MASKS = {
    # mask -> (constant action_dim, variable action_dim, selection indices)
    (1, 1, 1, 0, 0, 0): (13, 22, [7, 8, 9]),          # the usual 3-D hybrid
    (0, 0, 1, 0, 0, 0): (9, 16, [7]),                 # force on z alone
    (1, 1, 1, 1, 1, 1): (19, 31, [7, 8, 9, 10, 11, 12]),  # 6-D
}


@pytest.mark.parametrize("mask, expected", sorted(HYBRID_MASKS.items()))
def test_the_force_side_blocks_are_as_wide_as_the_mask(mask, expected):
    constant_dim, variable_dim, selection = expected
    axes = sum(mask)

    layout = ActionLayout(controller(use_force=True, force_axes=list(mask)))
    assert layout.action_dim == constant_dim
    assert layout.selection_indices == selection
    assert layout.force_slice.stop - layout.force_slice.start == axes

    variable = ActionLayout(
        controller(use_force=True, force_axes=list(mask), gain_mapping="variable_diagonal")
    )
    assert variable.action_dim == variable_dim
    # K spans all six axes; K_f only the eligible ones
    assert variable.pose_gain_slice.stop - variable.pose_gain_slice.start == 6
    assert variable.force_gain_slice.stop - variable.force_gain_slice.start == axes


def test_axes_outside_the_mask_stay_position_controlled():
    mask = [1, 1, 1, 0, 0, 0]
    interface = ActionInterface(controller(use_force=True, force_axes=mask))
    actions = torch.zeros(1, interface.action_dim)
    actions[0, interface.selection_indices] = -1.0  # hand every eligible axis to the force law
    actions[0, interface.layout.force_slice] = 1.0

    _, selection, force_target, _, _ = interface.split(actions)
    assert selection[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    # the force target lands on the eligible axes and nowhere else
    assert force_target[0, 3:].tolist() == [0.0, 0.0, 0.0]
    assert force_target[0, :3].tolist() == pytest.approx([50.0, 50.0, 50.0])


def test_off_mask_axes_keep_the_constant_force_gain():
    mask = [0, 0, 1, 0, 0, 0]
    defaults = [0.5, 0.4, 0.3, 0.02, 0.01, 0.03]
    interface = ActionInterface(
        controller(
            use_force=True,
            force_axes=mask,
            gain_mapping="variable_diagonal",
            default_force_gains=defaults,
            force_gain_min=[2.0] * 6,
            force_gain_max=[200.0] * 6,
        )
    )
    actions = torch.zeros(1, interface.action_dim)
    actions[0, interface.layout.force_gain_slice] = 1.0  # max on the one eligible axis

    _, _, _, _, force_gains = interface.split(actions)
    assert force_gains[0, 2].item() == pytest.approx(200.0)
    assert force_gains[0, [0, 1, 3, 4, 5]].tolist() == pytest.approx(
        [defaults[i] for i in (0, 1, 3, 4, 5)]
    )


def test_the_mask_does_not_change_the_single_branch_layouts():
    """force_axes is a force-side concept: pose-only is unaffected by it."""
    plain = ActionLayout(controller())
    masked = ActionLayout(controller(force_axes=[1, 1, 1, 0, 0, 0]))
    assert (plain.action_dim, plain.selection_indices) == (masked.action_dim, masked.selection_indices)

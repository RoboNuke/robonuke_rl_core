"""learners/windows.py: device-side rolling windows, group ranks and host-to-device staging."""

import torch

from robonuke_rl_core.learners.windows import group_rank, push_window, stage_to_device


def test_group_rank_counts_earlier_active_rows_of_the_same_group():
    key = torch.tensor([0, 1, 0, 0, 1])
    active = torch.tensor([True, True, False, True, True])
    assert group_rank(key, active).tolist()[0] == 0
    assert group_rank(key, active)[3].item() == 1  # row 0 before it in group 0 (row 2 inactive)
    assert group_rank(key, active)[4].item() == 1  # row 1 before it in group 1


def test_push_window_writes_each_groups_rows_in_order_and_wraps():
    window = torch.full((2, 3), float("nan"))
    nxt = torch.zeros(2, dtype=torch.long)
    key = torch.tensor([0, 0, 1, 1])
    push_window(window, nxt, key, torch.tensor([True, True, True, False]), torch.tensor([1.0, 2.0, 3.0, 9.0]))
    assert window[0].tolist()[:2] == [1.0, 2.0] and window[1, 0].item() == 3.0 and nxt.tolist() == [2, 1]
    push_window(window, nxt, key, torch.tensor([True, True, False, False]), torch.tensor([4.0, 5.0, 0.0, 0.0]))
    assert window[0].tolist() == [5.0, 2.0, 4.0] and nxt.tolist() == [1, 1]  # the oldest overwritten


def test_push_window_keeps_the_last_n_when_more_finish_at_once():
    window = torch.full((1, 2), float("nan"))
    nxt = torch.zeros(1, dtype=torch.long)
    key = torch.zeros(4, dtype=torch.long)
    push_window(window, nxt, key, torch.ones(4, dtype=torch.bool), torch.tensor([1.0, 1.0, 0.0, 1.0]))
    assert sorted(window[0].tolist()) == [0.0, 1.0] and nxt.tolist() == [0]  # a deque's last two


def test_stage_to_device_copies_host_values():
    dst = torch.zeros(3, dtype=torch.bool)
    stage_to_device(dst, [True, False, True])
    assert dst.tolist() == [True, False, True]

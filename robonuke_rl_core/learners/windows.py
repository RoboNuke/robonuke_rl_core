"""Rolling windows and per-group ranks that stay on the device.

A learner keeps "the last N finished episodes" per agent (``trainer.success_window``, ``windowed_metrics``) and the
phase logger the same per (agent, phase). Which envs finished this step is a device tensor; reading it to the host
to know how many slots to write is a synchronization per step, which is what these helpers avoid: every env writes,
the ones that did not finish write nothing (a zero added to a slot they share), and the ring pointer advances by a
device count.
"""

from __future__ import annotations

import torch

__all__ = ["group_rank", "push_window", "stage_to_device"]


def group_rank(key: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Per row, how many active rows before it (lower index) share its ``key``: ``(n,)`` long. An inactive row's
    rank is meaningless (it is never used). ``n x n`` work, for the few-hundred-env runs these windows serve."""
    same = (key.unsqueeze(1) == key.unsqueeze(0)) & active.unsqueeze(0)
    earlier = torch.ones_like(same).tril(diagonal=-1)
    return (same & earlier).sum(dim=1)


def push_window(window: torch.Tensor, next_slot: torch.Tensor, key: torch.Tensor, active: torch.Tensor,
                values: torch.Tensor) -> None:
    """Write ``values[i]`` of every active row ``i`` into ring ``window[key[i]]`` (``(groups, size)``) at that group's
    next free slots, in row order, and advance ``next_slot`` (``(groups,)`` long) by each group's active count. In
    place, no host synchronization. More active rows of one group in one step than the window holds: the last
    ``size`` of them are kept, as a ``deque`` would."""
    groups, size = window.shape
    rank = group_rank(key, active)
    total = torch.zeros((groups,), dtype=torch.long, device=key.device)
    total.index_put_((key,), active.to(torch.long), accumulate=True)
    kept = active & (rank >= total[key] - size)
    slot = (next_slot[key] + rank) % size
    weight = kept.to(window.dtype)
    scratch = torch.zeros_like(window)
    count = torch.zeros_like(window)
    scratch.index_put_((key, slot), values.to(window.dtype) * weight, accumulate=True)
    count.index_put_((key, slot), weight, accumulate=True)
    window.copy_(torch.where(count > 0, scratch / count.clamp(min=1), window))
    next_slot.index_put_((key,), active.to(next_slot.dtype), accumulate=True)
    next_slot.remainder_(size)


def stage_to_device(dst: torch.Tensor, values) -> None:
    """Copy host ``values`` (a list, a CPU tensor) into the device buffer ``dst`` without a host synchronization:
    through pinned memory, asynchronously. ``torch.tensor(values, device=cuda)`` or ``dst.copy_(cpu_tensor)``
    from pageable memory is a synchronizing copy. The pinned staging block comes from torch's caching host
    allocator, which keeps it until the copy has run."""
    host = torch.as_tensor(values, dtype=dst.dtype).reshape(dst.shape)
    if dst.device.type == "cuda":
        dst.copy_(host.pin_memory(), non_blocking=True)
    else:
        dst.copy_(host)

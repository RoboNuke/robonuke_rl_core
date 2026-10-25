"""n-step TD transitions for SAC (``sac.n_step``).

A 1-step transition ``(s_t, a_t, r_t, s_{t+1})`` bootstraps from ``gamma * Q(s_{t+1})``. Its
n-step form carries the next ``m <= n`` rewards and bootstraps ``m`` steps later::

    R = sum_{k < m} gamma^k r_{t+k}        next = s_{t+m}        discount = gamma^m

with ``m = n`` unless the episode ends first: a window stops at the first step that is
``terminated`` or ``truncated`` (that step included), so it never crosses into the next
episode. The window's last step gives ``next_*``, ``terminated`` and ``truncated``; every other
field (the observation, the action, states, aux values) is the first step's. The bootstrap is
uncorrected: the later rewards are whatever the behaviour policy collected.

* :class:`NStepWindow` folds the online stream: each env-step it takes one 1-step row per env
  and, once it holds ``n`` steps, hands back the n-step row of the step ``n - 1`` back, one row
  per env, so the replay memory still gets exactly one row per env per step (``n - 1`` steps
  late; the last ``n - 1`` steps of a run are never stored).
* :func:`n_step_rows` does the same to a file of demo rows, which must be in episode order with
  every episode's last row ``terminated`` or ``truncated`` (checked: a row that is not an
  episode's last must hand its next observation to the row after it).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["NStepWindow", "n_step_rows"]

#: fields taken from a window's last step; the rewards are summed; the rest are the first step's
LAST_STEP_FIELDS = ("terminated", "truncated")


def _from_last(name: str) -> bool:
    return name.startswith("next_") or name in LAST_STEP_FIELDS


def _fold(
    steps: List[Dict[str, torch.Tensor]], gamma: float, stops: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """The n-step row of ``steps[0]`` over ``steps`` (each ``{name: (rows, width)}``, in time
    order), with ``discounts`` = ``gamma ** m``. ``stops`` ``(n, rows)``: where a window stops
    (that step included); by default the steps' own ``terminated | truncated``."""
    if stops is None:
        stops = torch.stack([(s["terminated"] | s["truncated"]).reshape(-1) for s in steps])
    dtype = steps[0]["rewards"].dtype
    # step k counts while no earlier step of the window stopped it
    alive = torch.ones_like(stops)
    alive[1:] = torch.cumprod(stops[:-1].logical_not().to(torch.int32), dim=0).bool()
    powers = gamma ** torch.arange(len(steps), dtype=dtype, device=stops.device).unsqueeze(1)
    rewards = torch.stack([s["rewards"].reshape(-1) for s in steps])  # (n, rows)
    last = alive.sum(dim=0) - 1  # (rows,): the window's last step
    out: Dict[str, torch.Tensor] = {}
    for name, first in steps[0].items():
        if name == "rewards":
            out[name] = (alive.to(dtype) * powers * rewards).sum(dim=0).reshape(first.shape)
        elif _from_last(name):
            stacked = torch.stack([s[name] for s in steps])  # (n, rows, width)
            index = last.view(1, -1, 1).expand(1, -1, stacked.shape[-1])
            out[name] = stacked.gather(0, index)[0]
        else:
            out[name] = first
    out["discounts"] = (gamma ** (last + 1).to(dtype)).reshape(-1, 1)
    return out


class NStepWindow:
    """The last ``n`` env-steps of every env, folded into n-step rows (see the module doc)."""

    def __init__(self, n: int, gamma: float) -> None:
        if n < 2:
            raise ValueError(f"an n-step window needs n >= 2, got {n} (n = 1 is the plain 1-step row)")
        self.n, self.gamma = int(n), float(gamma)
        self._steps: List[Dict[str, torch.Tensor]] = []

    def push(self, step: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Add one env-step (every value ``(num_envs, width)``); the n-step row of the step
        ``n - 1`` back once the window is full, else None."""
        self._steps.append(step)
        if len(self._steps) < self.n:
            return None
        row = _fold(self._steps, self.gamma)
        self._steps.pop(0)
        return row


def n_step_rows(transitions: Dict[str, torch.Tensor], n: int, gamma: float) -> Dict[str, torch.Tensor]:
    """Demo rows ``{name: (rows, width)}`` in episode order as n-step rows (with ``discounts``).
    The file's last row ends its episode whatever its flags say; any other row that ends no
    episode must hand its ``next_observations`` to the next row's ``observations``, so a file whose
    rows were dropped or reordered, or whose episodes end unmarked, is refused."""
    rows = transitions["rewards"].shape[0]
    ends = (transitions["terminated"] | transitions["truncated"]).reshape(-1).clone()
    inside = ends[:-1].logical_not()
    if rows > 1:
        handed = (transitions["next_observations"][:-1] == transitions["observations"][1:]).all(dim=-1)
        broken = torch.nonzero(inside & handed.logical_not()).flatten()
        if broken.numel():
            raise ValueError(
                f"sac.n_step: demo row {int(broken[0])} ends no episode (neither terminated nor truncated) but "
                "its next observation is not the next row's observation: n-step windows need the demo rows in "
                "episode order with every episode's last row marked (store, not drop, the out-of-reach actions)"
            )
    ends[-1] = True  # a window never runs past the file
    position = torch.arange(rows, device=ends.device)
    index = [torch.clamp(position + k, max=rows - 1) for k in range(n)]
    steps = [{name: value[i] for name, value in transitions.items()} for i in index]
    return _fold(steps, gamma, stops=torch.stack([ends[i] for i in index]))

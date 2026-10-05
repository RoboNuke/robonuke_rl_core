"""The one memory class: a per-agent ring buffer with batched sampling.

Replaces the skrl-derived port from RoboNuke/generalized_hybrid_vic_action_space
``memory/multi_random.py``. Self-contained — no skrl base class, no ``(depth, num_envs)``
layout, no per-env ``randperm``.

Storage is one tensor per name, shaped ``(num_agents, capacity, dim)``: an agent's data is a
contiguous slab, so

* ``capacity`` is exactly the transitions the config asked for, per agent, for any positive
  integer (the old layout could only hold multiples of the envs per agent);
* a sampled batch flattens to ``(num_agents * rows, dim)`` in ``[agent 0 | agent 1 | ...]``
  order, which is what the vmap models reshape back to ``(num_agents, rows, dim)``;
* sampling is two kernels for the whole buffer instead of one ``randperm`` per env.

All agents share one write pointer: every ``add_samples`` call writes one env-step for every
env, so the agents stay in lockstep by construction. Within an agent the row for step ``s``
and its local env ``e`` is ``(s * envs_per_agent + e) % capacity``, so a buffer whose capacity
divides evenly reshapes to ``(num_agents, steps, envs_per_agent, dim)`` — see :meth:`time_view`,
which is how PPO gets the time axis GAE needs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import gymnasium
import numpy as np
import torch

__all__ = ["MultiRandomMemory"]


def _data_size(size: Union[int, Sequence[int], gymnasium.Space]) -> int:
    """Elements a single sample of ``size`` occupies in the last dimension."""
    if isinstance(size, (int, np.integer)):
        return int(size)
    if isinstance(size, (tuple, list)):
        return int(np.prod(size))
    if isinstance(size, gymnasium.spaces.Discrete):
        return 1  # stored as one index, not one-hot
    if isinstance(size, gymnasium.spaces.MultiDiscrete):
        return int(size.nvec.shape[0])
    if isinstance(size, gymnasium.Space):
        return int(gymnasium.spaces.flatdim(size))
    raise TypeError(f"cannot size a memory tensor from {size!r} ({type(size).__name__})")


class MultiRandomMemory:
    """Replay (SAC) or rollout (PPO) storage for ``num_agents`` independent agents.

    :param capacity: transitions held **per agent**; any positive integer >= the envs per agent.
    :param num_envs: total envs in the shared simulation, laid out
        ``[agent 0 envs, agent 1 envs, ...]``.
    :param num_agents: agents trained in parallel; must divide ``num_envs``.
    """

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        num_agents: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if num_envs % num_agents != 0:
            raise ValueError(
                f"num_envs ({num_envs}) must be divisible by num_agents ({num_agents})"
            )
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.envs_per_agent = self.num_envs // self.num_agents
        self.capacity = int(capacity)
        if self.capacity < self.envs_per_agent:
            raise ValueError(
                f"capacity ({capacity}) must be at least the envs per agent "
                f"({self.envs_per_agent}): one env-step already writes that many rows"
            )
        self.device = torch.device(device if device is not None else "cpu")

        self.tensors: Dict[str, torch.Tensor] = {}
        #: next row each agent writes
        self.pointer = 0
        #: rows written per agent so far, saturating at ``capacity``
        self.size = 0
        # (num_agents, 1) so it broadcasts against a (num_agents, rows) index tensor
        self._agent_index = torch.arange(self.num_agents, device=self.device).unsqueeze(-1)

    # ------------------------------------------------------------------ properties
    @property
    def filled(self) -> bool:
        """Whether every agent's buffer has been written at least once end to end."""
        return self.size >= self.capacity

    @property
    def num_steps(self) -> int:
        """Env-steps the buffer holds per agent. Requires an evenly divisible capacity."""
        self._require_even_capacity("num_steps")
        return self.capacity // self.envs_per_agent

    def _require_even_capacity(self, what: str) -> None:
        if self.capacity % self.envs_per_agent:
            raise ValueError(
                f"{what} needs a capacity that is a multiple of the envs per agent, but "
                f"capacity={self.capacity} and envs_per_agent={self.envs_per_agent}; the rows "
                "of one env-step would straddle the wrap point"
            )

    # ------------------------------------------------------------------ tensors
    def create_tensor(
        self,
        name: str,
        *,
        size: Union[int, Sequence[int], gymnasium.Space, None],
        dtype: Optional[torch.dtype] = None,
    ) -> bool:
        """Allocate ``(num_agents, capacity, size)``. Returns False when ``size`` is None.

        Floating-point tensors start as NaN: a row that is sampled before it is written shows
        up as NaN losses instead of plausible zeros.
        """
        if size is None:
            return False
        width = _data_size(size)
        if name in self.tensors:
            existing = self.tensors[name]
            if existing.shape[-1] != width:
                raise ValueError(
                    f"tensor '{name}' exists with width {existing.shape[-1]}, not {width}"
                )
            if dtype is not None and existing.dtype != dtype:
                raise ValueError(
                    f"tensor '{name}' exists with dtype {existing.dtype}, not {dtype}"
                )
            return False
        tensor = torch.zeros(
            (self.num_agents, self.capacity, width), device=self.device, dtype=dtype
        )
        if torch.is_floating_point(tensor):
            tensor.fill_(float("nan"))
        self.tensors[name] = tensor
        return True

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        """The stored tensor itself, ``(num_agents, capacity, dim)``; writes show through."""
        if name not in self.tensors:
            raise KeyError(f"no memory tensor '{name}'; created: {sorted(self.tensors)}")
        return self.tensors[name]

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Overwrite a whole tensor in place (PPO's returns and advantages)."""
        target = self.get_tensor_by_name(name)
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"tensor '{name}' is {tuple(target.shape)}, cannot be set from "
                f"{tuple(tensor.shape)}"
            )
        target.copy_(tensor)

    def time_view(self, name: str) -> torch.Tensor:
        """``(num_agents, steps, envs_per_agent, dim)`` view of a stored tensor.

        A view, not a copy: writing through it writes the buffer. Row ``s * envs_per_agent + e``
        holds step ``s`` of local env ``e``, so index 1 is time and index 2 is the env — what
        GAE needs. Raises unless the capacity divides evenly by the envs per agent.
        """
        self._require_even_capacity("time_view")
        tensor = self.get_tensor_by_name(name)
        return tensor.view(
            self.num_agents, self.capacity // self.envs_per_agent, self.envs_per_agent, -1
        )

    # ------------------------------------------------------------------ writing
    def add_samples(self, **values: torch.Tensor) -> None:
        """Write one env-step: every value is ``(num_envs, dim)`` in env order.

        One call advances the shared pointer by ``envs_per_agent`` rows, whichever tensors it
        carries; call it once per env-step with everything that step produced.
        """
        if not values:
            raise ValueError("add_samples() got no tensors")
        rows = self.envs_per_agent
        start = self.pointer
        end = start + rows
        for name, value in values.items():
            target = self.get_tensor_by_name(name)
            if value.shape[0] != self.num_envs:
                raise ValueError(
                    f"'{name}': expected {self.num_envs} rows (one per env), got "
                    f"{tuple(value.shape)}"
                )
            block = value.reshape(self.num_agents, rows, -1)
            if block.shape[-1] != target.shape[-1]:
                raise ValueError(
                    f"'{name}': expected width {target.shape[-1]}, got {block.shape[-1]}"
                )
            if end <= self.capacity:
                target[:, start:end] = block
            else:  # the step straddles the wrap point
                head = self.capacity - start
                target[:, start:] = block[:, :head]
                target[:, : end - self.capacity] = block[:, head:]
        self.pointer = end % self.capacity
        self.size = min(self.size + rows, self.capacity)

    # ------------------------------------------------------------------ sampling
    def _gather(self, names: Sequence[str], indices: torch.Tensor) -> List[torch.Tensor]:
        """Rows ``indices`` (``(num_agents, rows)``) of each tensor, flattened block-wise."""
        out = []
        for name in names:
            tensor = self.get_tensor_by_name(name)
            out.append(tensor[self._agent_index, indices].reshape(-1, tensor.shape[-1]))
        return out

    def sample(self, *, names: Sequence[str], batch_size: int) -> List[List[torch.Tensor]]:
        """``batch_size`` random transitions **per agent**, drawn with replacement.

        Returns ``[[tensor, ...]]`` (one mini-batch) with each tensor
        ``(num_agents * batch_size, dim)`` in ``[agent 0 | agent 1 | ...]`` order. Agent ``i``'s
        rows come from agent ``i``'s slab only, and never from a row that was never written.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if self.size == 0:
            raise ValueError("cannot sample from an empty memory")
        indices = torch.randint(
            0, self.size, (self.num_agents, batch_size), device=self.device
        )
        return [self._gather(names, indices)]

    def sample_all(
        self,
        *,
        names: Sequence[str],
        mini_batches: int = 1,
        shuffle: bool = True,
    ) -> List[List[torch.Tensor]]:
        """Every written row, once, split into ``mini_batches`` equal parts.

        Each agent gets its own permutation (one ``rand`` + one ``argsort`` for the whole
        buffer), so the mini-batch an agent's row lands in is independent of the other agents.
        """
        if mini_batches < 1:
            raise ValueError(f"mini_batches must be >= 1, got {mini_batches}")
        if self.size == 0:
            raise ValueError("cannot sample from an empty memory")
        if self.size % mini_batches:
            raise ValueError(
                f"mini_batches ({mini_batches}) must divide the {self.size} rows stored per "
                "agent, so every row is used exactly once"
            )
        rows = self.size // mini_batches
        if shuffle:
            order = torch.argsort(
                torch.rand(self.num_agents, self.size, device=self.device), dim=1
            )
        else:
            order = torch.arange(self.size, device=self.device).expand(self.num_agents, self.size)
        return [
            self._gather(names, order[:, start : start + rows])
            for start in range(0, self.size, rows)
        ]

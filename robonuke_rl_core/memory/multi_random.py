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

:meth:`add_rows` takes the agents out of lockstep (each writes the envs it keeps at its own
pointer). From then on the pointer, fill and rows-written of every agent live on the device
(``_agent_pointer_t``, ``_agent_size_t``, ``_agent_written_t``) and the write is a gather and a
scatter per tensor with no host synchronization: which envs are kept is a device tensor, so a
``nonzero`` per agent per step (the old write) would stall the CPU on the GPU every step. The
host-side views ``_agent_pointer`` / ``_agent_size`` / ``_agent_written`` / ``size`` are
properties that read the device (one synchronization each): for the update decision, the
periodic reset and the checkpoints, never for the per-step path.
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
        #: next row each agent writes (lockstep: :meth:`add_samples`)
        self.pointer = 0
        #: rows written per agent so far, saturating at ``capacity`` (lockstep; see :attr:`size`)
        self._size = 0
        # (num_agents, 1) so it broadcasts against a (num_agents, rows) index tensor
        self._agent_index = torch.arange(self.num_agents, device=self.device).unsqueeze(-1)
        #: ``size`` on the device, for :meth:`sample_graph_safe`; refreshed by :meth:`sync_size`
        self._size_on_device = torch.zeros((), dtype=torch.float32, device=self.device)
        #: each agent's fill on the device as float, for a memory written with add_rows / add_block: a captured
        #: sampler reads the per-agent bounds from here (kept current by every write)
        self._agent_size_on_device = torch.zeros((self.num_agents,), dtype=torch.float32, device=self.device)
        #: :meth:`add_rows` / :meth:`add_block`: each agent's own write pointer, fill and rows ever written (more
        #: than the capacity: wrapped), ``(num_agents,)`` long on the device; None until the first such write, after
        #: which ``size`` is the smallest agent's fill. Updated in place only (a captured graph may read them).
        self._agent_pointer_t: Optional[torch.Tensor] = None
        self._agent_size_t: Optional[torch.Tensor] = None
        self._agent_written_t: Optional[torch.Tensor] = None
        self._capacity_t = torch.tensor(self.capacity, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ the per-agent bookkeeping
    def _begin_agent_streams(self) -> None:
        """Leave lockstep: every agent's pointer and fill start from the shared ones, on the device."""
        if self._agent_pointer_t is not None:
            return
        self._agent_pointer_t = torch.full((self.num_agents,), int(self.pointer), dtype=torch.long, device=self.device)
        self._agent_size_t = torch.full((self.num_agents,), int(self._size), dtype=torch.long, device=self.device)
        self._agent_written_t = torch.zeros((self.num_agents,), dtype=torch.long, device=self.device)
        self._agent_size_on_device.copy_(self._agent_size_t.to(torch.float32))

    def _advance(self, count: torch.Tensor) -> None:
        """Each agent wrote ``count`` (``(num_agents,)`` long) rows: move its pointer, fill and total, in place."""
        self._agent_pointer_t.copy_((self._agent_pointer_t + count) % self.capacity)
        self._agent_size_t.copy_(torch.minimum(self._agent_size_t + count, self._capacity_t))
        self._agent_written_t.add_(count)
        self._agent_size_on_device.copy_(self._agent_size_t.to(torch.float32))

    def _host_list(self, tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
        return None if tensor is None else [int(v) for v in tensor.tolist()]

    def _set_host_list(self, name: str, values: Optional[Sequence[int]]) -> None:
        if values is None:
            self._agent_pointer_t = self._agent_size_t = self._agent_written_t = None
            return
        self._begin_agent_streams()
        getattr(self, name).copy_(torch.tensor([int(v) for v in values], dtype=torch.long, device=self.device))
        self._agent_size_on_device.copy_(self._agent_size_t.to(torch.float32))

    @property
    def size(self) -> int:
        """Rows written per agent, saturating at ``capacity``: the lockstep count, or, out of lockstep, the
        smallest agent's fill (a host synchronization; the per-step path never reads it)."""
        if self._agent_size_t is None:
            return self._size
        return int(self._agent_size_t.min().item())

    @size.setter
    def size(self, value: int) -> None:
        self._size = int(value)

    @property
    def _agent_pointer(self) -> Optional[List[int]]:
        """Each agent's write pointer as a host list (one synchronization), None in lockstep."""
        return self._host_list(self._agent_pointer_t)

    @_agent_pointer.setter
    def _agent_pointer(self, values: Optional[Sequence[int]]) -> None:
        self._set_host_list("_agent_pointer_t", values)

    @property
    def _agent_size(self) -> Optional[List[int]]:
        """Each agent's fill as a host list (one synchronization), None in lockstep."""
        return self._host_list(self._agent_size_t)

    @_agent_size.setter
    def _agent_size(self, values: Optional[Sequence[int]]) -> None:
        self._set_host_list("_agent_size_t", values)

    @property
    def _agent_written(self) -> Optional[List[int]]:
        """Each agent's rows ever written as a host list (one synchronization), None in lockstep."""
        return self._host_list(self._agent_written_t)

    @_agent_written.setter
    def _agent_written(self, values: Optional[Sequence[int]]) -> None:
        self._set_host_list("_agent_written_t", values)

    def fills(self) -> List[int]:
        """Each agent's fill as a host list: one synchronization. Out of lockstep every agent has its own; in
        lockstep they are all :attr:`size`."""
        return self._agent_size if self._agent_size_t is not None else [self._size] * self.num_agents

    def written_on_device(self) -> torch.Tensor:
        """Each agent's rows ever written, ``(num_agents,)`` long on the device, no synchronization."""
        if self._agent_written_t is not None:
            return self._agent_written_t
        return torch.full((self.num_agents,), int(self._size), dtype=torch.long, device=self.device)

    def fills_on_device(self) -> torch.Tensor:
        """Each agent's fill, ``(num_agents,)`` float32 on the device, no synchronization."""
        if self._agent_size_t is not None:
            return self._agent_size_on_device
        return torch.full((self.num_agents,), float(self._size), device=self.device)

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
    def add_rows(self, keep: torch.Tensor, **values: torch.Tensor) -> None:
        """Write only the envs ``keep`` selects (``(num_envs,)`` bool) of one env-step: every value is
        ``(num_envs, dim)`` in env order, and each agent appends its kept rows at its own pointer, so the agents'
        fills drift apart (sampling then bounds each agent by its own fill). From then on :meth:`add_samples` writes
        through here too, and there is no :meth:`time_view` (PPO's lockstep)."""
        if not values:
            raise ValueError("add_rows() got no tensors")
        keep = keep.reshape(-1).to(torch.bool)
        if keep.shape[0] != self.num_envs:
            raise ValueError(f"keep must hold {self.num_envs} flags (one per env), got {tuple(keep.shape)}")
        self._begin_agent_streams()
        blocks = {}
        for name, value in values.items():
            target = self.get_tensor_by_name(name)
            if value.shape[0] != self.num_envs:
                raise ValueError(f"'{name}': expected {self.num_envs} rows (one per env), got {tuple(value.shape)}")
            block = value.reshape(self.num_agents, self.envs_per_agent, -1)
            if block.shape[-1] != target.shape[-1]:
                raise ValueError(f"'{name}': expected width {target.shape[-1]}, got {block.shape[-1]}")
            blocks[name] = block
        kept = keep.view(self.num_agents, self.envs_per_agent)
        count = kept.sum(dim=1)  # (num_agents,): what each agent appends
        # every env gets its own slot in [pointer, pointer + envs_per_agent): the kept envs first, in order (their
        # new rows), then the others, which write their slot's current content back (a no-op). Distinct slots, so
        # the scatter is deterministic, and nothing is read to the host.
        kept_rank = torch.cumsum(kept.to(torch.long), dim=1) - 1
        other_rank = torch.cumsum((~kept).to(torch.long), dim=1) - 1
        offset = torch.where(kept, kept_rank, count.unsqueeze(1) + other_rank)
        positions = (self._agent_pointer_t.unsqueeze(1) + offset) % self.capacity  # (num_agents, envs_per_agent)
        select = kept.unsqueeze(-1)
        for name, block in blocks.items():
            target = self.tensors[name]
            current = target[self._agent_index, positions]
            target[self._agent_index, positions] = torch.where(select, block.to(target.dtype), current)
        self._advance(count)

    def add_block(self, **values: torch.Tensor) -> None:
        """Append the same ``rows`` rows (every value ``(rows, dim)``) to every agent's stream, each at its own
        pointer, in order (e.g. demo transitions written into the memory once, then overwritten like any other
        row). Leaves the memory out of lockstep, as :meth:`add_rows`."""
        if not values:
            raise ValueError("add_block() got no tensors")
        rows = {int(value.shape[0]) for value in values.values()}
        if len(rows) != 1:
            raise ValueError(f"add_block() values differ in rows: { {n: tuple(v.shape) for n, v in values.items()} }")
        count = rows.pop()
        if count > self.capacity:
            raise ValueError(f"add_block() of {count} rows exceeds the memory's capacity {self.capacity} per agent")
        self._begin_agent_streams()
        for name, value in values.items():
            target = self.get_tensor_by_name(name)
            if value.reshape(count, -1).shape[-1] != target.shape[-1]:
                raise ValueError(f"'{name}': expected width {target.shape[-1]}, got {tuple(value.shape)}")
        positions = (self._agent_pointer_t.unsqueeze(1) + torch.arange(count, device=self.device)) % self.capacity
        for name, value in values.items():
            target = self.tensors[name]
            rows = value.reshape(count, -1).to(self.device, target.dtype)
            target[self._agent_index, positions] = rows.unsqueeze(0).expand(self.num_agents, -1, -1)
        self._advance(torch.full((self.num_agents,), count, dtype=torch.long, device=self.device))

    def keep_rows(self, keep: torch.Tensor) -> List[int]:
        """Compact each agent's stored rows to those ``keep`` (``(num_agents, size)`` bool over the written rows)
        selects, in their stored order, and continue writing after them (the agents' fills then differ, as with
        :meth:`add_rows`). Returns the rows each agent kept."""
        self._begin_agent_streams()
        kept_counts = []
        for agent, size in enumerate(self._agent_size):  # a rare, host-driven compaction (one sync for the fills)
            rows = keep[agent, :size].nonzero(as_tuple=False).reshape(-1)
            count = int(rows.numel())
            for tensor in self.tensors.values():
                tensor[agent, :count] = tensor[agent, rows].clone()
            kept_counts.append(count)
        self._agent_size = kept_counts
        self._agent_pointer = [count % self.capacity for count in kept_counts]
        self._agent_written = kept_counts
        return kept_counts

    def add_samples(self, **values: torch.Tensor) -> None:
        """Write one env-step: every value is ``(num_envs, dim)`` in env order.

        One call advances the shared pointer by ``envs_per_agent`` rows, whichever tensors it
        carries; call it once per env-step with everything that step produced.
        """
        if self._agent_pointer_t is not None:
            # out of lockstep (add_rows, keep_rows): every env's row, each agent at its own pointer
            self.add_rows(torch.ones(self.num_envs, dtype=torch.bool, device=self.device), **values)
            return
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
        if self._agent_size_t is not None:  # add_rows: each agent bounded by its own fill
            # an agent with no rows yet (its fill 0) gets row 0 as filler: a learner masks such an agent out of the
            # update (SAC._update), so its filler rows reach no loss it is stepped on. No host read: a memory with
            # nothing in it samples NaN rows, which the masked update never steps on
            sizes = self._agent_size_on_device.unsqueeze(-1)
            uniform = torch.rand((self.num_agents, batch_size), device=self.device)
            indices = (uniform * sizes).long().clamp_(max=self.capacity - 1)
            return [self._gather(names, indices)]
        if self._size == 0:
            raise ValueError("cannot sample from an empty memory")
        indices = torch.randint(
            0, self.size, (self.num_agents, batch_size), device=self.device
        )
        return [self._gather(names, indices)]

    def count_where(self, flag: str, value: float) -> List[int]:
        """Per agent, how many of its written rows have ``flag`` (a one-wide tensor) equal to ``value``."""
        tensor = self.get_tensor_by_name(flag)
        sizes = self.fills()
        return [int((tensor[agent, : sizes[agent], 0] == value).sum()) for agent in range(self.num_agents)]

    def sample_where(self, *, names: Sequence[str], batch_size: int, flag: str, value: float) -> List[List[torch.Tensor]]:
        """:meth:`sample` restricted to the rows whose ``flag`` (a one-wide tensor) equals ``value``: ``batch_size``
        of them per agent, drawn uniformly with replacement from that agent's eligible rows (e.g. the online rows of a
        memory that also holds flagged demo rows). Raises when an agent has none."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        tensor = self.get_tensor_by_name(flag)
        sizes = self.fills()
        indices = torch.empty((self.num_agents, batch_size), dtype=torch.long, device=self.device)
        for agent in range(self.num_agents):
            eligible = (tensor[agent, : sizes[agent], 0] == value).nonzero(as_tuple=False).reshape(-1)
            if eligible.numel() == 0:
                raise ValueError(f"agent {agent} has no row with {flag} == {value} to sample")
            indices[agent] = eligible[torch.randint(0, eligible.numel(), (batch_size,), device=self.device)]
        return [self._gather(names, indices)]

    def sync_size(self) -> None:
        """Copy ``size`` to the device (an async fill, no host sync) for
        :meth:`sample_graph_safe`. Call before each replay of a captured sampler."""
        if self._agent_size_t is not None:
            self._size_on_device.copy_(self._agent_size_t.min().to(torch.float32))  # device work only
        else:
            self._size_on_device.fill_(float(self._size))

    def sample_graph_safe(self, *, names: Sequence[str], batch_size: int) -> List[List[torch.Tensor]]:
        """:meth:`sample`, with the row bound read from the device (``sync_size``) instead of
        baked in as a Python number, so a CUDA graph that captured it stays right as the
        memory fills. Uniform over the written rows, like :meth:`sample`; a different random
        stream (``floor(u * size)`` rather than ``randint``)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        uniform = torch.rand((self.num_agents, batch_size), device=self.device)
        if self._agent_size_t is not None:  # add_rows / add_block: each agent bounded by its own fill
            indices = (uniform * self._agent_size_on_device.unsqueeze(-1)).long().clamp_(max=self.capacity - 1)
        else:
            indices = (uniform * self._size_on_device).long().clamp_(max=self.capacity - 1)
        return [self._gather(names, indices)]

    def sample_range_graph_safe(self, *, names: Sequence[str], batch_size: int, low: torch.Tensor) -> List[List[torch.Tensor]]:
        """:meth:`sample_graph_safe` over each agent's rows ``[low, fill)`` only (``low`` a ``(num_agents,)`` device
        tensor, e.g. the end of a block of demo rows written first): device work only, so a captured graph stays right
        as the memory fills. The caller keeps ``low < fill`` and the memory unwrapped."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        sizes = self._agent_size_on_device if self._agent_size_t is not None else self._size_on_device.expand(self.num_agents)
        low = low.to(torch.float32)
        uniform = torch.rand((self.num_agents, batch_size), device=self.device)
        indices = (low.unsqueeze(-1) + uniform * (sizes - low).unsqueeze(-1)).long().clamp_(max=self.capacity - 1)
        return [self._gather(names, indices)]

    def written(self) -> List[int]:
        """Per agent, the rows ever written (a fill that stopped at the capacity has wrapped)."""
        return self._agent_written if self._agent_written_t is not None else [self._size] * self.num_agents

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

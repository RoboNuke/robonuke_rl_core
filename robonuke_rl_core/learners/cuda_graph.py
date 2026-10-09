"""Capture a learner's gradient step as CUDA graphs and replay them.

A small-batch update is launch-bound: thousands of microsecond kernels, each waiting on
Python. :class:`StepGraphs` records one kind of step (a key such as "critic only" or "critic +
actor") the first time it is needed after a few eager warm-up runs, and from then on replays
it as a single launch. The step function must do device work only — no host sync, no Python
value computed from a tensor, no tensor reassigned between calls (write in place) — and read
everything that changes between calls from tensors it reads in place (the replay memory's
size, the learning rates, the entropy coefficient). Its returned tensors are rewritten by
every replay of its graph, so read them before replaying that graph again.

Every run, eager or replayed, is a real step: the warm-up runs count. Captures stay valid
until the modules or optimizers they recorded are replaced: call :meth:`clear` then.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable

import torch

__all__ = ["StepGraphs"]


class StepGraphs:
    def __init__(self, warmup: int = 3) -> None:
        #: eager runs of a key before it is captured (on a side stream, as torch recommends,
        #: so lazy initialization such as cuBLAS workspaces happens outside the capture)
        self.warmup = int(warmup)
        self._graphs: Dict[Hashable, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Hashable, Any] = {}
        self._runs: Dict[Hashable, int] = {}

    def run(self, key: Hashable, step: Callable[[], Any]) -> Any:
        """One step of kind ``key``: eager while warming up, then captured once, then
        replayed. Returns the step's output tensors."""
        graph = self._graphs.get(key)
        if graph is not None:
            graph.replay()
            return self._outputs[key]
        runs = self._runs.get(key, 0)
        if runs < self.warmup:
            self._runs[key] = runs + 1
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = step()
            torch.cuda.current_stream().wait_stream(side)
            return out
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()  # recorded, not executed
        self._graphs[key] = graph
        self._outputs[key] = out
        graph.replay()  # this call's step
        return out

    @property
    def captured(self) -> int:
        return len(self._graphs)

    def clear(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._runs.clear()

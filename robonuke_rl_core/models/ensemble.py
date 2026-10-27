"""Stack N plain modules and run them in one batched pass.

``VmapEnsemble`` builds ``num_agents`` copies of an ordinary single-agent ``nn.Module``,
stacks their parameters with ``torch.func.stack_module_state``, and calls them with
``torch.vmap``. Writing a model is therefore writing normal PyTorch: no block layers, no
einsum, no per-agent ``ParameterList``.

The stacked parameters carry a leading agent dimension, so everything else in the package
works on them unchanged — ``BlockAdamW``, ``clip_grad_norm_per_agent``, and the per-agent
checkpoint slices.

Rules for the module being stacked:

* **No sampling and no in-place buffer mutation inside ``forward``.** vmap cannot batch
  either. Return distribution parameters; build the distribution outside (that is what the
  wrappers in ``simba.py`` do).
* Any vmap "falling back to a for-loop" warning is a failure, not a nuisance: it means an op
  in the module is not batchable. :meth:`VmapEnsemble.forward` turns it into an error.

``members > 1`` adds a second stacked axis **inside** each agent: ``members`` independent
copies per agent (a Q ensemble), parameters ``(num_agents, members, ...)``, all called in one
nested vmap on the same inputs, output ``(num_agents, members, rows, ...)``. The leading
dimension is still the agent, so the per-agent rules (BlockAdamW's keep mask, per-agent
clipping, per-agent checkpoint slices) hold unchanged: freezing an agent freezes all of its
members. Copies are built member-major (member 0 for every agent, then member 1, ...), the
order separate per-member ensembles were built in.
"""

from __future__ import annotations

import contextlib
import copy
import warnings
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state

__all__ = ["VmapEnsemble"]


class VmapEnsemble(nn.Module):
    """``num_agents`` copies of ``build_fn()``, batched with vmap."""

    def __init__(
        self, build_fn: Callable[[], nn.Module], num_agents: int, device=None, members: int = 1
    ) -> None:
        super().__init__()
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if members < 1:
            raise ValueError(f"members must be >= 1, got {members}")
        # Sequential RNG: every agent gets its own init, as the block models did. The build
        # runs under the target device, so every parameter is ALLOCATED there rather than
        # created on the CPU and copied — which also means a module's own init math (an
        # output scale, a log-std bias) meets tensors on its own device. The .to() after it
        # is a no-op for a well-behaved build_fn and a safety net for one that hardcodes a
        # device.
        context = torch.device(device) if device is not None else contextlib.nullcontext()
        with context:
            models = [build_fn() for _ in range(num_agents * members)]  # member-major
        if device is not None:
            models = [model.to(device) for model in models]
        for model in models[1:]:
            if type(model) is not type(models[0]):
                raise TypeError("build_fn must return the same module type every call")

        params, buffers = stack_module_state(models)
        if members > 1:
            # (members * agents, ...) member-major -> (agents, members, ...)
            def regroup(tensor):
                shaped = tensor.reshape(members, num_agents, *tensor.shape[1:])
                return shaped.transpose(0, 1).contiguous()

            params = {name: regroup(t) for name, t in params.items()}
            buffers = {name: regroup(t) for name, t in buffers.items()}
        self.members = members
        self.num_agents = num_agents
        self._param_names: List[str] = list(params)
        self._buffer_names: List[str] = list(buffers)
        # '.' is not allowed in an attribute name, so the dotted state-dict key is mangled
        for name, tensor in params.items():
            self.register_parameter(self._mangle(name), nn.Parameter(tensor.detach().clone()))
        for name, tensor in buffers.items():
            self.register_buffer(self._mangle(name), tensor.detach().clone())
        # A meta-device copy carries the structure for functional_call without any storage.
        # object.__setattr__ keeps it OUT of the module tree: registered as a submodule, its
        # meta parameters would show up in parameters() and state_dict(), where they would
        # break the optimizer (no agent dimension) and the checkpoints.
        object.__setattr__(self, "_meta", copy.deepcopy(models[0]).to("meta"))

    # ------------------------------------------------------------------ naming
    @staticmethod
    def _mangle(name: str) -> str:
        return name.replace(".", "__")

    def _stacked(self) -> tuple:
        params = {name: getattr(self, self._mangle(name)) for name in self._param_names}
        buffers = {name: getattr(self, self._mangle(name)) for name in self._buffer_names}
        return params, buffers

    # ------------------------------------------------------------------ the call
    def forward(self, *inputs: torch.Tensor) -> Any:
        """Call every agent's copy on its own slice of ``inputs``.

        Each input is ``(num_agents, rows, ...)``; the output keeps that layout.
        """
        params, buffers = self._stacked()
        meta = self._meta

        def single(p, b, *args):
            return functional_call(meta, (p, b), args)

        call = single
        if self.members > 1:
            # every member of an agent sees that agent's inputs: (members, rows, ...) out
            call = torch.vmap(single, in_dims=(0, 0) + (None,) * len(inputs))
        with warnings.catch_warnings():
            # a for-loop fallback would silently serialize the agents
            warnings.filterwarnings("error", message=".*falling back.*")
            return torch.vmap(call)(params, buffers, *inputs)

    def forward_flat(self, *inputs: torch.Tensor) -> Any:
        """Same, for flat ``(num_agents * rows, ...)`` tensors in and out. With ``members >
        1`` the output is ``(members, num_agents * rows, ...)``: one flat batch per member."""
        rows = self._rows(inputs[0])
        shaped = [self.split(tensor, rows) for tensor in inputs]
        out = self.forward(*shaped)
        flatten = self.flatten if self.members == 1 else self._flatten_members
        if torch.is_tensor(out):
            return flatten(out)
        return type(out)(flatten(tensor) for tensor in out)

    @staticmethod
    def _flatten_members(batched: torch.Tensor) -> torch.Tensor:
        """``(num_agents, members, rows, ...) -> (members, num_agents * rows, ...)``."""
        agents, members, rows = batched.shape[:3]
        return batched.transpose(0, 1).reshape(members, agents * rows, *batched.shape[3:])

    def _rows(self, flat: torch.Tensor) -> int:
        if flat.shape[0] % self.num_agents != 0:
            raise ValueError(
                f"{flat.shape[0]} rows do not divide by num_agents={self.num_agents}: the batch "
                "must hold the same number of rows per agent, laid out [agent 0 | agent 1 | ...]"
            )
        return flat.shape[0] // self.num_agents

    def split(self, flat: torch.Tensor, rows: int) -> torch.Tensor:
        """``(num_agents * rows, ...) -> (num_agents, rows, ...)``."""
        return flat.view(self.num_agents, rows, *flat.shape[1:])

    @staticmethod
    def flatten(batched: torch.Tensor) -> torch.Tensor:
        """``(num_agents, rows, ...) -> (num_agents * rows, ...)``."""
        # the explicit product (not -1) keeps a zero-column tensor unambiguous, e.g. the
        # (num_agents, rows, 0) log_std of an all-Bernoulli policy
        return batched.reshape(batched.shape[0] * batched.shape[1], *batched.shape[2:])

    # ------------------------------------------------------------------ weight normalization
    @torch.no_grad()
    def normalize_weights(self) -> None:
        """FlashSAC's weight normalization (``sac.weight_normalization``), in place on the
        stacked parameters: every ``nn.Linear`` weight row (its input-feature vector) to unit
        norm, and every normalization layer's affine ``(scale, shift)`` jointly to norm
        ``sqrt(width)`` (LayerNorm / BatchNorm; RMSNorm's scale alone). Linear biases and
        anything else (a free log std) are untouched. Row-wise, so each agent and member
        keeps to its own slice. Device ops only: a CUDA graph can capture it."""
        params, _ = self._stacked()
        for name, module in self._meta.named_modules():
            prefix = f"{name}." if name else ""
            if isinstance(module, nn.Linear) and not getattr(module, "skip_weight_norm", False):
                weight = params[f"{prefix}weight"]
                weight.copy_(torch.nn.functional.normalize(weight, dim=-1, eps=1e-8))
            elif isinstance(module, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm, nn.RMSNorm)) or getattr(
                module, "normalize_affine", False
            ):
                scale = params.get(f"{prefix}weight")
                if scale is None:  # no affine parameters
                    continue
                shift = params.get(f"{prefix}bias")
                squares = scale * scale + (shift * shift if shift is not None else 0.0)
                factor = (scale.shape[-1] ** 0.5) * torch.rsqrt(squares.sum(dim=-1, keepdim=True) + 1e-8)
                scale.mul_(factor)
                if shift is not None:
                    shift.mul_(factor)

    # ------------------------------------------------------------------ per-agent state
    def agent_state_dict(self, agent: int) -> Dict[str, torch.Tensor]:
        """Agent ``agent``'s slice, keyed exactly like the plain module's state dict."""
        self._check_agent(agent)
        params, buffers = self._stacked()
        return {
            name: tensor[agent].detach().clone().cpu()
            for name, tensor in list(params.items()) + list(buffers.items())
        }

    @torch.no_grad()
    def load_agent_state_dict(self, agent: int, state: Dict[str, torch.Tensor]) -> None:
        """Write a plain module's state dict into slot ``agent``."""
        self._check_agent(agent)
        params, buffers = self._stacked()
        stacked = {**params, **buffers}
        extra = set(state) - set(stacked)
        missing = set(stacked) - set(state)
        if extra or missing:
            raise KeyError(
                f"state dict does not match the stacked module: unexpected {sorted(extra)}, "
                f"missing {sorted(missing)}"
            )
        for name, tensor in stacked.items():
            value = state[name]
            if value.shape != tensor.shape[1:]:
                raise ValueError(
                    f"'{name}' has shape {tuple(value.shape)}, expected "
                    f"{tuple(tensor.shape[1:])} for one agent"
                )
            tensor[agent].copy_(value.to(tensor.device))

    def _check_agent(self, agent: int) -> None:
        if not 0 <= agent < self.num_agents:
            raise ValueError(f"agent {agent} is out of range for num_agents={self.num_agents}")

    def __repr__(self) -> str:
        return (
            f"VmapEnsemble(num_agents={self.num_agents}, members={self.members}, "
            f"module={type(self._meta).__name__}, params={len(self._param_names)})"
        )

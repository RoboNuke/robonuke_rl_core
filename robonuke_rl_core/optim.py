"""One optimizer over stacked per-agent parameters.

AdamW's update is elementwise, so N independent AdamW instances are the same thing as one
update rule applied to parameters with a leading ``num_agents`` dimension — plus a per-agent
step count, learning rate and keep mask. No vmap: broadcasting does it.

The update mirrors ``torch.optim.AdamW``'s single-tensor path op for op (decoupled weight
decay first, ``eps`` added to the bias-corrected second moment outside the sqrt), and
``tests/optim/test_block_adamw.py`` holds it to ``torch.optim.AdamW`` as the reference.

Freezing: ``step(keep)`` computes the full update and writes it only where ``keep`` is true.
A frozen agent comes out bit-identical — weights, both moments, and its step count — so
neither momentum nor weight decay can move it while another agent trains.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

__all__ = ["BlockAdamW", "lr_at", "clip_grad_norm_per_agent"]


class BlockAdamW:
    """AdamW over parameters whose leading dimension is the agent."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        num_agents: int,
        lr: float,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("BlockAdamW got no parameters")
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        for index, param in enumerate(params):
            if param.dim() < 1 or param.shape[0] != num_agents:
                raise ValueError(
                    f"BlockAdamW parameter #{index} has shape {tuple(param.shape)}: every "
                    f"parameter needs a leading agent dimension of {num_agents}. A parameter "
                    "shared across agents would couple them."
                )
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        self.params = params
        self.num_agents = num_agents
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        device = params[0].device
        # the per-agent scalars are kept in float64 and cast at each op, which is what
        # torch.optim does with its Python-float lr and bias corrections
        self.lr = torch.full((num_agents,), float(lr), device=device, dtype=torch.float64)
        self.t = torch.zeros(num_agents, device=device, dtype=torch.int64)
        self.exp_avg = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]
        self.exp_avg_sq = [torch.zeros_like(p, memory_format=torch.preserve_format) for p in params]

    # ------------------------------------------------------------------ the step
    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self, keep: Optional[torch.Tensor] = None) -> None:
        """One AdamW step for every agent ``keep`` marks (all of them when ``keep`` is None)."""
        if keep is None:
            advance = torch.ones(self.num_agents, device=self.t.device, dtype=torch.int64)
            mask = None
        else:
            if keep.shape != (self.num_agents,) or keep.dtype != torch.bool:
                raise ValueError(
                    f"keep must be a bool tensor of shape ({self.num_agents},), got "
                    f"{keep.dtype} {tuple(keep.shape)}"
                )
            keep = keep.to(self.t.device)
            advance = keep.to(torch.int64)
            mask = keep

        # a frozen agent's step count does not advance; clamp so its (discarded) bias
        # correction cannot divide by zero
        step_count = (self.t + advance).clamp(min=1).to(torch.float64)
        bias_correction1 = 1.0 - self.beta1**step_count  # (N,)
        bias_correction2 = 1.0 - self.beta2**step_count
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = bias_correction2.sqrt()
        decay = 1.0 - self.lr * self.weight_decay

        for index, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                raise ValueError(
                    f"BlockAdamW parameter #{index} with shape {tuple(param.shape)} has no "
                    "gradient. Every parameter must take part in the loss; call backward "
                    "before step."
                )
            view = (-1,) + (1,) * (param.dim() - 1)  # broadcast the per-agent scalars
            dtype = param.dtype
            exp_avg = self.exp_avg[index]
            exp_avg_sq = self.exp_avg_sq[index]

            # torch.optim.AdamW, single-tensor path, op for op
            new_param = param * decay.to(dtype).view(view)
            new_exp_avg = exp_avg.lerp(grad, 1.0 - self.beta1)
            new_exp_avg_sq = torch.addcmul(
                exp_avg_sq * self.beta2, grad, grad, value=1.0 - self.beta2
            )
            denom = new_exp_avg_sq.sqrt() / bias_correction2_sqrt.to(dtype).view(view) + self.eps
            new_param = new_param - step_size.to(dtype).view(view) * new_exp_avg / denom

            if mask is None:
                param.copy_(new_param)
                exp_avg.copy_(new_exp_avg)
                exp_avg_sq.copy_(new_exp_avg_sq)
            else:
                keep_view = mask.view(view)
                param.copy_(torch.where(keep_view, new_param, param))
                exp_avg.copy_(torch.where(keep_view, new_exp_avg, exp_avg))
                exp_avg_sq.copy_(torch.where(keep_view, new_exp_avg_sq, exp_avg_sq))

        self.t += advance

    def set_lr(self, lr) -> None:
        """Set the learning rate: one value for every agent, or one per agent."""
        if torch.is_tensor(lr):
            if lr.shape != (self.num_agents,):
                raise ValueError(
                    f"set_lr expects a scalar or a tensor of shape ({self.num_agents},), got "
                    f"{tuple(lr.shape)}"
                )
            self.lr.copy_(lr.to(self.lr.device, self.lr.dtype))
        else:
            self.lr.fill_(float(lr))

    # ------------------------------------------------------------------ state
    def state_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t.detach().clone().cpu(),
            "lr": self.lr.detach().clone().cpu(),
            "exp_avg": [m.detach().clone().cpu() for m in self.exp_avg],
            "exp_avg_sq": [m.detach().clone().cpu() for m in self.exp_avg_sq],
            "hyper": {
                "num_agents": self.num_agents,
                "betas": (self.beta1, self.beta2),
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        missing = {"t", "lr", "exp_avg", "exp_avg_sq"} - set(state)
        if missing:
            raise KeyError(f"BlockAdamW state is missing keys: {sorted(missing)}")
        for key in ("exp_avg", "exp_avg_sq"):
            if len(state[key]) != len(self.params):
                raise ValueError(
                    f"BlockAdamW state has {len(state[key])} '{key}' tensors but the optimizer "
                    f"holds {len(self.params)} parameters"
                )
        self.t.copy_(state["t"].to(self.t.device))
        self.lr.copy_(state["lr"].to(self.lr.device))
        for index, param in enumerate(self.params):
            for key, moments in (("exp_avg", self.exp_avg), ("exp_avg_sq", self.exp_avg_sq)):
                saved = state[key][index]
                if saved.shape != param.shape:
                    raise ValueError(
                        f"BlockAdamW state '{key}'[{index}] has shape {tuple(saved.shape)}, "
                        f"expected {tuple(param.shape)}"
                    )
                moments[index].copy_(saved.to(param.device))

    def agent_state_dict(self, agent: int) -> Dict[str, Any]:
        """Agent ``agent``'s slice of the state: moments without the leading dim."""
        self._check_agent(agent)
        return {
            "t": int(self.t[agent]),
            "lr": float(self.lr[agent]),
            "exp_avg": [m[agent].detach().clone().cpu() for m in self.exp_avg],
            "exp_avg_sq": [m[agent].detach().clone().cpu() for m in self.exp_avg_sq],
        }

    @torch.no_grad()
    def load_agent_state_dict(self, agent: int, state: Dict[str, Any]) -> None:
        """Write one agent's slice back, leaving every other agent untouched."""
        self._check_agent(agent)
        missing = {"t", "lr", "exp_avg", "exp_avg_sq"} - set(state)
        if missing:
            raise KeyError(f"agent optimizer state is missing keys: {sorted(missing)}")
        for key in ("exp_avg", "exp_avg_sq"):
            if len(state[key]) != len(self.params):
                raise ValueError(
                    f"agent optimizer state has {len(state[key])} '{key}' tensors but the "
                    f"optimizer holds {len(self.params)} parameters"
                )
        self.t[agent] = int(state["t"])
        self.lr[agent] = float(state["lr"])
        for index, param in enumerate(self.params):
            for key, moments in (("exp_avg", self.exp_avg), ("exp_avg_sq", self.exp_avg_sq)):
                saved = state[key][index]
                if saved.shape != param.shape[1:]:
                    raise ValueError(
                        f"agent optimizer state '{key}'[{index}] has shape {tuple(saved.shape)}, "
                        f"expected {tuple(param.shape[1:])}"
                    )
                moments[index][agent].copy_(saved.to(param.device))

    def _check_agent(self, agent: int) -> None:
        if not 0 <= agent < self.num_agents:
            raise ValueError(f"agent {agent} is out of range for num_agents={self.num_agents}")

    def __repr__(self) -> str:
        return (
            f"BlockAdamW(num_agents={self.num_agents}, params={len(self.params)}, "
            f"lr={self.lr.tolist()}, weight_decay={self.weight_decay})"
        )


def lr_at(update: int, total_updates: int, lr: float, lr_end: float, schedule: str) -> float:
    """The learning rate for update ``update``, from the global update count only.

    ``constant`` returns ``lr``; ``cosine`` anneals ``lr -> lr_end`` over ``total_updates``,
    matching ``CosineAnnealingLR``'s closed form. The schedule never reads the data, so it
    cannot couple the agents.
    """
    if schedule == "constant":
        return float(lr)
    if schedule != "cosine":
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {schedule!r}")
    if total_updates < 1:
        raise ValueError(f"total_updates must be >= 1 for a cosine schedule, got {total_updates}")
    progress = min(max(update, 0), total_updates) / total_updates
    return float(lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * progress)))


def clip_grad_norm_per_agent(modules, num_agents: int, max_norm: float) -> torch.Tensor:
    """Clip each agent's gradients by its own norm; return the pre-clip norms ``(num_agents,)``.

    ``modules`` is one ``nn.Module`` or an iterable of them. Every parameter must carry a
    leading agent dimension — a shared parameter would make one agent's gradient scale depend
    on another agent's data, so it raises instead. ``max_norm <= 0`` only reports the norms.

    Vectorized over agents (no per-agent host sync), with torch's ``max_norm / (norm + 1e-6)``
    clip coefficient.
    """
    if isinstance(modules, nn.Module):
        modules = [modules]
    modules = list(modules)
    if not modules:
        raise ValueError("clip_grad_norm_per_agent got no modules")

    grads = []
    for module in modules:
        for name, param in module.named_parameters():
            if param.dim() < 1 or param.shape[0] != num_agents:
                raise ValueError(
                    f"{type(module).__name__}.{name} has shape {tuple(param.shape)}: every "
                    f"parameter needs a leading agent dimension of {num_agents}. A shared "
                    "parameter would couple the agents."
                )
            if param.grad is not None:
                grads.append(param.grad)

    device = next(iter(modules)).parameters().__next__().device
    if not grads:
        return torch.zeros(num_agents, device=device)

    # one (num_agents,) norm per parameter, then the per-agent total over parameters
    squares = torch.stack([g.reshape(num_agents, -1).pow(2).sum(dim=1) for g in grads])
    norms = squares.sum(dim=0).sqrt()
    if max_norm > 0:
        scale = (max_norm / (norms + 1e-6)).clamp(max=1.0)
        for grad in grads:
            grad.mul_(scale.view((-1,) + (1,) * (grad.dim() - 1)))
    return norms

"""Per-agent running input normalization.

``BlockRunningNorm`` is skrl's ``RunningStandardScaler`` with a leading ``num_agents``
dimension: agent ``i``'s rows update agent ``i``'s statistics and nothing else. Same
update rule (parallel-variance), same epsilon, same clip threshold, so one agent's stats
match what a separate ``RunningStandardScaler`` fed the same rows would hold.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BlockRunningNorm"]


class BlockRunningNorm(nn.Module):
    """Running mean/variance normalizer with one independent set of stats per agent.

    Input rows are laid out per agent along ``agent_axis``: ``(num_agents * rows, size)``
    (the block-parallel convention — ``[agent 0 rows, agent 1 rows, ...]``).
    """

    running_mean: torch.Tensor
    running_variance: torch.Tensor
    current_count: torch.Tensor

    def __init__(
        self,
        num_agents: int,
        size: int,
        agent_axis: int = 0,
        *,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if agent_axis != 0:
            raise ValueError(
                f"agent_axis must be 0: rows arrive as [agent 0 rows, agent 1 rows, ...]; got {agent_axis}"
            )
        self.num_agents = num_agents
        self.size = int(size)
        self.agent_axis = agent_axis
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.register_buffer("running_mean", torch.zeros(num_agents, self.size, dtype=torch.float64, device=device))
        self.register_buffer("running_variance", torch.ones(num_agents, self.size, dtype=torch.float64, device=device))
        self.register_buffer("current_count", torch.ones(num_agents, dtype=torch.float64, device=device))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """``(num_agents * rows, size) -> (num_agents, rows, size)``."""
        if x.shape[-1] != self.size:
            raise ValueError(
                f"BlockRunningNorm(size={self.size}) got input with last dim {x.shape[-1]}"
            )
        rows = x.shape[0] // self.num_agents
        if rows * self.num_agents != x.shape[0]:
            raise ValueError(
                f"input rows ({x.shape[0]}) must be divisible by num_agents ({self.num_agents})"
            )
        return x.view(self.num_agents, rows, self.size)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        """Parallel-variance update, per agent (skrl's ``_parallel_variance``)."""
        blocks = self._split(x).to(torch.float64)
        count = blocks.shape[1]
        mean = blocks.mean(dim=1)
        var = blocks.var(dim=1)  # unbiased, as in skrl
        delta = mean - self.running_mean
        total = self.current_count.unsqueeze(-1) + count
        m2 = (
            self.running_variance * self.current_count.unsqueeze(-1)
            + var * count
            + delta**2 * self.current_count.unsqueeze(-1) * count / total
        )
        self.running_mean = self.running_mean + delta * count / total
        self.running_variance = m2 / total
        self.current_count = self.current_count + count

    def __call__(self, x: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Normalize ``x`` (or scale it back when ``inverse``), updating stats when ``train``."""
        if train:
            self._update(x)
        rows = x.shape[0] // self.num_agents
        mean = self.running_mean.float().repeat_interleave(rows, dim=0)
        std = torch.sqrt(self.running_variance.float()).repeat_interleave(rows, dim=0)
        if inverse:
            return std * torch.clamp(x, -self.clip_threshold, self.clip_threshold) + mean
        return torch.clamp((x - mean) / (std + self.epsilon), -self.clip_threshold, self.clip_threshold)

    def forward(self, x: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        return self.__call__(x, train=train, inverse=inverse)

    def state_dict_for(self, agent: int) -> dict:
        """Agent ``agent``'s stats, shaped like a single-agent normalizer's state."""
        return {
            "running_mean": self.running_mean[agent].detach().clone().cpu(),
            "running_variance": self.running_variance[agent].detach().clone().cpu(),
            "current_count": self.current_count[agent].detach().clone().cpu(),
        }

    @torch.no_grad()
    def load_state_dict_into(self, agent: int, state: dict) -> None:
        """Write a single-agent state back into slot ``agent``."""
        missing = {"running_mean", "running_variance", "current_count"} - set(state)
        if missing:
            raise KeyError(f"normalizer state is missing keys: {sorted(missing)}")
        self.running_mean[agent].copy_(state["running_mean"].to(self.running_mean.device))
        self.running_variance[agent].copy_(state["running_variance"].to(self.running_variance.device))
        self.current_count[agent].copy_(state["current_count"].to(self.current_count.device))

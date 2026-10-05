"""Memory configuration: the `memory` section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryCfg"]


@dataclass
class MemoryCfg:
    """The SAC replay buffer.

    ``memory_size`` is the capacity in transitions **per agent**, used exactly as given: the
    buffer is one ``(num_agents, memory_size, dim)`` tensor, so any positive integer works.
    PPO's rollout buffer is sized by ``ppo.rollouts`` instead (``rollouts`` env-steps per env,
    i.e. ``rollouts * envs_per_agent`` transitions per agent); it is not duplicated here.
    """

    memory_size: int = 1_000_000

    def validate(self, cfg: Any) -> None:
        if self.memory_size < 1:
            raise ValueError(f"memory.memory_size must be >= 1, got {self.memory_size}")

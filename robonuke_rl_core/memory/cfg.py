"""Memory configuration: the `memory` section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryCfg"]


@dataclass
class MemoryCfg:
    """The SAC replay buffer.

    ``memory_size`` is transitions **per agent**: skrl allocates one
    ``(memory_size // num_envs, num_envs, *)`` tensor, so every agent's env partition holds
    that depth. PPO's rollout buffer is sized by ``ppo.rollouts`` instead; it is not
    duplicated here.
    """

    memory_size: int = 1_000_000

    def validate(self, cfg: Any) -> None:
        if self.memory_size < 1:
            raise ValueError(f"memory.memory_size must be >= 1, got {self.memory_size}")

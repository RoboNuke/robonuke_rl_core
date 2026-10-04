"""Memory configuration: the `memory` section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryCfg"]


@dataclass
class MemoryCfg:
    """Replay / rollout buffer.

    ``size`` is transitions **per agent**: skrl allocates one
    ``(size // num_envs, num_envs, *)`` tensor, so every agent's partition holds that depth.
    PPO ignores it and uses ``ppo.rollouts`` as the per-env depth instead.
    """

    size: int = 1_000_000
    replacement: bool = True

    def validate(self, cfg: Any) -> None:
        if self.size < 1:
            raise ValueError(f"memory.size must be >= 1, got {self.size}")

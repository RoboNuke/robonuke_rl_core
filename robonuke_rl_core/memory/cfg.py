"""Memory configuration: the `memory` section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MemoryCfg", "replay_depth"]


@dataclass
class MemoryCfg:
    """The SAC replay buffer.

    ``memory_size`` is transitions **per agent**. skrl allocates one
    ``(depth, num_envs, *)`` tensor; with ``depth = memory_size // envs_per_agent`` (see
    :func:`replay_depth`) each agent's envs hold exactly ``memory_size`` transitions. PPO's
    rollout buffer is sized by ``ppo.rollouts`` instead; it is not duplicated here.
    """

    memory_size: int = 1_000_000

    def validate(self, cfg: Any) -> None:
        if self.memory_size < 1:
            raise ValueError(f"memory.memory_size must be >= 1, got {self.memory_size}")


def replay_depth(memory_size: int, envs_per_agent: int) -> int:
    """Per-env buffer depth that gives each agent exactly ``memory_size`` transitions.

    Raises when ``memory_size`` is not a multiple of ``envs_per_agent``, so the capacity a run
    uses is always the one its config states.
    """
    if envs_per_agent < 1:
        raise ValueError(f"envs_per_agent must be >= 1, got {envs_per_agent}")
    if memory_size < envs_per_agent or memory_size % envs_per_agent:
        lower = max(envs_per_agent, memory_size - memory_size % envs_per_agent)
        upper = lower + envs_per_agent if memory_size > lower else lower
        raise ValueError(
            f"memory.memory_size ({memory_size}) must be a multiple of the envs per agent "
            f"({envs_per_agent}) so each agent holds exactly memory_size transitions; nearest "
            f"valid sizes: {lower} or {upper}"
        )
    return memory_size // envs_per_agent

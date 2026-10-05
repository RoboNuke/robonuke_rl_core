"""The Forge boundary.

Forge- and Factory-style Direct envs expose internals that other Isaac Lab envs simply do
not have: a smoothed wrist force/torque signal, an operational-space ``cfg.ctrl`` block,
fingertip state on the env object, and ``_reset_idx`` semantics a wrapper can lean on.
Everything in this package that reads those lives under ``envs/forge/``, carries a ``Forge``
class-name prefix, and calls :func:`require_forge_env` before it touches anything.

**The rule is the task name: a Forge env is one whose task name contains "forge",
case-insensitively.** Not duck typing, not a capability probe — a name check, made once,
up front. A wrapper applied to another env family then fails immediately and says so, rather
than dying on a missing attribute halfway through a rollout, and a wrapper for a different
family starts a sibling directory instead of becoming a special case in here.

Each wrapper still documents the env attributes it reads; that list is its interface
contract, and it appears in the error so the next family's port knows what it must provide.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["FORGE_MARKER", "is_forge_task", "require_forge_env"]

#: a task whose name contains this (case-insensitive) is a Forge-family env
FORGE_MARKER = "forge"


def is_forge_task(task_name: Any) -> bool:
    """Whether ``task_name`` names a Forge-family task."""
    return FORGE_MARKER in str(task_name).lower()


def require_forge_env(task_name: Any, wrapper: str, reads: Sequence[str] = ()) -> None:
    """Raise unless ``task_name`` is a Forge-family task.

    :param wrapper: the wrapper's class name, for the message.
    :param reads: the env attributes the wrapper reads — its interface contract.
    """
    if is_forge_task(task_name):
        return
    contract = ", ".join(reads) if reads else "Forge-specific env internals"
    raise ValueError(
        f"{wrapper} only works on a Forge-family task, and {task_name!r} is not one: the rule "
        f"is that the task name contains {FORGE_MARKER!r} (case-insensitive). It reads "
        f"{contract}, which other Isaac Lab envs do not provide. A wrapper for another env "
        "family belongs in its own package beside robonuke_rl_core/envs/forge/, not in it."
    )

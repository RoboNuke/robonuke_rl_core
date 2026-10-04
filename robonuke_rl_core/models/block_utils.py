"""Per-agent slicing of block-parallel state dicts, optimizer state, and gradients.

A block-parallel parameter carries a leading ``num_agents`` dimension, so agent ``i``'s
weights are ``param[i]``. These helpers cut one agent out of such a module (for a
per-agent checkpoint), write one agent back into a slot, and clip gradients per agent.

Ported from RoboNuke/generalized_hybrid_vic_action_space ``models/block_simba.py``;
``clip_grad_norm_per_agent`` is the per-agent form of the clipping in
RoboNuke/Continuous_Force_RL ``agents/block_ppo.py``.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "slice_block_state_dict",
    "assign_block_slice",
    "slice_optimizer_state",
    "merge_optimizer_states",
    "clip_grad_norm_per_agent",
]


def _is_block_tensor(t, num_agents: int) -> bool:
    """A tensor is block-parallel if its leading dim equals ``num_agents``."""
    return torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] == num_agents


def _per_agent_paramlist_prefixes(block_module: nn.Module, num_agents: int) -> list[str]:
    """Dotted prefixes of ``nn.ParameterList`` children holding one entry per agent."""
    return [
        name
        for name, mod in block_module.named_modules()
        if isinstance(mod, nn.ParameterList) and len(mod) == num_agents
    ]


def slice_block_state_dict(block_module: nn.Module, agent_idx: int, num_agents: int) -> dict:
    """Return a state dict with every block tensor sliced to ``agent_idx``.

    Output tensors lose the leading dim (``(N, out, in)`` -> ``(out, in)``). Per-agent
    ``ParameterList`` entries keep only ``agent_idx``, renumbered to ``0``, so the result
    is shaped like a single-agent module's state dict.
    """
    prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    sliced = {}
    for name, param in block_module.state_dict().items():
        prefix = next((p for p in prefixes if name.startswith(p + ".")), None)
        if prefix is not None:
            head, _, rest = name[len(prefix) + 1 :].partition(".")
            if not head.isdigit():
                raise ValueError(
                    f"expected an integer index after ParameterList prefix '{prefix}.' in "
                    f"state_dict key '{name}', got '{head}'"
                )
            if int(head) != agent_idx:
                continue  # another agent's entry
            name = f"{prefix}.0" + (f".{rest}" if rest else "")
            sliced[name] = param.detach().clone().cpu()
            continue
        if _is_block_tensor(param, num_agents):
            sliced[name] = param[agent_idx].detach().clone().cpu()
        else:
            sliced[name] = param.detach().clone().cpu() if torch.is_tensor(param) else param
    return sliced


def assign_block_slice(
    block_module: nn.Module, agent_idx: int, num_agents: int, agent_state_dict: dict
) -> None:
    """Write a single-agent state dict into ``block_module``'s slot ``agent_idx``."""
    prefixes = _per_agent_paramlist_prefixes(block_module, num_agents)
    block_state = block_module.state_dict()

    remapped = {}
    for name, value in agent_state_dict.items():
        prefix = next((p for p in prefixes if name.startswith(p + ".")), None)
        if prefix is not None:
            head, _, rest = name[len(prefix) + 1 :].partition(".")
            if head == "0":
                name = f"{prefix}.{agent_idx}" + (f".{rest}" if rest else "")
        remapped[name] = value

    extra = set(remapped) - set(block_state)
    if extra:
        raise KeyError(f"unexpected keys in the single-agent state_dict: {sorted(extra)}")

    paramlist_keys = {k for k in block_state if any(k.startswith(p + ".") for p in prefixes)}
    with torch.no_grad():
        for name, agent_param in remapped.items():
            block_param = block_state[name]
            if not torch.is_tensor(block_param):
                continue
            if name in paramlist_keys or not _is_block_tensor(block_param, num_agents):
                block_param.copy_(agent_param.to(block_param.device))
            else:
                block_param[agent_idx].copy_(agent_param.to(block_param.device))


def slice_optimizer_state(opt_state_dict: dict, agent_idx: int, num_agents: int) -> dict:
    """Slice every block tensor of an optimizer state dict to ``agent_idx``.

    ``_sliced_keys`` records which ``(param_id, key)`` pairs were sliced so
    :func:`merge_optimizer_states` can restack them without guessing.
    """
    for key in ("state", "param_groups"):
        if key not in opt_state_dict:
            raise KeyError(f"optimizer state_dict is missing required key '{key}'")

    out_state, sliced_keys = {}, set()
    for param_id, param_state in opt_state_dict["state"].items():
        new_state = {}
        for key, value in param_state.items():
            if _is_block_tensor(value, num_agents):
                new_state[key] = value[agent_idx].detach().clone().cpu()
                sliced_keys.add((param_id, key))
            elif torch.is_tensor(value):
                new_state[key] = value.detach().clone().cpu()
            else:
                new_state[key] = value
        out_state[param_id] = new_state
    return {
        "state": out_state,
        "param_groups": opt_state_dict["param_groups"],
        "_sliced_keys": sliced_keys,
    }


def merge_optimizer_states(per_agent_state_dicts: list, num_agents: int) -> dict:
    """Stack per-agent optimizer state dicts back into a block-shaped state dict."""
    if len(per_agent_state_dicts) != num_agents:
        raise ValueError(
            f"expected {num_agents} per-agent optimizer state_dicts, got {len(per_agent_state_dicts)}"
        )
    first = per_agent_state_dicts[0]
    if "_sliced_keys" not in first:
        raise KeyError(
            "per-agent optimizer state_dict is missing the '_sliced_keys' sidecar; produce the "
            "slice with slice_optimizer_state()"
        )
    for key in ("state", "param_groups"):
        if key not in first:
            raise KeyError(f"per-agent optimizer state_dict is missing required key '{key}'")

    sliced_keys = first["_sliced_keys"]
    out_state = {}
    for param_id, param_state in first["state"].items():
        new_state = {}
        for key, value in param_state.items():
            if (param_id, key) in sliced_keys:
                new_state[key] = torch.stack(
                    [per_agent_state_dicts[i]["state"][param_id][key] for i in range(num_agents)],
                    dim=0,
                )
            else:
                new_state[key] = value
        out_state[param_id] = new_state
    return {"state": out_state, "param_groups": first["param_groups"]}


def _agent_of(name: str, prefixes: List[str]) -> int | None:
    """Agent index of a per-agent ``ParameterList`` entry, from its state-dict name."""
    for prefix in prefixes:
        if name.startswith(prefix + "."):
            head = name[len(prefix) + 1 :].partition(".")[0]
            if head.isdigit():
                return int(head)
    return None


def clip_grad_norm_per_agent(modules, num_agents: int, max_norm: float) -> torch.Tensor:
    """Clip each agent's gradients by its own norm; return the pre-clip norms ``(num_agents,)``.

    ``modules`` is one ``nn.Module`` or an iterable of them. Every parameter must be either a
    block parameter (leading dim ``num_agents``) or an entry of an ``nn.ParameterList`` with one
    entry per agent. A shared parameter would make one agent's gradient scale depend on another
    agent's data, so it raises instead.

    Ownership is read from the module tree, not from attributes on the parameters: a
    ``copy.deepcopy`` of a module drops custom tensor attributes.
    """
    if isinstance(modules, nn.Module):
        modules = [modules]
    modules = list(modules)
    if not modules:
        raise ValueError("clip_grad_norm_per_agent got no modules")

    block: List[torch.nn.Parameter] = []
    per_agent: List[Tuple[int, torch.nn.Parameter]] = []
    device = None
    for module in modules:
        prefixes = _per_agent_paramlist_prefixes(module, num_agents)
        for name, param in module.named_parameters():
            device = param.device if device is None else device
            if param.dim() >= 1 and param.shape[0] == num_agents:
                block.append(param)
                continue
            agent = _agent_of(name, prefixes)
            if agent is None:
                raise ValueError(
                    f"{type(module).__name__}.{name} with shape {tuple(param.shape)} is neither a "
                    f"block parameter (leading dim == num_agents == {num_agents}) nor an entry of "
                    "a per-agent nn.ParameterList. A shared parameter would couple the agents."
                )
            per_agent.append((agent, param))

    norms = torch.zeros(num_agents, device=device)
    for agent in range(num_agents):
        squares = [p.grad[agent].pow(2).sum() for p in block if p.grad is not None]
        squares += [
            p.grad.pow(2).sum() for owner, p in per_agent if owner == agent and p.grad is not None
        ]
        if squares:
            norms[agent] = torch.sqrt(torch.stack(squares).sum())

    if max_norm > 0:
        for agent in range(num_agents):
            if norms[agent] > max_norm:
                scale = max_norm / norms[agent]
                for p in block:
                    if p.grad is not None:
                        p.grad[agent].mul_(scale)
                for owner, p in per_agent:
                    if owner == agent and p.grad is not None:
                        p.grad.mul_(scale)
    return norms

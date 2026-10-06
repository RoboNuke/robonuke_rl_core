"""Reading config *files*: the ``base`` chain, one layer per file, and the CLI layer.

Split out of :mod:`robonuke_rl_core.config` for one reason: the HPC submitters run on a
login node with no Isaac Lab, no GPU and no torch, and they need to read an experiment's
chain to find its ``hpc`` section. ``config.py`` cannot be imported there — it registers
the `eval` section, which pulls ``evaluation.py``, which imports torch. So the file-reading
half lives here, where nothing heavier than OmegaConf is needed, and both
``config.load_config`` and ``hpc.submit.read_submit_config`` call the **same** functions.
One implementation of "what does `base` mean", not two that can drift.

Nothing here builds a :class:`~robonuke_rl_core.config.Config`: no section schemas, no task
cfg, no validation. It stops at ordered, interpolation-checked layers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Tuple

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "OVERRIDE_RE",
    "chain",
    "file_layer",
    "load_file_chain",
    "cli_layer",
    "reject_interpolation",
]

#: a CLI override is ``a.b.c=value``; anything else belongs on the parser
OVERRIDE_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*=.*$", re.DOTALL)


def chain(path: str | Path) -> List[Path]:
    """Follow ``base`` from ``path``; return the files most-base first.

    Each ``base`` is relative to the file that names it. A missing file or a cycle raises,
    printing the chain followed so far — a chain is only debuggable if the error shows it.
    """
    start = Path(path).expanduser()
    if not start.is_file():
        raise FileNotFoundError(f"config file not found: {start}")
    followed: List[Path] = []
    current = start.resolve()
    while True:
        if current in followed:
            shown = " -> ".join(str(p) for p in followed + [current])
            raise ValueError(f"'base' chain is a cycle: {shown}")
        followed.append(current)
        base = OmegaConf.load(current).get("base")
        if base is None:
            break
        nxt = Path(str(base)).expanduser()
        nxt = (nxt if nxt.is_absolute() else current.parent / nxt).resolve()
        if not nxt.is_file():
            shown = " -> ".join(str(p) for p in followed)
            raise FileNotFoundError(
                f"base config file not found: {nxt}\n  named by 'base: {base}' in {current}\n"
                f"  'base' is relative to the file that names it\n  chain so far: {shown}"
            )
        current = nxt
    return list(reversed(followed))


def file_layer(path: Path) -> DictConfig:
    """One config file as a layer, without its ``base`` key."""
    layer = OmegaConf.load(path)
    if not isinstance(layer, DictConfig):
        raise ValueError(f"{path}: the top level of a config file must be a mapping")
    layer.pop("base", None)
    return layer


def load_file_chain(path: str | Path) -> List[Tuple[str, DictConfig]]:
    """``(name, layer)`` for every file in ``path``'s chain, most-base first.

    Interpolation is rejected here, so a layer that reaches either caller is already clean.
    """
    layers = [(str(p), file_layer(p)) for p in chain(path)]
    for where, layer in layers:
        reject_interpolation(OmegaConf.to_container(layer, resolve=False), where)
    return layers


def cli_layer(overrides: Any) -> DictConfig:
    """Leftover CLI args as the last layer."""
    overrides = list(overrides or [])
    for arg in overrides:
        if not isinstance(arg, str) or not OVERRIDE_RE.match(arg):
            raise ValueError(
                f"unrecognized argument {arg!r}: a config override must look like "
                "'section.field=value' (e.g. task.cfg.scene.num_envs=128); declare every other "
                "argument on the parser"
            )
    return OmegaConf.from_dotlist(overrides)


def reject_interpolation(data: Any, where: str, path: str = "") -> None:
    """Configs never use OmegaConf interpolation; ``to_object`` would resolve it silently."""
    if isinstance(data, dict):
        for key, sub in data.items():
            reject_interpolation(sub, where, f"{path}.{key}" if path else str(key))
    elif isinstance(data, list):
        for index, sub in enumerate(data):
            reject_interpolation(sub, where, f"{path}[{index}]")
    elif isinstance(data, str) and "${" in data:
        raise ValueError(
            f"{where} uses OmegaConf interpolation at '{path}': {data!r}. This project does "
            "not use interpolation in configs; write the value out."
        )

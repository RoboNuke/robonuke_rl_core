"""Config manager: one resolved config per run.

Load order: class defaults and task defaults -> most-base file -> ... -> passed file -> CLI.

OmegaConf does the merging, type checks, required-field checks, CLI parsing and YAML output.
There is no interpolation (``${...}``) in our configs. This module imports without Isaac Lab:
Isaac Lab is imported only inside :func:`load_task_cfg`, :func:`task_cfg_to_dict` and
:func:`apply_task_cfg`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

#: keys that may never be a section name: ``base`` steers the file chain, ``task`` is built
#: separately from the env cfg, ``meta`` and ``derived`` are written by the pipeline
RESERVED = ("base", "task", "meta", "derived")
RESOLVED_NAME = "resolved_config.yaml"
OVERRIDE_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*=.*$", re.DOTALL)


# ----------------------------------------------------------------------------- sections
@dataclass
class ExperimentCfg:
    """One seed per run: every agent shares one Isaac Sim instance and a subset of its envs."""

    num_agents: int = 1
    seed: int = MISSING

    def validate(self, cfg: Config) -> None:
        if self.num_agents < 1:
            raise ValueError(f"experiment.num_agents must be >= 1, got {self.num_agents}")


@dataclass
class WandbCfg:
    """Weights & Biases destination. Run names derive from ``group`` (see ``derived``)."""

    entity: str = MISSING
    project: str = MISSING
    group: str = MISSING
    tags: List[str] = field(default_factory=list)

    def validate(self, cfg: Config) -> None:
        if any(char.isspace() for char in self.group) or not self.group:
            raise ValueError(
                f"wandb.group must be non-empty and free of whitespace, got {self.group!r}: run "
                "names derive from it"
            )


SECTIONS: Dict[str, type] = {}


def register_section(name: str, cls: type) -> None:
    """Register ``cls`` as the top-level section ``name``."""
    if name in RESERVED:
        raise ValueError(f"'{name}' is reserved and cannot be a section name; reserved: {RESERVED}")
    if name in SECTIONS:
        taken = SECTIONS[name]
        raise ValueError(
            f"section '{name}' is already registered as {taken.__module__}:{taken.__qualname__}"
        )
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"section '{name}': {cls!r} is not a @dataclass")
    SECTIONS[name] = cls


def resolve_callable(ref: str) -> Any:
    """Resolve a ``"module:name"`` string to the function or class it names."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"{ref!r} is not a \"module:name\" reference, e.g. 'torch.nn:ReLU'")
    try:
        target = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import module '{module_name}' of {ref!r}: {exc}") from exc
    for part in attr.split("."):
        if not hasattr(target, part):
            raise ValueError(f"{ref!r}: '{module_name}' has no attribute '{part}'")
        target = getattr(target, part)
    return target


# ---------------------------------------------------------------------------- task cfg
def load_task_cfg(name: str) -> Any:
    """The task's default env cfg object, from the Isaac Lab gym registry."""
    try:
        import isaaclab_tasks  # noqa: F401  (registers the tasks)
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    except ImportError as exc:
        raise RuntimeError(
            f"Isaac Lab is not importable, so task '{name}' cannot be loaded ({exc}). Start the "
            "sim app (AppLauncher) on a machine with Isaac Lab first."
        ) from exc
    return load_cfg_from_registry(name, "env_cfg_entry_point")


def task_cfg_to_dict(env_cfg: Any) -> dict:
    """Env cfg object -> plain dict, the way Isaac Lab's hydra integration does it."""
    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings
    from isaaclab.utils import replace_slices_with_strings

    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    return replace_slices_with_strings(env_cfg.to_dict())


def apply_task_cfg(env_cfg: Any, data: dict) -> Any:
    """Plain dict -> env cfg object, the reverse of :func:`task_cfg_to_dict`."""
    from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_strings_with_slices

    env_cfg.from_dict(replace_strings_with_slices(data))
    return replace_strings_with_env_cfg_spaces(env_cfg)


# ------------------------------------------------------------------------------- result
@dataclass
class Config:
    """One run's resolved config. Nothing may change it after :func:`load_config` returns."""

    task_name: str
    task_cfg: Any
    sections: Dict[str, Any]
    derived: Dict[str, Any]
    meta: Dict[str, Any]
    root: DictConfig

    def __getattr__(self, name: str) -> Any:
        sections = self.__dict__.get("sections", {})
        if name in sections:
            return sections[name]
        raise AttributeError(f"no attribute or section '{name}'; sections: {list(sections)}")

    def __getitem__(self, name: str) -> Any:
        return self.sections[name]


# ----------------------------------------------------------------------------- the chain
def _chain(path: str | Path) -> List[Path]:
    """Follow ``base`` from ``path``; return the files most-base first."""
    start = Path(path).expanduser()
    if not start.is_file():
        raise FileNotFoundError(f"config file not found: {start}")
    chain: List[Path] = []
    current = start.resolve()
    while True:
        if current in chain:
            shown = " -> ".join(str(p) for p in chain + [current])
            raise ValueError(f"'base' chain is a cycle: {shown}")
        chain.append(current)
        base = OmegaConf.load(current).get("base")
        if base is None:
            break
        nxt = Path(str(base)).expanduser()
        nxt = (nxt if nxt.is_absolute() else current.parent / nxt).resolve()
        if not nxt.is_file():
            shown = " -> ".join(str(p) for p in chain)
            raise FileNotFoundError(
                f"base config file not found: {nxt}\n  named by 'base: {base}' in {current}\n"
                f"  'base' is relative to the file that names it\n  chain so far: {shown}"
            )
        current = nxt
    return list(reversed(chain))


def _file_layer(path: Path) -> DictConfig:
    """One config file as a layer, without its ``base`` key."""
    layer = OmegaConf.load(path)
    if not isinstance(layer, DictConfig):
        raise ValueError(f"{path}: the top level of a config file must be a mapping")
    layer.pop("base", None)
    return layer


def _cli_layer(overrides: Any) -> DictConfig:
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


# ---------------------------------------------------------------------------- the pipeline
def load_config(path: str | Path, overrides: Any = None) -> Config:
    """Resolve a config file (plus its ``base`` chain and CLI overrides) into one config."""
    layers = [(str(p), _file_layer(p)) for p in _chain(path)]
    layers.append(("CLI", _cli_layer(overrides)))
    return _build(layers)


def load_from_run(run_dir: str | Path, overrides: Any = None) -> Config:
    """Use a past run's ``resolved_config.yaml`` as the only file layer, then CLI overrides."""
    path = Path(run_dir).expanduser()
    if path.is_dir():
        path = path / RESOLVED_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no resolved config at {path}")
    layer = _file_layer(path)
    for computed in ("meta", "derived"):
        layer.pop(computed, None)
    return _build([(str(path), layer), ("CLI", _cli_layer(overrides))])


def _build(layers: List[tuple]) -> Config:
    # 3. the task name: the last layer that sets it wins
    task_name = None
    for _, layer in layers:
        name = OmegaConf.select(layer, "task.name")
        if name is not None:
            task_name = name
    if task_name is None:
        raise ValueError("no layer sets task.name; set it in a config file or with task.name=<id>")

    # 4. the schema: sections from their dataclasses, task from the env cfg defaults
    env_cfg = load_task_cfg(task_name)
    task_defaults = task_cfg_to_dict(env_cfg)
    root = OmegaConf.create({"task": {"name": MISSING, "cfg": task_defaults}})
    for name, cls in SECTIONS.items():
        root[name] = OmegaConf.structured(cls)
    OmegaConf.set_struct(root, True)

    # 5. merge every layer in order; an error names the layer it came from
    for where, layer in layers:
        try:
            root = OmegaConf.merge(root, layer)
        except OmegaConfBaseException as exc:
            raise ValueError(f"config error from {where}: {exc}") from exc

    # 6. required fields
    missing = OmegaConf.missing_keys(root)
    if missing:
        raise ValueError(
            "these required config fields are not set by any layer: "
            + ", ".join(sorted(missing))
        )

    # 7. the task dict is not type-checked by OmegaConf, so check it against the defaults
    _check_task_types(OmegaConf.to_container(root.task.cfg), task_defaults)

    # 8. the objects
    sections = {name: OmegaConf.to_object(root[name]) for name in SECTIONS}
    task_cfg = apply_task_cfg(env_cfg, OmegaConf.to_container(root.task.cfg))

    # 9. derived values
    derived = {
        "run_names": [
            f"{sections['wandb'].group}_a{i}" for i in range(sections["experiment"].num_agents)
        ]
    }

    cfg = Config(task_name, task_cfg, sections, derived, _meta(), root)

    # 10. cross-field rules
    for name, obj in sections.items():
        hook = getattr(obj, "validate", None)
        if callable(hook):
            hook(cfg)
    return cfg


def _check_task_types(merged: Any, default: Any, path: str = "task.cfg") -> None:
    """Compare every task leaf with the type of its default. int is fine for float."""
    if isinstance(default, dict):
        for key, sub in default.items():
            _check_task_types(merged[key], sub, f"{path}.{key}")
        return
    if default is None:
        return
    if isinstance(default, bool):
        ok = isinstance(merged, bool)
    elif isinstance(default, int):
        ok = isinstance(merged, int) and not isinstance(merged, bool)
    elif isinstance(default, float):
        ok = isinstance(merged, float) or (isinstance(merged, int) and not isinstance(merged, bool))
    elif isinstance(default, (list, tuple)):
        ok = isinstance(merged, (list, tuple))  # OmegaConf has no tuple: a tuple default reads back as a list
    else:
        ok = isinstance(merged, type(default))
    if not ok:
        raise TypeError(
            f"{path}: expected {type(default).__name__}, got {type(merged).__name__} {merged!r}"
            + (". Write floats as 1.0e-4: YAML reads 1e-4 as a string" if isinstance(default, float) else "")
        )


# ------------------------------------------------------------------------------ dump / cli
def dump(cfg: Config, path: str | Path) -> Path:
    """Write ``resolved_config.yaml``: meta, derived, then the whole merged config."""
    out = Path(path)
    if out.suffix not in (".yaml", ".yml"):
        out = out / RESOLVED_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = OmegaConf.create({"meta": cfg.meta, "derived": cfg.derived})
    doc = OmegaConf.merge(doc, cfg.root)
    out.write_text(OmegaConf.to_yaml(doc))
    return out


def _git(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _meta() -> Dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    commit = _git(package_dir)
    if commit is None:
        raise RuntimeError(
            f"cannot read the git commit of the package at {package_dir}: it is not a git checkout "
            "with a commit. A run must record the code that produced it."
        )
    return {
        "pkg_commit": commit,
        "project_commit": _git(Path.cwd()),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--config`` and ``--from_run``; exactly one of them is required."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, default=None, help="experiment config YAML file")
    group.add_argument("--from_run", type=str, default=None, help="past run dir to start from")
    return parser


def load_from_args(args: argparse.Namespace, overrides: Any = None) -> Config:
    """Dispatch on the args added by :func:`add_config_args`."""
    if bool(args.config) == bool(args.from_run):
        raise ValueError("pass exactly one of --config <file> or --from_run <run_dir>")
    if args.config:
        return load_config(args.config, overrides)
    return load_from_run(args.from_run, overrides)


# package sections are registered here, at the bottom of this file
register_section("experiment", ExperimentCfg)
register_section("wandb", WandbCfg)

"""Config manager: one resolved config per run.

Load order: class defaults and task defaults -> most-base file -> ... -> passed file -> CLI.

OmegaConf does the merging, type checks, required-field checks, CLI parsing and YAML output.
There is no interpolation (``${...}``) in our configs. This module imports without Isaac Lab:
Isaac Lab is imported only inside :func:`load_task_cfg`, :func:`task_cfg_to_dict` and
:func:`apply_task_cfg`.
"""

from __future__ import annotations

import argparse
import copy
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

from .configfile import (
    OVERRIDE_RE,
    chain,
    cli_layer,
    file_layer,
    load_file_chain,
    reject_interpolation,
)

#: keys that may never be a section name: ``base`` steers the file chain, ``task`` is built
#: separately from the env cfg, ``meta`` and ``derived`` are written by the pipeline
RESERVED = ("base", "task", "meta", "derived")
RESOLVED_NAME = "resolved_config.yaml"


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
    #: "online", "offline" or "disabled" (what wandb.init's mode accepts)
    mode: str = "online"

    def validate(self, cfg: Config) -> None:
        from .logging import MODES

        if self.mode not in MODES:
            raise ValueError(f"wandb.mode must be one of {MODES}, got {self.mode!r}")
        if not self.group or any(char.isspace() or char == "/" for char in self.group):
            raise ValueError(
                f"wandb.group must be non-empty and free of whitespace and '/', got {self.group!r}: "
                "run names and run directories derive from it"
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
    """Env cfg object -> plain dict, the way Isaac Lab's hydra integration does it.

    Works on a deep copy: Isaac Lab's space conversion edits the object in place, and this is
    also called on the live env's cfg (see :func:`dump`).
    """
    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings
    from isaaclab.utils import replace_slices_with_strings

    env_cfg = replace_env_cfg_spaces_with_strings(copy.deepcopy(env_cfg))
    return replace_slices_with_strings(env_cfg.to_dict())


def apply_task_cfg(env_cfg: Any, data: dict) -> Any:
    """Plain dict -> env cfg object, the reverse of :func:`task_cfg_to_dict`.

    The object's spaces are serialized to strings first: ``data`` holds them in that form
    (``task_cfg_to_dict`` serialized them), and Isaac Lab's ``from_dict`` refuses a value whose
    type differs from the attribute's current one.
    """
    from isaaclab.envs.utils.spaces import (
        replace_env_cfg_spaces_with_strings,
        replace_strings_with_env_cfg_spaces,
    )
    from isaaclab.utils import replace_strings_with_slices

    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
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
    #: every ``task.cfg`` path a file or the CLI set -> the layer that set it last, plus
    #: ``seed`` (set from ``experiment.seed``). :func:`check_env_kept_overrides` checks these.
    task_overrides: Dict[str, str]

    def __getattr__(self, name: str) -> Any:
        sections = self.__dict__.get("sections", {})
        if name in sections:
            return sections[name]
        raise AttributeError(f"no attribute or section '{name}'; sections: {list(sections)}")

    def __getitem__(self, name: str) -> Any:
        return self.sections[name]


# ----------------------------------------------------------------------------- the chain
# Reading files -- the `base` chain, one layer per file, the CLI layer -- lives in
# `configfile.py`, which imports nothing heavier than OmegaConf. The HPC submitters run on a
# login node with no torch and need the same chain reader; this module cannot be imported
# there (registering the `eval` section pulls evaluation.py, which imports torch). These
# aliases keep the private names this module has always used.
_chain = chain
_file_layer = file_layer
_cli_layer = cli_layer
_reject_interpolation = reject_interpolation


# ---------------------------------------------------------------------------- the pipeline
def load_config(path: str | Path, overrides: Any = None) -> Config:
    """Resolve a config file (plus its ``base`` chain and CLI overrides) into one config."""
    layers = load_file_chain(path)
    layers.append(("CLI", _cli_layer(overrides)))
    return _build(layers)


def load_from_run(
    run_dir: str | Path, overrides: Any = None, extra_files: Any = None
) -> Config:
    """Rebuild a past run's config, optionally with extra file layers on top.

    Layer order is the run's ``resolved_config.yaml``, then each file in ``extra_files``
    (each following its own ``base`` chain), then the CLI. That is the mechanism eval uses:
    the run says what was trained, the eval config says what to test it under, the CLI still
    wins. A missing extra file raises naming the path.
    """
    path = Path(run_dir).expanduser()
    if path.is_dir():
        path = path / RESOLVED_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no resolved config at {path}")
    layer = _file_layer(path)
    for computed in ("meta", "derived"):
        layer.pop(computed, None)
    task_cfg = OmegaConf.select(layer, "task.cfg")
    if task_cfg is not None:
        task_cfg.pop("seed", None)  # written from experiment.seed, never a layer's own value

    layers = [(str(path), layer)]
    for extra in list(extra_files or []):
        layers += load_file_chain(extra)
    layers.append(("CLI", _cli_layer(overrides)))
    return _build(layers)


def _build(layers: List[tuple]) -> Config:
    # 3. the task name: the last layer that sets it wins. The model architecture is picked
    #    the same way (default 'simba'); it decides which dataclasses sit behind
    #    model.actor / model.critic in step 4. Also per layer: no interpolation, no
    #    task.cfg.seed, and record every task.cfg path the layer sets.
    task_name = None
    model_architecture = None  # (value, which layer set it)
    task_overrides: Dict[str, str] = {}
    for where, layer in layers:
        _reject_interpolation(OmegaConf.to_container(layer, resolve=False), where)
        name = OmegaConf.select(layer, "task.name")
        if name is not None:
            task_name = name
        architecture = OmegaConf.select(layer, "model.architecture")
        if architecture is not None:
            model_architecture = (architecture, where)
        layer_task_cfg = OmegaConf.select(layer, "task.cfg")
        if layer_task_cfg is not None:
            if "seed" in layer_task_cfg:
                raise ValueError(
                    f"{where} sets task.cfg.seed: set experiment.seed instead; the env seed is "
                    "always written from it"
                )
            for leaf in _leaf_paths(OmegaConf.to_container(layer_task_cfg, resolve=False)):
                task_overrides[leaf] = where
    if task_name is None:
        raise ValueError("no layer sets task.name; set it in a config file or with task.name=<id>")

    # 4. the schema: sections from their dataclasses, task from the env cfg defaults
    env_cfg = load_task_cfg(task_name)
    task_defaults = task_cfg_to_dict(env_cfg)
    root = OmegaConf.create({"task": {"name": MISSING, "cfg": task_defaults}})
    for name, cls in SECTIONS.items():
        if name == "model" and model_architecture is not None:
            try:
                cls = model_cfg_class(model_architecture[0])
            except ValueError as exc:
                raise ValueError(f"config error from {model_architecture[1]}: {exc}") from exc
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

    # 6b. one seed: experiment.seed is the env's seed too
    if "seed" not in root.task.cfg:
        raise ValueError(
            f"task '{task_name}' has no 'seed' field in its env cfg, so experiment.seed cannot "
            "seed the env"
        )
    root.task.cfg.seed = root.experiment.seed
    task_overrides["seed"] = "experiment.seed"

    # 7. the task dict is not type-checked by OmegaConf, so check it against the defaults.
    #    The checked copy is what gets applied (an int for a float field is converted), and it
    #    goes back into root so the dumped config and the env hold the same value.
    checked = _check_task_types(OmegaConf.to_container(root.task.cfg), task_defaults)
    root.task.cfg = OmegaConf.create(checked)

    # 8. the objects
    sections = {name: OmegaConf.to_object(root[name]) for name in SECTIONS}
    task_container = OmegaConf.to_container(root.task.cfg)
    # The seed is set on the object, not through the dict: every Isaac Lab env cfg defaults
    # `seed` to None, and `from_dict` refuses a value whose type differs from the current one.
    seed = task_container.pop("seed")
    task_cfg = apply_task_cfg(env_cfg, task_container)
    task_cfg.seed = seed

    # 9. derived values
    derived = {
        "run_names": [
            f"{sections['wandb'].group}_a{i}" for i in range(sections["experiment"].num_agents)
        ]
    }

    cfg = Config(task_name, task_cfg, sections, derived, _meta(), root, task_overrides)

    # 10. cross-field rules
    for name, obj in sections.items():
        hook = getattr(obj, "validate", None)
        if callable(hook):
            hook(cfg)
    return cfg


def _leaf_paths(data: Any, path: str = "") -> List[str]:
    """Dotted paths of every leaf (a list counts as one leaf)."""
    if isinstance(data, dict):
        out: List[str] = []
        for key, sub in data.items():
            out += _leaf_paths(sub, f"{path}.{key}" if path else str(key))
        return out
    return [path]


def _check_task_types(merged: Any, default: Any, path: str = "task.cfg") -> Any:
    """Check every task leaf against the type of its default; return the value to apply.

    int is accepted where the default is a float and is **converted**: Isaac Lab's
    ``from_dict`` requires ``isinstance(value, type(current))``, so an int left as an int
    would be refused there (``Expected: float, Received: int``).
    """
    if isinstance(default, dict):
        return {key: _check_task_types(merged[key], sub, f"{path}.{key}") for key, sub in default.items()}
    if default is None:
        return merged
    if isinstance(default, bool):
        ok = isinstance(merged, bool)
    elif isinstance(default, int):
        ok = isinstance(merged, int) and not isinstance(merged, bool)
    elif isinstance(default, float):
        if isinstance(merged, int) and not isinstance(merged, bool):
            return float(merged)
        ok = isinstance(merged, float)
    elif isinstance(default, (list, tuple)):
        ok = isinstance(merged, (list, tuple))  # OmegaConf has no tuple: a tuple default reads back as a list
    else:
        ok = isinstance(merged, type(default))
    if not ok:
        raise TypeError(
            f"{path}: expected {type(default).__name__}, got {type(merged).__name__} {merged!r}"
        )
    return merged


# ------------------------------------------------------------------------------ dump / cli
def _normalize(value: Any) -> Any:
    """Tuples read back from YAML as lists; compare them as lists."""
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


_ABSENT = object()


def _at(data: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(data, dict) or key not in data:
            return _ABSENT
        data = data[key]
    return data


def check_env_kept_overrides(cfg: Config, env_cfg: Any) -> None:
    """Raise if the env changed a ``task.cfg`` value that a file or the CLI set.

    Some envs rewrite parts of their cfg in ``__init__`` (Factory/Forge recompute
    ``observation_space`` and ``state_space`` and copy ``scene.fixed_asset``/``held_asset`` from
    ``task``), which would silently discard an override. Call this after the env exists.
    """
    expected = _normalize(OmegaConf.to_container(cfg.root.task.cfg, resolve=False))
    live = _normalize(task_cfg_to_dict(env_cfg))
    changed = []
    for dotted, where in sorted(cfg.task_overrides.items()):
        want, got = _at(expected, dotted), _at(live, dotted)
        if want != got:
            got_text = "<missing>" if got is _ABSENT else repr(got)
            changed.append(f"  task.cfg.{dotted}: {where} set {want!r}, the env holds {got_text}")
    if changed:
        raise ValueError(
            "the env changed task.cfg values that the config set, so those settings did not take "
            "effect:\n" + "\n".join(changed) + "\nSet the field the env derives them from instead."
        )


def dump(cfg: Config, path: str | Path, env_cfg: Any) -> Path:
    """Write ``resolved_config.yaml`` after the env exists: meta, derived, then every value.

    The ``task.cfg`` section comes from ``env_cfg``, the live env's cfg (``env.unwrapped.cfg``),
    so the file holds what the env actually ran with. Raises first if the env discarded an
    override (:func:`check_env_kept_overrides`).
    """
    check_env_kept_overrides(cfg, env_cfg)
    out = Path(path)
    if out.suffix not in (".yaml", ".yml"):
        out = out / RESOLVED_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    body = OmegaConf.to_container(cfg.root, resolve=False)
    body["task"]["cfg"] = task_cfg_to_dict(env_cfg)
    doc = {"meta": cfg.meta, "derived": cfg.derived, **body}
    out.write_text(OmegaConf.to_yaml(OmegaConf.create(doc)))
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


# package sections are registered here, at the bottom of this file. The imports sit here, not
# at the top, because the config classes they bring in import nothing from this module.
from .learners.cfg import PPOCfg, SACCfg, TrainerCfg  # noqa: E402
from .losses.cfg import LossesCfg  # noqa: E402
from .envs.cfg import ControllerCfg, WrappersCfg  # noqa: E402
from .evaluation import EvalCfg  # noqa: E402
from .memory.cfg import MemoryCfg  # noqa: E402
from .models.cfg import SimbaModelCfg, model_cfg_class  # noqa: E402
from .hpc.cfg import HpcCfg  # noqa: E402

register_section("experiment", ExperimentCfg)
register_section("wandb", WandbCfg)
register_section("trainer", TrainerCfg)
register_section("sac", SACCfg)
register_section("ppo", PPOCfg)
register_section("model", SimbaModelCfg)  # the default; model.architecture swaps it
register_section("memory", MemoryCfg)
register_section("losses", LossesCfg)
register_section("eval", EvalCfg)
register_section("controller", ControllerCfg)
register_section("wrappers", WrappersCfg)
register_section("hpc", HpcCfg)

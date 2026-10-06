"""What every launcher shares: names, the login-node config read, and the sbatch command.

Three ideas carry this module.

**wandb is the interface, so names are derived from it.** A run is found by project, group
and tags long before anyone looks at a SLURM queue, so the launcher takes ``--project`` and
``--group_prefix`` and derives everything else: the group, the SLURM job name, the log
paths. :data:`NAMING` writes the table out.

**The launcher owns three keys.** Because it computes ``wandb.project``, ``wandb.group`` and
``wandb.tags``, a user override of any of them on the launcher's command line is
**rejected**, naming the flag to use instead. Two sources of truth for a run's name is how
runs get lost.

**Fail before queue.** Every config in a batch is read and named before anything is
submitted, so a typo in the last file does not leave half a sweep running.

This module reads config *files* only — through :mod:`robonuke_rl_core.configfile`, the same
chain reader ``config.load_config`` uses. It cannot build a full config: that needs the
task's env cfg, which needs Isaac Lab, which a login node does not have. See the import
discipline in :mod:`robonuke_rl_core.hpc`.
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from ..configfile import OVERRIDE_RE, cli_layer, load_file_chain
from .cfg import SUBMIT_REQUIRED, HpcCfg

__all__ = [
    "NAMING",
    "RESERVED_OVERRIDES",
    "SubmitConfig",
    "SubmitError",
    "read_submit_config",
    "require_submit_fields",
    "group_name",
    "check_group",
    "merge_tags",
    "reject_reserved_overrides",
    "collect_configs",
    "job_script",
    "chain_script",
    "job_env",
    "sbatch_command",
    "run_sbatch",
    "Submission",
    "report",
    "run_names",
    "existing_run_dir",
    "preflight",
    "log_paths",
    "python_command",
    "forwarded",
]

#: how a job's names are derived. Kept as data so the tests and the docs read the same table.
NAMING = {
    "wandb project": "--project",
    "wandb group": "{group_prefix}_{config_stem}[_{LABEL}-{value}]",
    "run names": "{group}_a{i}  (derived.run_names, unchanged)",
    "slurm job name": "the group, exactly",
    "slurm logs": "{hpc.exp_log_dir}/{project}/{group}_%j.out / .err",
    "wandb tags": "config wandb.tags + every --tag + (sweeps) {LABEL}-{value}",
}

#: dotted paths the launcher computes, so a user may not set them on its command line
RESERVED_OVERRIDES = {
    "wandb.project": "--project",
    "wandb.group": "--group_prefix",
    "wandb.tags": "--tag",
}

#: the separator inside a swept ``{LABEL}-{value}`` pair; joins between parts stay ``_``
PAIR_SEPARATOR = "-"


class SubmitError(RuntimeError):
    """Anything that must stop a submit before a job is queued."""


# ------------------------------------------------------------------------------ the names
def group_name(group_prefix: str, config_stem: str, suffix: str = "") -> str:
    """``{prefix}_{stem}`` plus a sweep's ``_{LABEL}-{value}``."""
    group = f"{group_prefix}_{config_stem}"
    return f"{group}_{suffix}" if suffix else group


def check_group(group: str, where: str) -> None:
    """The same rule ``WandbCfg.validate`` applies, checked before anything is queued.

    Run names and run directories derive from the group, so a group with a ``/`` in it does
    not fail at submit — it fails hours later, after the job has started.
    """
    if not group or any(character.isspace() or character == "/" for character in group):
        raise SubmitError(
            f"{where}: computed wandb group {group!r} is not usable — it must be non-empty "
            "and free of whitespace and '/', because run names and run directories derive "
            "from it. Check --project / --group_prefix and the config file name."
        )


def merge_tags(
    config_tags: Sequence[str], cli_tags: Sequence[str], sweep_tag: str = ""
) -> List[str]:
    """Config chain's tags, then every ``--tag``, then the sweep's — first wins, order kept."""
    merged: List[str] = []
    for tag in list(config_tags) + list(cli_tags) + ([sweep_tag] if sweep_tag else []):
        tag = str(tag)
        if tag and tag not in merged:
            merged.append(tag)
    return merged


def reject_reserved_overrides(overrides: Sequence[str]) -> None:
    """Refuse a user override of a key the launcher owns, naming the flag that sets it."""
    for override in overrides:
        key = str(override).split("=", 1)[0].strip()
        if key in RESERVED_OVERRIDES:
            raise SubmitError(
                f"{key} is set by the launcher, so it cannot be passed as an override; use "
                f"{RESERVED_OVERRIDES[key]} instead. Two sources of truth for a run's name "
                "is how runs get lost."
            )


# ------------------------------------------------------------------- reading a config file
@dataclass
class SubmitConfig:
    """What a launcher needs from one experiment file, read without Isaac Lab."""

    path: Path
    #: the file name without ``.yaml`` — the ``config_stem`` in the naming table
    stem: str
    hpc: HpcCfg
    #: from the chain, for ``launch_eval``'s run paths; may be empty
    entity: str = ""
    #: the chain's ``wandb.tags``, before the CLI's are merged in
    tags: List[str] = field(default_factory=list)
    #: the chain's ``trainer.output_dir``, for ``--skip_existing``
    output_dir: str = ""
    #: the chain's ``experiment.num_agents``, for the run paths a train-then-eval job evals
    num_agents: int = 1


def _select(layers: Sequence[Tuple[str, DictConfig]], path: str, default: Any = None) -> Any:
    """The last layer that sets ``path`` wins, exactly as the merge would resolve it."""
    value = default
    for _, layer in layers:
        found = OmegaConf.select(layer, path)
        if found is not None:
            value = found
    return value


def read_submit_config(path: str | Path, overrides: Sequence[str] = ()) -> SubmitConfig:
    """Read one experiment's chain and resolve its `hpc` section, plus what the job needs.

    Struct mode is on, so an unknown ``hpc`` key raises here rather than being ignored into
    a job that then asks SLURM for nothing in particular. ``hpc.*`` CLI overrides are
    applied last, like any CLI layer.
    """
    from ..learners.cfg import TrainerCfg  # import-safe: no torch

    path = Path(path)
    layers = load_file_chain(path)
    node = OmegaConf.structured(HpcCfg)
    OmegaConf.set_struct(node, True)
    for where, layer in layers:
        section = OmegaConf.select(layer, "hpc")
        if section is not None:
            try:
                node = OmegaConf.merge(node, section)
            except Exception as exc:  # OmegaConf's own type/struct errors
                raise SubmitError(f"hpc section error from {where}: {exc}") from exc
    cli = OmegaConf.select(cli_layer(overrides), "hpc")
    if cli is not None:
        try:
            node = OmegaConf.merge(node, cli)
        except Exception as exc:
            raise SubmitError(f"hpc section error from the command line: {exc}") from exc

    hpc: HpcCfg = OmegaConf.to_object(node)
    try:
        hpc.validate(None)
    except ValueError as exc:
        raise SubmitError(f"{path}: {exc}") from exc

    tags = _select(layers, "wandb.tags", default=[]) or []
    return SubmitConfig(
        path=path,
        stem=path.stem,
        hpc=hpc,
        entity=str(_select(layers, "wandb.entity", default="") or ""),
        tags=[str(tag) for tag in tags],
        output_dir=str(_select(layers, "trainer.output_dir", default=TrainerCfg().output_dir)),
        num_agents=int(_select(layers, "experiment.num_agents", default=1) or 1),
    )


def require_submit_fields(submit: SubmitConfig) -> None:
    """Enforce the fields a job cannot be built without. Not in ``validate``, on purpose.

    A local training run never touches SLURM, so these are empty-by-default in the section
    and required only here, where a job is about to be queued.
    """
    missing = [name for name in SUBMIT_REQUIRED if not getattr(submit.hpc, name)]
    if missing:
        raise SubmitError(
            f"{submit.path}: hpc.{', hpc.'.join(missing)} must be set to submit a job. Set "
            "them in the project's configs/base/hpc.yaml (cluster-wide), in the experiment "
            "file, or on the command line (e.g. hpc.account=my-grp). hpc.cache_home must be "
            "scratch with room for GBs of Kit and shader caches, never an NFS home."
        )


# ------------------------------------------------------------------------ collecting files
def collect_configs(paths: Sequence[str]) -> List[Path]:
    """Experiment files from a mix of folders and explicit paths.

    A folder contributes every ``*.yaml`` directly inside it, sorted, **skipping
    underscore-prefixed names** — those are overlays and base files, not runnable
    experiments. An explicit path is taken as given, underscore or not: naming a file is an
    unambiguous request. Zero configs is an error, never a quiet no-op.
    """
    found: List[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_dir():
            inside = sorted(
                child
                for child in candidate.iterdir()
                if child.is_file()
                and child.suffix == ".yaml"
                and not child.name.startswith("_")
            )
            if not inside:
                raise SubmitError(
                    f"{candidate} holds no runnable *.yaml (underscore-prefixed files are "
                    "treated as overlays and skipped)"
                )
            found.extend(inside)
        elif candidate.is_file():
            found.append(candidate)
        else:
            raise SubmitError(f"config path not found: {candidate}")
    if not found:
        raise SubmitError("no configs to submit; pass a folder or one or more *.yaml files")
    # the same file named twice (a folder plus an explicit path) runs once
    unique: List[Path] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


# ----------------------------------------------------------------------------- the sbatch
def job_script() -> Path:
    """The bash entry point, shipped as package data and submitted by path."""
    return Path(__file__).resolve().parent / "hpc_job.bash"


def chain_script() -> Path:
    """The in-container train-then-eval wrapper, bind-mounted into the job."""
    return Path(__file__).resolve().parent / "hpc_job_chain.bash"


def job_env(
    hpc: HpcCfg, *, package_root: Path, project_root: Path, extra: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """The ``RNK_*`` variables the job script reads. ``ALL`` carries WANDB_API_KEY too."""
    env = {
        "RNK_PKG_ROOT": str(package_root),
        "RNK_PROJECT_ROOT": str(project_root),
        "RNK_SIF": hpc.sif_image,
        "RNK_APPTAINER_BIN": hpc.apptainer_bin,
        "RNK_CACHE_HOME": hpc.cache_home,
        "RNK_BINDS": ",".join(hpc.binds),
        "RNK_PYTHON": hpc.container_python,
    }
    env.update(extra or {})
    return env


def sbatch_command(
    *,
    hpc: HpcCfg,
    job_name: str,
    out_path: Path,
    err_path: Path,
    env: Dict[str, str],
    argv: Sequence[str],
) -> List[str]:
    """The full ``sbatch ...`` argv for one job.

    The resource flags come from **that config's** `hpc` values, which is the whole point of
    the section: a recording eval can ask for more walltime than the training run did
    without anyone editing a shell file.
    """
    exports = ",".join(["ALL"] + [f"{key}={value}" for key, value in env.items()])
    return [
        "sbatch",
        "-A", hpc.account,
        "-p", hpc.partitions,
        "--time", hpc.time,
        f"--gres=gpu:{hpc.gpus}",
        "--mem", hpc.mem,
        "-c", str(hpc.cpus),
        "--signal", hpc.signal,
        "-J", job_name,
        "-o", str(out_path),
        "-e", str(err_path),
        f"--export={exports}",
        str(job_script()),
        *argv,
    ]


@dataclass
class Submission:
    """One job's outcome, for the end-of-run summary."""

    group: str
    config: Path
    status: str  # "submitted" | "skipped" | "failed" | "dry_run"
    detail: str = ""


def run_sbatch(command: Sequence[str], *, dry_run: bool) -> Tuple[bool, str]:
    """Submit, or print what would be submitted. Returns ``(ok, detail)``.

    A submission failure after the fail-before-queue gate does not abort the batch: the
    other configs are still worth queueing, and the summary reports what failed.
    """
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        return True, "dry run"
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return False, str(exc)
    if done.returncode != 0:
        return False, (done.stderr or done.stdout).strip() or f"sbatch exit {done.returncode}"
    return True, (done.stdout or "").strip()


def preflight(submits: Sequence[SubmitConfig], *, dry_run: bool) -> None:
    """Checks that must pass before the first job is queued."""
    if not dry_run and shutil.which("sbatch") is None:
        raise SubmitError(
            "sbatch is not on PATH; submitters run on a cluster login node. Use --dry_run to "
            "see the commands without submitting."
        )
    for submit in submits:
        require_submit_fields(submit)
        if not dry_run and not Path(submit.hpc.sif_image).is_file():
            raise SubmitError(
                f"{submit.path}: hpc.sif_image {submit.hpc.sif_image!r} is not a file. Build "
                "it with hpc/build_image.sh, or point the field at the built image."
            )


def log_paths(hpc: HpcCfg, project: str, name: str, *, create: bool = True) -> Tuple[Path, Path]:
    """``{exp_log_dir}/{project}/{name}_%j.out`` and ``.err``.

    SLURM will not create the directory, so a real submit does. A dry run passes
    ``create=False``: printing what *would* happen must not leave anything behind.
    """
    directory = Path(hpc.exp_log_dir) / project
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_%j.out", directory / f"{name}_%j.err"


def report(submissions: Sequence[Submission]) -> int:
    """Print the summary and return the exit code: nonzero if anything failed."""
    by_status: Dict[str, List[Submission]] = {}
    for item in submissions:
        by_status.setdefault(item.status, []).append(item)
    counts = ", ".join(f"{len(items)} {status}" for status, items in sorted(by_status.items()))
    print(f"\n[launch] {counts or 'nothing to do'}", flush=True)
    for status in sorted(by_status):
        for item in by_status[status]:
            detail = f"  ({item.detail})" if item.detail else ""
            print(f"[launch]   {status:9s} {item.group}{detail}", flush=True)
    return 1 if by_status.get("failed") else 0


def python_command(hpc: HpcCfg, script: str, argv: Sequence[str]) -> List[str]:
    """``<container python> <script> <argv...>``, the command the job execs."""
    return [hpc.container_python, script, *argv]


def forwarded(overrides: Sequence[str]) -> List[str]:
    """User dotted overrides, verbatim, after a shape check.

    Forwarded unchanged — ``hpc.*`` included — so the training process re-applies them and
    ``resolved_config.yaml`` records exactly what the job ran with, down to the resources
    that shaped it.
    """
    for override in overrides:
        if not OVERRIDE_RE.match(str(override)):
            raise SubmitError(
                f"unrecognized argument {override!r}: an override must look like "
                "'section.field=value' (e.g. task.cfg.scene.num_envs=128)"
            )
    return [str(override) for override in overrides]


def run_names(group: str, num_agents: int) -> List[str]:
    """``derived.run_names`` without loading the config: the rule is ``{group}_a{i}``.

    Duplicated from ``config._build`` on purpose — the submitter cannot import that module
    (torch), and the rule is one f-string. ``tests/hpc/test_naming.py`` pins the two
    together so a change to either is caught.
    """
    return [f"{group}_a{index}" for index in range(num_agents)]


def existing_run_dir(output_dir: str, project: str, group: str) -> Tuple[bool, str]:
    """``--skip_existing``: is ``{output_dir}/{project}/{group}`` there and non-empty?

    The reason is returned either way, because a silent skip hides a bug: "not present" and
    "present but empty" are different situations and both are worth printing.
    """
    directory = Path(output_dir) / project / group
    if not directory.exists():
        return False, f"no run dir at {directory}"
    if not any(directory.iterdir()):
        return False, f"run dir {directory} is empty"
    return True, f"run dir {directory} exists and is non-empty"

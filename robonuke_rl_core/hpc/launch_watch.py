"""``launch_watch``: one SLURM job per training config, running the project's checkpoint
watcher over that config's run directory.

    python launchers/launch_watch.py <folder | config.yaml ...> --project P --group_prefix G \\
        --eval_config F [--follow] [--dry_run] [a.b.c=value ...]

The job runs ``hpc.watch_script`` (a project script: the package has no watcher of its own)
with ``--run <run dir> --eval_config F --headless`` and, by default, ``--once``: evaluate every
checkpoint the run has on disk, then exit -- for runs that have finished. ``--follow`` drops
``--once`` so the watcher keeps polling for new checkpoints, for a watcher started beside a
training. The run directory is ``{trainer.output_dir}/{project}/{group}`` with the group named by
the **same rule** ``launch_train`` used (``{group_prefix}_{config stem}``), so the command that
launched the trainings launches their watchers with the verb swapped.

**Resources come from the eval config's own chain**, as for ``launch_eval``: the watcher's
walltime, partitions and memory are the evaluation's business, not the training file's. The
eval chain also supplies ``hpc.watch_script`` (normally from the project's base ``hpc.yaml``).

**Fail before queue.** Every training config is read and named, the eval config resolved, the
watch script checked to exist, and every run directory checked to exist before the first job
goes out; a real submit aborts on a missing run directory (the training has not started), a
``--dry_run`` prints the command anyway and says which directories are missing.

No wandb runs are created here -- the watcher logs into its own eval runs beside the training
run -- so naming is for SLURM only: job ``watch_{group}``, logs under the same name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from . import submit as S
from .launch_train import add_common_args, package_root, project_root

__all__ = ["main", "watch_argv"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="the training experiment files, or folders of them (non-recursive, '_'-prefixed skipped)",
    )
    parser.add_argument("--project", required=True, help="the trainings' wandb project (also the log folder)")
    parser.add_argument(
        "--group_prefix",
        required=True,
        help="the trainings' group prefix: the run dir is {trainer.output_dir}/{project}/{group_prefix}_{stem}",
    )
    parser.add_argument(
        "--eval_config",
        required=True,
        help="the eval conditions the watcher layers on each run; its chain supplies the job's resources",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="keep polling for new checkpoints (a watcher beside a training); default: --once, evaluate what "
        "is on disk and exit",
    )
    add_common_args(parser)
    return parser


def watch_argv(script: str, run_dir: str, eval_config: str, *, follow: bool, forwarded: Sequence[str]) -> List[str]:
    """The watcher's command line inside the job: ``<script> --run <run dir> --eval_config F --headless
    [--once] <overrides...>``."""
    return [
        script,
        "--run",
        str(run_dir),
        "--eval_config",
        str(eval_config),
        "--headless",
        *([] if follow else ["--once"]),
        *forwarded,
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    try:
        return _run(args, overrides)
    except S.SubmitError as exc:
        print(f"[launch] error: {exc}", flush=True)
        return 2


def _run(args: argparse.Namespace, overrides: Sequence[str]) -> int:
    S.reject_reserved_overrides(overrides)
    forwarded = S.forwarded(overrides)
    configs = S.collect_configs(args.configs)

    eval_path = Path(args.eval_config)
    if not eval_path.is_file():
        raise S.SubmitError(f"--eval_config {eval_path} is not a file")
    # the eval config's chain: the job's resources and the watch script
    eval_config = S.read_submit_config(eval_path, overrides)
    S.preflight([eval_config], dry_run=args.dry_run)

    package = Path(args.package_root).resolve() if args.package_root else package_root()
    project_dir = Path(args.project_root).resolve() if args.project_root else project_root()

    script = eval_config.hpc.watch_script
    if not script:
        raise S.SubmitError(
            f"{eval_path}: hpc.watch_script is empty; launch_watch runs the project's checkpoint watcher, so "
            "name it in the project's configs/base/hpc.yaml (e.g. hpc.watch_script: scripts/watch_residual_eval.py)"
        )
    if not (project_dir / script).is_file():
        raise S.SubmitError(
            f"{eval_path}: hpc.watch_script {script!r} is not a file under the project root {project_dir}"
        )

    # ---- the gate: name every run and find its directory before anything is queued ----
    planned = []
    missing = []
    for path in configs:
        config = S.read_submit_config(path, overrides)
        group = S.group_name(args.group_prefix, config.stem)
        S.check_group(group, str(path))
        run_dir = Path(config.output_dir) / args.project / group
        if not (project_dir / run_dir).is_dir():
            missing.append(str(run_dir))
        planned.append((config, group, run_dir))
    if missing:
        listed = ", ".join(missing)
        if not args.dry_run:
            raise S.SubmitError(
                f"no run directory for {len(missing)} of {len(planned)} config(s): {listed}. The watcher reads the "
                "training's checkpoints from there, so launch it after the training has started (same --project "
                "and --group_prefix as launch_train). Nothing was submitted."
            )
        print(f"[launch] dry_run: {len(missing)} run director{'y' if len(missing) == 1 else 'ies'} missing "
              f"(a real submit would abort): {listed}", flush=True)

    env = S.job_env(eval_config.hpc, package_root=package, project_root=project_dir)
    submissions: List[S.Submission] = []
    for config, group, run_dir in planned:
        name = f"watch_{group}"
        argv = watch_argv(script, str(run_dir), str(eval_path), follow=args.follow, forwarded=forwarded)
        out_path, err_path = S.log_paths(eval_config.hpc, args.project, name, create=not args.dry_run)
        sbatch = S.sbatch_command(
            hpc=eval_config.hpc,
            job_name=name,
            out_path=out_path,
            err_path=err_path,
            env=env,
            argv=S.python_command(eval_config.hpc, argv[0], argv[1:]),
        )
        ok, detail = S.run_sbatch(sbatch, dry_run=args.dry_run)
        status = ("dry_run" if args.dry_run else "submitted") if ok else "failed"
        submissions.append(S.Submission(name, config.path, status, "" if ok else detail))

    return S.report(submissions)

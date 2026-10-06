"""``launch_eval``: one SLURM job per (training run x eval config).

    python launchers/launch_eval.py --eval_config F [--eval_config F2 ...] \\
        ( --run entity/project/run ... | --project P [--entity E] [--group G ...] \\
          [--wandb_tag T ...] ) \\
        [--checkpoint best|<step>] [--dry_run] [a.b.c=value ...]

Runs are named explicitly with ``--run``, or found by querying wandb for a project plus any
of group / tags — the useful half of picking runs by hand. Zero matches is an **error** that
lists the filters, never a silent no-op: a sweep's evals quietly matching nothing looks
exactly like a sweep whose evals all passed.

**Resources come from the eval config's own chain.** A heavy recording eval sets its own
``hpc.time`` and ``hpc.mem`` in its file and gets them, which is the point of making the
resources config rather than a shell file.

No wandb runs are created here — ``eval.py`` writes into the *training* run — so naming is
for SLURM only: job ``eval_{run_name}_{eval_stem}``, logs under the same name.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import submit as S
from .launch_train import add_common_args, package_root, project_root

__all__ = ["main", "find_runs", "run_path_parts"]


def run_path_parts(path: str) -> Tuple[str, str, str]:
    """Split ``entity/project/run`` the way ``eval.py --run`` reads it."""
    parts = [piece for piece in str(path).split("/") if piece]
    if len(parts) != 3:
        raise S.SubmitError(
            f"--run {path!r} must be 'entity/project/<run id or name>', got {len(parts)} parts"
        )
    return parts[0], parts[1], parts[2]


def find_runs(
    *,
    entity: str,
    project: str,
    groups: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> List[str]:
    """Query wandb for ``entity/project`` runs matching any group and all tags.

    ``import wandb`` happens **here**, not at module import: the login node needs it only
    for this one path, and the import discipline for ``robonuke_rl_core.hpc`` is that
    nothing heavy loads just to submit a job.
    """
    import wandb  # noqa: PLC0415  (lazy on purpose -- see the docstring)

    filters: dict = {}
    if groups:
        filters["group"] = {"$in": list(groups)}
    if tags:
        filters["tags"] = {"$all": list(tags)}
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters or None)
    names = [run.name for run in runs]
    if not names:
        raise S.SubmitError(
            "no wandb runs matched: entity="
            f"{entity!r} project={project!r} groups={list(groups) or '<any>'} "
            f"tags={list(tags) or '<any>'}. Nothing was submitted."
        )
    return [f"{entity}/{project}/{name}" for name in sorted(names)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval_config",
        action="append",
        default=[],
        dest="eval_configs",
        required=True,
        help="an eval config to run; repeatable. Its own chain supplies the job's resources",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        dest="runs",
        help="entity/project/run to evaluate; repeatable. Skips the wandb query",
    )
    parser.add_argument("--project", default=None, help="wandb project to query for runs")
    parser.add_argument(
        "--entity",
        default=None,
        help="wandb entity to query (default: wandb.entity from the eval config's chain)",
    )
    parser.add_argument(
        "--group", action="append", default=[], dest="groups", help="only runs in this group"
    )
    parser.add_argument(
        "--wandb_tag",
        action="append",
        default=[],
        dest="wandb_tags",
        help="only runs carrying every one of these tags",
    )
    parser.add_argument("--checkpoint", default="best", help="best | <step> | <file name>")
    add_common_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    try:
        return _run(args, overrides)
    except S.SubmitError as exc:
        print(f"[launch] error: {exc}", flush=True)
        return 2


def _run(args: argparse.Namespace, overrides: Sequence[str]) -> int:
    # eval creates no wandb run, so the launcher owns no wandb keys here -- but rejecting
    # them keeps one rule across the three verbs
    S.reject_reserved_overrides(overrides)
    forwarded = S.forwarded(overrides)

    if not args.runs and not args.project:
        raise S.SubmitError("pass --run paths, or --project (with optional --group/--wandb_tag)")

    # ---- the gate: read every eval config and resolve every run before queueing ----
    configs = []
    for raw in args.eval_configs:
        path = Path(raw)
        if not path.is_file():
            raise S.SubmitError(f"--eval_config {path} is not a file")
        configs.append(S.read_submit_config(path, overrides))
    S.preflight(configs, dry_run=args.dry_run)

    runs = list(args.runs)
    for path in runs:
        run_path_parts(path)  # shape-check every explicit path before anything is queued
    if not runs:
        entity = args.entity or next((c.entity for c in configs if c.entity), "")
        if not entity:
            raise S.SubmitError(
                "no wandb entity: pass --entity, or set wandb.entity in the eval config's chain"
            )
        runs = find_runs(
            entity=entity,
            project=args.project,
            groups=args.groups,
            tags=args.wandb_tags,
        )
    print(f"[launch] {len(runs)} run(s) to evaluate:", flush=True)
    for path in runs:
        print(f"[launch]   {path}", flush=True)

    package = Path(args.package_root).resolve() if args.package_root else package_root()
    project_dir = Path(args.project_root).resolve() if args.project_root else project_root()

    submissions: List[S.Submission] = []
    for config in configs:
        for path in runs:
            _, run_project, run_name = run_path_parts(path)
            name = f"eval_{run_name}_{config.stem}"
            argv = [
                "scripts/eval.py",
                "--run",
                path,
                "--eval_config",
                str(config.path),
                "--checkpoint",
                str(args.checkpoint),
                "--headless",
                *forwarded,
            ]
            out_path, err_path = S.log_paths(
                config.hpc, run_project, name, create=not args.dry_run
            )
            sbatch = S.sbatch_command(
                hpc=config.hpc,
                job_name=name,
                out_path=out_path,
                err_path=err_path,
                env=S.job_env(
                    config.hpc, package_root=package, project_root=project_dir
                ),
                argv=S.python_command(config.hpc, argv[0], argv[1:]),
            )
            ok, detail = S.run_sbatch(sbatch, dry_run=args.dry_run)
            status = ("dry_run" if args.dry_run else "submitted") if ok else "failed"
            submissions.append(S.Submission(name, config.path, status, "" if ok else detail))

    return S.report(submissions)

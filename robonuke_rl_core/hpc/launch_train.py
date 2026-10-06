"""``launch_train``: one SLURM job per experiment config.

    python launchers/launch_train.py <folder | config.yaml ...> \\
        --project P --group_prefix G [--tag T ...] \\
        [--skip_existing] [--eval_config F] [--checkpoint best] [--dry_run] \\
        [a.b.c=value ...]

**Fail before queue.** Every config is read, resolved and named before the first job is
submitted, so a typo in the last file of a folder does not leave the first half of a batch
running. After that gate a *submission* failure is reported and the batch continues: the
other configs are still worth queueing.

**The launcher owns the wandb names.** ``wandb.project``, ``wandb.group`` and ``wandb.tags``
are computed here and appended **last**, so they are the final CLI layer. Passing any of them
as an override is rejected (see :data:`~robonuke_rl_core.hpc.submit.RESERVED_OVERRIDES`).
Every other override is forwarded verbatim, ``hpc.*`` included, so the job re-applies it and
``resolved_config.yaml`` records exactly what ran.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from . import submit as S

__all__ = ["main", "add_common_args", "package_root", "project_root"]


def package_root() -> Path:
    """The package clone this launcher is running from — bound over the image's install."""
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """The project clone, i.e. the cwd the configs are relative to."""
    return Path.cwd().resolve()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Flags every launcher shares."""
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print the full sbatch command for every job and submit nothing",
    )
    parser.add_argument(
        "--package_root",
        default=None,
        help="the package clone to bind into the image (default: this file's repo)",
    )
    parser.add_argument(
        "--project_root",
        default=None,
        help="the project clone to bind and use as cwd (default: the current directory)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="experiment files, or folders of them (non-recursive, '_'-prefixed skipped)",
    )
    parser.add_argument("--project", required=True, help="wandb project (also the log folder)")
    parser.add_argument(
        "--group_prefix",
        required=True,
        help="wandb group is {group_prefix}_{config stem}; the SLURM job name is the group",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        dest="tags",
        help="extra wandb tag; repeatable. Merged after the config chain's wandb.tags",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip a config whose run dir already exists and is non-empty",
    )
    parser.add_argument(
        "--eval_config",
        default=None,
        help="run this eval config after training, once per agent, in the same job",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="which checkpoint the post-training eval uses (default: best)",
    )
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
    S.reject_reserved_overrides(overrides)
    forwarded = S.forwarded(overrides)
    configs = S.collect_configs(args.configs)

    # ---- the gate: read, resolve and name EVERYTHING before anything is queued ----
    planned = []
    for path in configs:
        config = S.read_submit_config(path, overrides)
        group = S.group_name(args.group_prefix, config.stem)
        S.check_group(group, str(path))
        planned.append((config, group))
    S.preflight([config for config, _ in planned], dry_run=args.dry_run)

    if args.eval_config and not Path(args.eval_config).is_file():
        raise S.SubmitError(f"--eval_config {args.eval_config} is not a file")

    package = Path(args.package_root).resolve() if args.package_root else package_root()
    project = Path(args.project_root).resolve() if args.project_root else project_root()

    submissions: List[S.Submission] = []
    for config, group in planned:
        if args.skip_existing:
            exists, why = S.existing_run_dir(config.output_dir, args.project, group)
            # the reason is printed either way: a silent skip hides a bug, and "not present"
            # and "present but empty" are different situations
            print(f"[launch] {group}: {why}", flush=True)
            if exists:
                submissions.append(S.Submission(group, config.path, "skipped", why))
                continue

        tags = S.merge_tags(config.tags, args.tags)
        argv = [
            "scripts/train.py",
            "--config",
            str(config.path),
            "--headless",
            *forwarded,
            # last, so the launcher's names are the final CLI layer
            f"wandb.project={args.project}",
            f"wandb.group={group}",
            f"wandb.tags=[{','.join(tags)}]",
        ]
        command = S.python_command(config.hpc, argv[0], argv[1:])

        env = S.job_env(config.hpc, package_root=package, project_root=project)
        if args.eval_config:
            # a train-then-eval job cannot be one exec'd python, so the job runs the
            # in-container wrapper instead; the eval half is non-fatal there
            env["RNK_CHAIN_SCRIPT"] = str(S.chain_script())
            env["RNK_EVAL_CONFIG"] = str(args.eval_config)
            env["RNK_EVAL_RUNS"] = ",".join(
                f"{config.entity}/{args.project}/{name}"
                for name in S.run_names(group, config.num_agents)
            )
            env["RNK_EVAL_CHECKPOINT"] = str(args.checkpoint)
            if not config.entity:
                raise S.SubmitError(
                    f"{config.path}: --eval_config needs wandb.entity in the config chain, so "
                    "the post-training eval can name the runs it evaluates"
                )

        out_path, err_path = S.log_paths(
            config.hpc, args.project, group, create=not args.dry_run
        )
        sbatch = S.sbatch_command(
            hpc=config.hpc,
            job_name=group,
            out_path=out_path,
            err_path=err_path,
            env=env,
            argv=command,
        )
        ok, detail = S.run_sbatch(sbatch, dry_run=args.dry_run)
        status = ("dry_run" if args.dry_run else "submitted") if ok else "failed"
        submissions.append(S.Submission(group, config.path, status, "" if ok else detail))

    return S.report(submissions)

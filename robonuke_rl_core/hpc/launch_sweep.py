"""``launch_sweep``: ``launch_train``, once per (config x swept value).

    python launchers/launch_sweep.py <folder | config.yaml ...> \\
        --project P --group_prefix G --sweep_param dotted.path --label L \\
        --value V [--value V2 ...] [--tag T ...] [--skip_existing] [--dry_run] \\
        [a.b.c=value ...]

**No overlay files are generated.** The CLI override *is* the mechanism: each job gets
``{sweep_param}={value}`` appended to the forwarded overrides, and the job's own
``resolved_config.yaml`` plus its ``task_overrides`` are the record that generated overlays
existed to provide. Nothing to clean up afterwards, and nothing that can go stale.

The swept value appears in the group and in a tag as ``{LABEL}-{value}``: a hyphen inside
the pair, because the label and the value belong together, while the joins between parts of
a name stay underscores. A sweep over 100/500 with ``--label kp`` gives groups
``..._kp-100`` / ``..._kp-500`` and tags ``kp-100`` / ``kp-500``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import yaml

from . import submit as S
from .launch_train import add_common_args, package_root, project_root

__all__ = ["main", "parse_value"]


def parse_value(raw: str) -> Tuple[str, str]:
    """``--value V`` or ``--value NAME=V`` -> ``(name for the group, value text)``.

    The value text is forwarded **verbatim** so YAML does the parsing once, in the job:
    ``80``, ``0.08``, ``true``, ``[0.0,15.0,0.0]`` all work, lists written without spaces.

    A scalar names itself. A list or a dict cannot: ``[0.0,15.0,0.0]`` in a group name would
    be unreadable and would carry characters a wandb group may not have, so the explicit
    ``NAME=value`` form is **required** there and the error says so. Any ``=`` is read as
    that form.
    """
    name, separator, text = raw.partition("=")
    if separator:
        if not name:
            raise S.SubmitError(f"--value {raw!r}: the name before '=' is empty")
        return name, text
    try:
        parsed: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise S.SubmitError(f"--value {raw!r} is not valid YAML: {exc}") from exc
    if isinstance(parsed, (list, dict)):
        raise S.SubmitError(
            f"--value {raw!r} is a {type(parsed).__name__}, which cannot name itself in a "
            f"wandb group. Use the NAME=value form, e.g. --value low={raw}"
        )
    return raw, raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("configs", nargs="+", help="experiment files, or folders of them")
    parser.add_argument("--project", required=True, help="wandb project")
    parser.add_argument("--group_prefix", required=True, help="wandb group prefix")
    parser.add_argument(
        "--sweep_param", required=True, help="the dotted config path to sweep, e.g. sac.actor_lr"
    )
    parser.add_argument(
        "--label",
        required=True,
        help="short name for the swept parameter; appears as {LABEL}-{value}",
    )
    parser.add_argument(
        "--value",
        action="append",
        default=[],
        dest="values",
        required=True,
        help="a value to sweep; repeatable. NAME=value for lists and dicts",
    )
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument("--skip_existing", action="store_true")
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
    if args.sweep_param in S.RESERVED_OVERRIDES:
        raise S.SubmitError(
            f"--sweep_param {args.sweep_param} is set by the launcher; use "
            f"{S.RESERVED_OVERRIDES[args.sweep_param]}"
        )
    forwarded = S.forwarded(overrides)
    pairs = [parse_value(raw) for raw in args.values]
    if not args.label or any(c.isspace() or c == "/" for c in args.label):
        raise S.SubmitError(f"--label must be whitespace- and '/'-free, got {args.label!r}")
    configs = S.collect_configs(args.configs)

    # ---- the gate: every (config, value) named before anything is queued ----
    planned = []
    for path in configs:
        config = S.read_submit_config(path, overrides)
        for value_name, value_text in pairs:
            suffix = f"{args.label}{S.PAIR_SEPARATOR}{value_name}"
            group = S.group_name(args.group_prefix, config.stem, suffix)
            S.check_group(group, f"{path} ({args.sweep_param}={value_text})")
            planned.append((config, group, suffix, value_text))
    S.preflight([config for config, _, _, _ in planned], dry_run=args.dry_run)

    package = Path(args.package_root).resolve() if args.package_root else package_root()
    project = Path(args.project_root).resolve() if args.project_root else project_root()

    submissions: List[S.Submission] = []
    for config, group, suffix, value_text in planned:
        if args.skip_existing:
            exists, why = S.existing_run_dir(config.output_dir, args.project, group)
            print(f"[launch] {group}: {why}", flush=True)
            if exists:
                submissions.append(S.Submission(group, config.path, "skipped", why))
                continue

        tags = S.merge_tags(config.tags, args.tags, suffix)
        argv = [
            "scripts/train.py",
            "--config",
            str(config.path),
            "--headless",
            # the user's overrides, then the sweep's, then the launcher's wandb keys
            *forwarded,
            f"{args.sweep_param}={value_text}",
            f"wandb.project={args.project}",
            f"wandb.group={group}",
            f"wandb.tags=[{','.join(tags)}]",
        ]
        out_path, err_path = S.log_paths(
            config.hpc, args.project, group, create=not args.dry_run
        )
        sbatch = S.sbatch_command(
            hpc=config.hpc,
            job_name=group,
            out_path=out_path,
            err_path=err_path,
            env=S.job_env(config.hpc, package_root=package, project_root=project),
            argv=S.python_command(config.hpc, argv[0], argv[1:]),
        )
        ok, detail = S.run_sbatch(sbatch, dry_run=args.dry_run)
        status = ("dry_run" if args.dry_run else "submitted") if ok else "failed"
        submissions.append(S.Submission(group, config.path, status, "" if ok else detail))

    return S.report(submissions)

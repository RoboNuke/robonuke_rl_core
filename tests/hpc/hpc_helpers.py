"""A temp config tree the launcher tests drive, so no test depends on examples/."""

from __future__ import annotations

from pathlib import Path

BASE = """\
hpc:
  account: virl-grp
  partitions: dgxh,tiamat
  sif_image: {sif}
  cache_home: {cache}
  exp_log_dir: {logs}
experiment:
  num_agents: 2
  seed: 1
wandb:
  entity: hur
  project: placeholder
  group: placeholder
  tags: [base_tag]
trainer:
  learner: sac
  total_timesteps: 10
  output_dir: {runs}
task:
  name: Isaac-Forge-PegInsert-Direct-v0
"""

EXPERIMENT = """\
base: {base}
{extra}"""


def tree(tmp_path: Path, *, names=("alpha", "beta"), extra: str = "") -> dict:
    """Build `tmp_path/configs/` with a base plus one experiment per name.

    Returns the paths a test needs. The `.sif` is a real (empty) file so preflight's
    existence check passes without a cluster.
    """
    root = tmp_path / "configs"
    root.mkdir(parents=True, exist_ok=True)
    sif = tmp_path / "image.sif"
    sif.write_text("")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    runs = tmp_path / "runs"
    logs = tmp_path / "exp_logs"

    base = root / "_base.yaml"  # underscore-prefixed: an overlay, never collected as runnable
    base.write_text(
        BASE.format(sif=sif, cache=cache, runs=runs, logs=logs)
    )
    configs = {}
    for name in names:
        path = root / f"{name}.yaml"
        path.write_text(EXPERIMENT.format(base=base.name, extra=extra))
        configs[name] = path
    return {
        "root": root,
        "base": base,
        "configs": configs,
        "sif": sif,
        "cache": cache,
        "runs": runs,
        "logs": logs,
    }


def required_overrides(paths: dict) -> list:
    """Nothing: the temp base already sets every required-at-submit field."""
    return []


def sbatch_lines(captured: str) -> list:
    """The sbatch command lines out of a --dry_run's stdout."""
    return [line for line in captured.splitlines() if line.startswith("sbatch ")]


def flag(line: str, name: str) -> str:
    """The value after `name` in an sbatch command line."""
    parts = line.split()
    return parts[parts.index(name) + 1]

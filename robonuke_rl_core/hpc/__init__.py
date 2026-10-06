"""Launching package runs on a SLURM + Apptainer cluster.

**Import discipline: nothing in this package may import torch, wandb or Isaac Lab at module
level.** The submitters run on a login node, which has a light python and none of the
Isaac environment; a stray heavy import turns "submit a job" into "install Isaac Lab on the
login node". wandb is imported lazily, inside the one function that queries runs for
``launch_eval``. ``tests/hpc/test_import_discipline.py`` enforces this in a subprocess.
"""

from __future__ import annotations

from .cfg import HpcCfg

__all__ = ["HpcCfg"]

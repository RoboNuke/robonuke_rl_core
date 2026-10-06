"""HPC configuration: the `hpc` section.

**The resources are config, not a shell file.** Different experiments need different
walltime, memory and GPUs; editing an ``hpc_env.bash`` between submissions is the thing this
removes. Cluster-wide values live in a base YAML in the project, an experiment overrides
what differs, and a one-off goes on the CLI (``hpc.time=2-00:00:00``). Because the section
rides the normal chain it also lands in ``resolved_config.yaml`` — every run documents the
resources it ran under.

**Every field has a default**, empty string where there is no sane one. A ``MISSING`` field
here would make every *local* training run fail on a field it never needs. Required-ness is
enforced by the submitter instead (:func:`~robonuke_rl_core.hpc.submit.require_submit_fields`),
at the one moment it matters: just before something is queued.

Secrets never enter the config. ``WANDB_API_KEY`` stays an environment variable, read from
the login shell and carried into the job by ``sbatch --export=ALL``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

__all__ = ["HpcCfg", "TIME_RE", "SIGNAL_RE", "SUBMIT_REQUIRED"]

#: SLURM walltime: ``[D-]HH:MM:SS``
TIME_RE = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
#: ``--signal``: ``[R:]SIG@seconds``
SIGNAL_RE = re.compile(r"^(?:[RB]:)?[A-Z]+\d*@\d+$")
#: fields the submitter requires to be non-empty, in the order they are reported
SUBMIT_REQUIRED = ("account", "partitions", "sif_image", "cache_home")


@dataclass
class HpcCfg:
    """SLURM resources and container details for one job."""

    # ---- SLURM ----
    #: ``-A``; the allocation to charge. Required at submit.
    account: str = ""
    #: ``-p``; comma-separated partition list, tried in order. Required at submit.
    partitions: str = ""
    #: ``--time`` as ``[D-]HH:MM:SS``
    time: str = "0-09:00:00"
    #: ``--gres=gpu:{gpus}``. The package trains every agent in one process on one GPU, so
    #: everything here is written and tested for 1.
    gpus: int = 1
    #: ``--mem``, e.g. ``32G``
    mem: str = "32G"
    #: ``-c``, CPUs per task
    cpus: int = 12
    #: ``--signal``; SLURM sends this ahead of the walltime kill, so the job can bail out
    signal: str = "TERM@300"
    #: where the ``.out`` / ``.err`` land, relative to the project root
    exp_log_dir: str = "exp_logs"

    # ---- container ----
    #: absolute path to the ``.sif`` on the cluster. Required at submit.
    sif_image: str = ""
    #: ``apptainer`` or ``singularity``
    apptainer_bin: str = "apptainer"
    #: the python inside the image
    container_python: str = "python"
    #: bound as the container ``HOME``. Kit and shader caches land here and run to GBs, so
    #: it must be scratch, never an NFS home with a quota. Required at submit.
    cache_home: str = ""
    #: extra ``host:container`` mounts
    binds: List[str] = field(default_factory=list)

    def validate(self, cfg: Any) -> None:
        """Formats only, and only on values that are set.

        An unset field is not an error here — a local run never needs one. The submitter
        decides what it cannot live without.
        """
        if self.gpus < 1:
            raise ValueError(f"hpc.gpus must be >= 1, got {self.gpus}")
        if self.cpus < 1:
            raise ValueError(f"hpc.cpus must be >= 1, got {self.cpus}")
        if self.time and not TIME_RE.match(self.time):
            raise ValueError(
                f"hpc.time must be SLURM's [D-]HH:MM:SS, got {self.time!r} "
                "(e.g. '0-09:00:00' or '12:00:00')"
            )
        if self.signal and not SIGNAL_RE.match(self.signal):
            raise ValueError(
                f"hpc.signal must be SIG@seconds, got {self.signal!r} (e.g. 'TERM@300')"
            )
        for index, bind in enumerate(self.binds):
            if not bind or any(character.isspace() for character in bind):
                raise ValueError(
                    f"hpc.binds[{index}] must be a whitespace-free 'host:container' mount, "
                    f"got {bind!r}"
                )

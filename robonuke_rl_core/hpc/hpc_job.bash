#!/usr/bin/env bash
#SBATCH --job-name=rnk_job
#SBATCH --time=0-09:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --signal=TERM@300
#
# The container entry point for one package job.
#
# Every #SBATCH header above is a FALLBACK only: the submitter passes the real resources as
# sbatch flags, which win. They exist so the script is still submittable by hand.
#
# sbatch SPOOLS this file, so by the time it runs it is a copy in SLURM's spool directory
# with no siblings. It can therefore source nothing and must be self-contained. Everything
# it needs arrives as exported RNK_* variables and as argv (the python command to run).
#
#   RNK_PKG_ROOT        the package clone on the cluster (bound OVER the image's install)
#   RNK_PROJECT_ROOT    the project clone (bound, used as cwd; never installed)
#   RNK_SIF             absolute path to the .sif
#   RNK_APPTAINER_BIN   apptainer | singularity
#   RNK_CACHE_HOME      bound as the container HOME: Kit and shader caches, GBs, scratch
#   RNK_BINDS           extra "host:container" mounts, comma-separated (may be empty)
#   RNK_PYTHON          the python inside the image
#   RNK_CHAIN_SCRIPT    optional: run this in-container wrapper instead of exec'ing argv

set -Eeuo pipefail
trap 'echo "[hpc_job] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

: "${RNK_PYTHON:=python}"
: "${RNK_APPTAINER_BIN:=apptainer}"
: "${RNK_BINDS:=}"
: "${RNK_CHAIN_SCRIPT:=}"

say() { echo "[hpc_job] $*"; }

# ------------------------------------------------------------------ 1. validate the inputs
require_dir() {
    if [[ -z "${2:-}" || ! -d "${2}" ]]; then
        echo "[hpc_job] $1 must be a directory, got '${2:-<unset>}'" >&2
        exit 2
    fi
}
require_dir RNK_PKG_ROOT "${RNK_PKG_ROOT:-}"
require_dir RNK_PROJECT_ROOT "${RNK_PROJECT_ROOT:-}"
require_dir RNK_CACHE_HOME "${RNK_CACHE_HOME:-}"

if [[ -z "${RNK_SIF:-}" || ! -f "${RNK_SIF}" ]]; then
    echo "[hpc_job] RNK_SIF must be an existing .sif, got '${RNK_SIF:-<unset>}'" >&2
    exit 2
fi
if ! command -v "${RNK_APPTAINER_BIN}" >/dev/null 2>&1; then
    echo "[hpc_job] '${RNK_APPTAINER_BIN}' is not on PATH" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "[hpc_job] no command given; the submitter passes the python argv after the script" >&2
    exit 2
fi

# ------------------------------------------------------------------ 2. wandb mode
# An unattended job must never die over logging. No key -> offline, so the run is still
# recorded locally and can be synced later. A short key is a bad paste, which must warn
# rather than kill training. An explicit WANDB_MODE always wins.
if [[ -n "${WANDB_MODE:-}" ]]; then
    say "wandb mode: ${WANDB_MODE} (set explicitly)"
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=offline
    say "wandb mode: offline (no WANDB_API_KEY in the job environment)"
elif [[ ${#WANDB_API_KEY} -lt 40 ]]; then
    export WANDB_MODE=offline
    say "WARNING: WANDB_API_KEY is only ${#WANDB_API_KEY} chars; a key is 40. Going offline."
else
    say "wandb mode: online"
fi

# ------------------------------------------------------------------ 3. binds
# Both repos are bound at their own paths and reached through PYTHONPATH -- the pattern the
# existing image already uses for its project repo. Nothing is installed into the image, so
# any image carrying the stack works, and updating either repo on the cluster is a git pull.
binds=("${RNK_PKG_ROOT}:${RNK_PKG_ROOT}" "${RNK_PROJECT_ROOT}:${RNK_PROJECT_ROOT}")
if [[ -n "${RNK_BINDS}" ]]; then
    IFS=',' read -r -a extra <<< "${RNK_BINDS}"
    for mount in "${extra[@]}"; do
        [[ -n "${mount}" ]] && binds+=("${mount}")
    done
fi
if [[ -n "${RNK_CHAIN_SCRIPT}" ]]; then
    if [[ ! -f "${RNK_CHAIN_SCRIPT}" ]]; then
        echo "[hpc_job] RNK_CHAIN_SCRIPT '${RNK_CHAIN_SCRIPT}' is not a file" >&2
        exit 2
    fi
    binds+=("${RNK_CHAIN_SCRIPT}:/opt/hpc_job_chain.bash")
fi

bind_args=()
for mount in "${binds[@]}"; do
    bind_args+=(--bind "${mount}")
done

say "image      : ${RNK_SIF}"
say "package    : ${RNK_PKG_ROOT} (on PYTHONPATH)"
say "project    : ${RNK_PROJECT_ROOT} (cwd)"
say "cache home : ${RNK_CACHE_HOME}"
say "command    : $*"

# ------------------------------------------------------------------ 4. run
cd "${RNK_PROJECT_ROOT}"

# exec, not a subshell: SLURM's --signal=TERM@300 must reach the training process itself,
# so it gets the warning rather than bash getting it and the job dying mid-write.
if [[ -n "${RNK_CHAIN_SCRIPT}" ]]; then
    exec "${RNK_APPTAINER_BIN}" exec --nv --writable-tmpfs \
        --home "${RNK_CACHE_HOME}:/root" \
        "${bind_args[@]}" \
        --env PYTHONPATH="${RNK_PKG_ROOT}:${RNK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        --env OMNI_KIT_ACCEPT_EULA=YES \
        --env TORCHDYNAMO_DISABLE=1 \
        --env PYTHONUNBUFFERED=1 \
        "${RNK_SIF}" bash /opt/hpc_job_chain.bash "$@"
fi

exec "${RNK_APPTAINER_BIN}" exec --nv --writable-tmpfs \
    --home "${RNK_CACHE_HOME}:/root" \
    "${bind_args[@]}" \
    --env PYTHONPATH="${RNK_PKG_ROOT}:${RNK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env TORCHDYNAMO_DISABLE=1 \
    --env PYTHONUNBUFFERED=1 \
    "${RNK_SIF}" "$@"

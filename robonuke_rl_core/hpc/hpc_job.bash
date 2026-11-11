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
#   RNK_SIF_SM70        optional: the .sif to use instead when the GPU is compute capability
#                       7.0 (a V100) -- the cu128 torch wheels ship no sm_70 kernels
#   RNK_APPTAINER_BIN   apptainer | singularity
#   RNK_CACHE_HOME      bound as the container HOME: Kit and shader caches, GBs, scratch
#   RNK_BINDS           extra "host:container" mounts, comma-separated (may be empty)
#   RNK_EXTRA_REPOS     further repos, comma-separated (may be empty): each is bound at its
#                       own path and appended to the container PYTHONPATH after the two above
#   RNK_PYTHON          the python inside the image
#   RNK_CHAIN_SCRIPT    optional: run this in-container wrapper instead of exec'ing argv

set -Eeuo pipefail
trap 'echo "[hpc_job] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

: "${RNK_PYTHON:=python}"
: "${RNK_APPTAINER_BIN:=apptainer}"
: "${RNK_SIF_SM70:=}"
: "${RNK_BINDS:=}"
: "${RNK_EXTRA_REPOS:=}"
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

repos=()
if [[ -n "${RNK_EXTRA_REPOS}" ]]; then
    IFS=',' read -r -a repos <<< "${RNK_EXTRA_REPOS}"
    for repo in "${repos[@]}"; do
        if [[ -n "${repo}" ]]; then
            require_dir "RNK_EXTRA_REPOS entry" "${repo}"
        fi
    done
fi

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
# rather than kill training. An explicit WANDB_MODE always wins -- in the package too:
# logging.py lets the environment's WANDB_MODE override the config's wandb.mode.
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

# ------------------------------------------------------------------ 3. the GPU
# Two things only the compute node can tell. A MIG slice has no graphics, so Isaac Sim's Kit
# cannot start on one: refuse here, naming the fix, rather than let Kit die inside the
# container with a far less readable error. And a V100 (compute capability 7.0) runs none of
# the cu128 image's kernels, so when the submitter exported an sm_70 image it replaces
# RNK_SIF on such a node.
image="${RNK_SIF}"
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_list="$(nvidia-smi -L 2>&1 || true)"
    gpu_caps="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 || true)"
    say "gpus       :"
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then say "  ${line}"; fi
    done <<< "${gpu_list}"
    while IFS= read -r line; do
        if [[ -n "${line}" ]]; then say "  ${line}  (name, compute capability)"; fi
    done <<< "${gpu_caps}"

    if grep -q "MIG" <<< "${gpu_list}"; then
        echo "[hpc_job] this job landed on a MIG slice. A MIG device has no graphics, so Isaac" \
            "Sim's Kit cannot start on it. Set hpc.constraint to a node feature that lands the" \
            "job on whole GPUs (e.g. hpc.constraint=vram80g|vram140g) and resubmit." >&2
        exit 2
    fi

    first_cap="$(head -n 1 <<< "${gpu_caps}" | awk -F',' '{gsub(/ /, "", $NF); print $NF}')"
    if [[ "${first_cap}" == "7.0" ]]; then
        if [[ -n "${RNK_SIF_SM70}" ]]; then
            image="${RNK_SIF_SM70}"
            say "compute capability 7.0 (V100): using ${image} instead of ${RNK_SIF}, because" \
                "the cu128 torch wheels ship no sm_70 kernels"
        else
            say "WARNING: compute capability 7.0 (V100) and no RNK_SIF_SM70 (hpc.sif_image_sm70):" \
                "the cu128 image carries no sm_70 kernels, so torch may not run here. Continuing."
        fi
    fi
else
    say "WARNING: nvidia-smi is not on PATH; cannot log the GPU or check for a MIG slice"
fi

if [[ ! -f "${image}" ]]; then
    echo "[hpc_job] the chosen image '${image}' is not a file" >&2
    exit 2
fi
# hpc-share purges files idle for 90 days; a touch per job keeps a working image alive.
# Someone else's image, or a read-only share, is not a reason to fail the job.
touch "${image}" 2>/dev/null || true

# ------------------------------------------------------------------ 4. binds
# Both repos are bound at their own paths and reached through PYTHONPATH -- the pattern the
# existing image already uses for its project repo. Nothing is installed into the image, so
# any image carrying the stack works, and updating either repo on the cluster is a git pull.
# An extra repo (RNK_EXTRA_REPOS) follows the same pattern, after the two.
binds=("${RNK_PKG_ROOT}:${RNK_PKG_ROOT}" "${RNK_PROJECT_ROOT}:${RNK_PROJECT_ROOT}")
extra_pythonpath=""
for repo in "${repos[@]}"; do
    if [[ -n "${repo}" ]]; then
        binds+=("${repo}:${repo}")
        extra_pythonpath="${extra_pythonpath}:${repo}"
    fi
done
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

# Deduplicate by destination: when the package and project clones are the same directory
# (running a launcher from the package repo) apptainer warns "destination is already in the
# mount point list", and a warning nobody needs is a warning that hides a real one.
bind_args=()
seen_destinations=":"
for mount in "${binds[@]}"; do
    destination="${mount##*:}"
    case "${seen_destinations}" in
        *":${destination}:"*) continue ;;
    esac
    seen_destinations="${seen_destinations}${destination}:"
    bind_args+=(--bind "${mount}")
done

# The container PYTHONPATH is exactly package:project[:extra...]. The login shell's
# PYTHONPATH is NOT appended: it names host installs the container cannot see -- or, through
# a bind, can see and then shadows the image's stack with.
container_pythonpath="${RNK_PKG_ROOT}:${RNK_PROJECT_ROOT}${extra_pythonpath}"

say "image      : ${image}"
say "package    : ${RNK_PKG_ROOT} (on PYTHONPATH)"
say "project    : ${RNK_PROJECT_ROOT} (cwd)"
if [[ -n "${extra_pythonpath}" ]]; then
    say "extra repos: ${extra_pythonpath#:} (on PYTHONPATH, after the two)"
fi
say "cache home : ${RNK_CACHE_HOME}"
say "command    : $*"

# ------------------------------------------------------------------ 5. run
cd "${RNK_PROJECT_ROOT}"

# exec, not a subshell: SLURM's --signal=TERM@300 must reach the training process itself,
# so it gets the warning rather than bash getting it and the job dying mid-write.
if [[ -n "${RNK_CHAIN_SCRIPT}" ]]; then
    exec "${RNK_APPTAINER_BIN}" exec --nv --writable-tmpfs \
        --home "${RNK_CACHE_HOME}:/root" \
        "${bind_args[@]}" \
        --env PYTHONPATH="${container_pythonpath}" \
        --env OMNI_KIT_ACCEPT_EULA=YES \
        --env PYTHONUNBUFFERED=1 \
        "${image}" bash /opt/hpc_job_chain.bash "$@"
fi

exec "${RNK_APPTAINER_BIN}" exec --nv --writable-tmpfs \
    --home "${RNK_CACHE_HOME}:/root" \
    "${bind_args[@]}" \
    --env PYTHONPATH="${container_pythonpath}" \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env PYTHONUNBUFFERED=1 \
    "${image}" "$@"

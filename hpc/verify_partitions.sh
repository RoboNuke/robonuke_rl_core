#!/usr/bin/env bash
#
# Prove the images on every partition: one smoke job per partition, through the REAL job
# path (hpc_job.bash: image switch, MIG guard, binds, cache home), then one table.
#
#   hpc/verify_partitions.sh                      # dgxh dgx2 tiamat eecs2
#   hpc/verify_partitions.sh dgxh eecs2           # a subset
#   TIERS=abc hpc/verify_partitions.sh dgx2       # fewer tiers
#
# Run on a LOGIN node from the package clone (`hpc/install_hpc.sh` does, as its last step).
# Results: ${RNK_ROOT}/verify/<timestamp>/<partition>/{job.out,tier_*.log,summary.json}.
#
# Tier e (camera + mp4) needs RT cores: expected to fail on dgxh (H100) and dgx2 (V100),
# reported as such; the required tiers (abcd by default) decide pass/fail.
#
# Every knob is env-overridable.

set -Eeuo pipefail
trap 'echo "[verify] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/.." && pwd)"

: "${SHARE:=${HOME}/hpc-share}"
: "${RNK_ROOT:=${SHARE}/robonuke}"
: "${ACCOUNT:=virl-grp}"
: "${SIF:=${RNK_ROOT}/images/general-cu128.sif}"
: "${SIF_SM70:=${RNK_ROOT}/images/general-cu126.sif}"
: "${CACHE_HOME:=${RNK_ROOT}/cache_home}"
: "${TIERS:=abcde}"
: "${REQUIRED:=abcd}"
: "${TIME:=1:30:00}"
: "${CPUS:=8}"
: "${MEM:=32G}"
# keep dgxh jobs off the MIG-sliced node (dgxh-1): whole-GPU features only
: "${CONSTRAINT_dgxh:=vram80g|vram140g}"
# 0: submit and return; 1: wait and print the table
: "${WAIT:=1}"
: "${POLL_SECONDS:=60}"
: "${MAX_WAIT_MINUTES:=240}"

PARTITIONS=("$@")
[[ ${#PARTITIONS[@]} -eq 0 ]] && PARTITIONS=(dgxh dgx2 tiamat eecs2)

say() { echo "[verify] $*"; }

for f in "${SIF}" "${PKG_ROOT}/robonuke_rl_core/hpc/hpc_job.bash" "${PKG_ROOT}/hpc/smoke/smoke.py" \
         "${PKG_ROOT}/tests/gpu_forge.yaml"; do
    [[ -f "${f}" ]] || { echo "[verify] missing ${f}" >&2; exit 2; }
done
if [[ -n "${SIF_SM70}" && ! -f "${SIF_SM70}" ]]; then
    say "WARNING: ${SIF_SM70} does not exist; V100 jobs will run the cu128 image and tier a will fail there"
    SIF_SM70=""
fi
command -v sbatch >/dev/null 2>&1 || { echo "[verify] sbatch not on PATH (module load slurm?)" >&2; exit 2; }
mkdir -p "${CACHE_HOME}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${RNK_ROOT}/verify/${STAMP}"
mkdir -p "${OUT_ROOT}"
# The container sees only what is bound: the package, the project and HOME. The results
# directory is none of those, so bind it (by its real path; ~/hpc-share is a symlink) or
# every log the smoke test writes lands in the job's throwaway overlay and is gone.
OUT_ROOT="$(readlink -f "${OUT_ROOT}")"
say "image      : ${SIF}"
say "sm70 image : ${SIF_SM70:-<none>}"
say "cache home : ${CACHE_HOME}"
say "package    : ${PKG_ROOT}"
say "results    : ${OUT_ROOT}"

# The job environment hpc_job.bash expects (what submit.py exports for a real job).
export RNK_PKG_ROOT="${PKG_ROOT}"
export RNK_PROJECT_ROOT="${PKG_ROOT}"
export RNK_SIF="${SIF}"
export RNK_SIF_SM70="${SIF_SM70}"
export RNK_APPTAINER_BIN="${RNK_APPTAINER_BIN:-apptainer}"
export RNK_CACHE_HOME="${CACHE_HOME}"
export RNK_BINDS="${OUT_ROOT}:${OUT_ROOT}"
export RNK_EXTRA_REPOS=""
export RNK_PYTHON="python"
export WANDB_MODE=offline

declare -A JOBS
for part in "${PARTITIONS[@]}"; do
    out="${OUT_ROOT}/${part}"
    mkdir -p "${out}"
    constraint_var="CONSTRAINT_${part}"
    constraint="${!constraint_var:-}"
    args=(-A "${ACCOUNT}" -p "${part}" --gres=gpu:1 -c "${CPUS}" --mem "${MEM}" -t "${TIME}"
          -J "rnk_smoke_${part}" -o "${out}/job.out" -e "${out}/job.out" --export=ALL --parsable)
    [[ -n "${constraint}" ]] && args+=("--constraint=${constraint}")
    job_id="$(sbatch "${args[@]}" "${PKG_ROOT}/robonuke_rl_core/hpc/hpc_job.bash" \
        python hpc/smoke/smoke.py --out "${out}" --tiers "${TIERS}" --required "${REQUIRED}")"
    JOBS["${part}"]="${job_id}"
    say "submitted ${part}: job ${job_id}${constraint:+ (constraint ${constraint})}"
done

if [[ "${WAIT}" != "1" ]]; then
    say "not waiting (WAIT=0). Summaries will land in ${OUT_ROOT}/<partition>/summary.json"
    exit 0
fi

# ------------------------------------------------------------------ wait
ids="$(IFS=,; echo "${JOBS[*]}")"
deadline=$(( $(date +%s) + MAX_WAIT_MINUTES * 60 ))
while true; do
    pending="$(squeue -h -j "${ids}" -o "%i %P %T %M %R" 2>/dev/null || true)"
    [[ -z "${pending}" ]] && break
    if (( $(date +%s) > deadline )); then
        say "WARNING: still queued/running after ${MAX_WAIT_MINUTES} min; leaving them. Re-run the table with:"
        say "  python3 ${HERE}/smoke/report.py ${OUT_ROOT}"
        break
    fi
    say "$(date +%H:%M) waiting on: $(echo "${pending}" | tr '\n' ';')"
    sleep "${POLL_SECONDS}"
done

# ------------------------------------------------------------------ the table
python3 "${HERE}/smoke/report.py" "${OUT_ROOT}"

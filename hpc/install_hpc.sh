#!/usr/bin/env bash
#
# One-shot install of the package's HPC setup on the OSU CoE cluster. Run on a LOGIN node,
# from a clone of this package:
#
#   bash hpc/install_hpc.sh                  # everything: preflight, layout, repos, images, verify
#   bash hpc/install_hpc.sh --no-build       # skip the image builds (images already there)
#   bash hpc/install_hpc.sh --no-verify      # skip the per-partition smoke jobs
#   bash hpc/install_hpc.sh --no-repos       # leave the research clones alone
#   bash hpc/install_hpc.sh --rebuild        # build even when a verified image exists
#
# Idempotent: re-running does only what is missing. Everything large lands under
# ${RNK_ROOT} (default ~/hpc-share/robonuke): the 25 GB home quota is not touched.
#
# What it sets up
#   1. preflight   apptainer, your Slurm accounts, the partitions' GPUs and features, quotas
#   2. layout      ${RNK_ROOT}/{images,cache_home,build,apptainer_cache,verify,runs}
#   3. repos       the three research clones under ${RNK_REPOS} (default ~/hpc-share),
#                  with runs/ wandb/ exp_logs/ redirected to ${RNK_ROOT}/runs/<repo>/,
#                  and a small conda env (${LAUNCH_ENV}) for the launchers on the login node
#   4. images      general-cu128.sif and general-cu126.sif, built on a compute node by
#                  hpc/build_image.sbatch (one job each, ~15 min)
#   5. config      ${RNK_ROOT}/hpc.env and the `hpc:` YAML block to paste into a project
#   6. verify      hpc/verify_partitions.sh: one smoke job per partition, one table
#
# Every knob is env-overridable.

set -Eeuo pipefail
trap 'echo "[install] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/.." && pwd)"

DO_BUILD=1 DO_VERIFY=1 DO_REPOS=1 REBUILD=0
for arg in "$@"; do
    case "${arg}" in
        --no-build) DO_BUILD=0 ;;
        --no-verify) DO_VERIFY=0 ;;
        --no-repos) DO_REPOS=0 ;;
        --rebuild) REBUILD=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "[install] unknown argument ${arg}" >&2; exit 2 ;;
    esac
done

: "${SHARE:=${HOME}/hpc-share}"
: "${RNK_ROOT:=${SHARE}/robonuke}"
: "${RNK_REPOS:=${SHARE}}"
: "${ACCOUNT:=virl-grp}"
# the partitions this account can use with a GPU; `gpu` admits eecs/coeit only
: "${PARTITIONS:=dgxh,dgx2,tiamat,eecs2}"
# whole-GPU features of those partitions: keeps jobs off the MIG-sliced dgxh node
: "${CONSTRAINT:=vram80g|vram140g|v100|l40s|rtx6000}"
: "${BUILD_PARTITIONS:=dgxh,tiamat,eecs2}"
: "${FLAVORS:=cu128 cu126}"
: "${GITHUB_ORG:=git@github.com:RoboNuke}"
: "${REPOS:=robonuke_rl_core hybrid_demos Robonuke_Furniture_Bench}"
: "${CONDA_BIN:=${SHARE}/miniforge/bin/conda}"
: "${LAUNCH_ENV:=rnk-launch}"
: "${POLL_SECONDS:=60}"
: "${BUILD_MAX_MINUTES:=300}"

IMG_DIR="${RNK_ROOT}/images"
say() { echo "[install] $*"; }
hr() { echo; echo "================================================================ $*"; }

# ============================================================ 1. preflight
hr "1. preflight"
if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
fi
command -v apptainer >/dev/null 2>&1 || { echo "[install] apptainer is not on PATH" >&2; exit 2; }
say "apptainer  : $(apptainer --version)"
if ! command -v sbatch >/dev/null 2>&1; then
    module load slurm 2>/dev/null || true
fi
command -v sbatch >/dev/null 2>&1 || { echo "[install] sbatch is not on PATH; run from a login shell" >&2; exit 2; }
say "slurm      : $(sbatch --version)"
say "user       : ${USER} on $(hostname)"

say "Slurm associations (account | partition | qos):"
sacctmgr -nP show assoc "user=${USER}" format=account,partition,qos | sed 's/^/             /'
if ! sacctmgr -nP show assoc "user=${USER}" format=account | grep -qx "${ACCOUNT}"; then
    echo "[install] you have no association for account '${ACCOUNT}' (ACCOUNT=...)" >&2
    exit 2
fi
say "partitions ${PARTITIONS} (node | features | gres):"
sinfo -N -p "${PARTITIONS}" -o "%P %N %f %G" | sed 's/^/             /'
for part in ${PARTITIONS//,/ }; do
    allow="$(scontrol show partition "${part}" 2>/dev/null | grep -o 'AllowAccounts=[^ ]*' || true)"
    case "${allow}" in
        ""|*=ALL|*"${ACCOUNT}"*) ;;
        *) say "WARNING: ${part} ${allow} does not include ${ACCOUNT}; jobs there will be refused" ;;
    esac
done

say "quotas:"
df -h "${HOME}" "${SHARE}" 2>/dev/null | sed 's/^/             /'
home_pct="$(df --output=pcent "${HOME}" | tail -1 | tr -dc '0-9')"
if (( home_pct >= 85 )); then
    say "WARNING: home is ${home_pct}% full (fixed 25 GB quota). Nothing here writes to it, but"
    say "         ~/.apptainer and ~/.cache are the usual culprits: move them to ${SHARE}."
fi
[[ -d "${SHARE}" ]] || { echo "[install] ${SHARE} does not exist (SHARE=...)" >&2; exit 2; }

# ============================================================ 2. layout
hr "2. layout under ${RNK_ROOT}"
mkdir -p "${RNK_ROOT}"/{images,cache_home,build,apptainer_cache,verify,runs}
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${RNK_ROOT}/apptainer_cache}"
say "created    : images/ cache_home/ build/ apptainer_cache/ verify/ runs/"
if ! grep -q "APPTAINER_CACHEDIR" "${HOME}/.bashrc" 2>/dev/null; then
    say "add to ~/.bashrc so apptainer never caches into the 25 GB home:"
    say "    export APPTAINER_CACHEDIR=${RNK_ROOT}/apptainer_cache"
fi

# ============================================================ 3. repos
if [[ "${DO_REPOS}" == "1" ]]; then
    hr "3. research clones under ${RNK_REPOS}"
    # GitHub's -T greeting exits 1 even on success, so under pipefail the pipeline would too
    github_greeting="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 || true)"
    if grep -q "successfully authenticated" <<< "${github_greeting}"; then
        say "github     : ssh key accepted"
    else
        say "WARNING: this host cannot authenticate to GitHub over ssh. Either add this host's"
        say "         key (~/.ssh/id_*.pub) to your GitHub account, or run the installer through"
        say "         an agent-forwarding login (ssh -A). Existing clones are left as they are."
    fi
    for repo in ${REPOS}; do
        dest="${RNK_REPOS}/${repo}"
        if [[ "${repo}" == "robonuke_rl_core" && "${dest}" -ef "${PKG_ROOT}" ]]; then
            say "${repo}: this clone (${PKG_ROOT}), left as is"
        elif [[ -d "${dest}/.git" ]]; then
            if git -C "${dest}" pull --ff-only >/dev/null 2>&1; then
                say "${repo}: updated ($(git -C "${dest}" log --oneline -n1))"
            else
                say "WARNING: ${repo}: could not fast-forward (auth, or local changes); at $(git -C "${dest}" log --oneline -n1)"
            fi
        else
            if git clone "${GITHUB_ORG}/${repo}.git" "${dest}" 2>&1 | sed 's/^/             /'; then
                say "${repo}: cloned to ${dest}"
            else
                say "WARNING: ${repo}: clone failed; clone it yourself to ${dest}"
                continue
            fi
        fi
        # outputs off the quota'd home and out of the clone: runs/ wandb/ exp_logs/ -> RNK_ROOT
        for sub in runs wandb exp_logs; do
            target="${RNK_ROOT}/runs/${repo}/${sub}"
            if [[ -L "${dest}/${sub}" ]]; then
                continue
            elif [[ -e "${dest}/${sub}" ]]; then
                say "${repo}/${sub} exists and is not a symlink; leaving it (move it to ${target} yourself)"
            else
                mkdir -p "${target}"
                ln -s "${target}" "${dest}/${sub}"
                say "${repo}/${sub} -> ${target}"
            fi
        done
    done

    # a light python for the launchers on the login node (omegaconf, pyyaml, wandb). conda's
    # envs_dirs decide where the env lands, so ask conda for its python rather than guessing.
    if [[ -x "${CONDA_BIN}" ]]; then
        if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${LAUNCH_ENV}"; then
            say "creating conda env ${LAUNCH_ENV} for the launchers"
            "${CONDA_BIN}" create -y -q -n "${LAUNCH_ENV}" python=3.11 >/dev/null
        fi
        launch_python="$("${CONDA_BIN}" run -n "${LAUNCH_ENV}" python -c 'import sys; print(sys.executable)')"
        "${launch_python}" -m pip install -q omegaconf pyyaml wandb pytest
        if [[ -d "${RNK_REPOS}/robonuke_rl_core/.git" ]]; then
            "${launch_python}" -m pip install -q --no-deps -e "${RNK_REPOS}/robonuke_rl_core"
        else
            "${launch_python}" -m pip install -q --no-deps -e "${PKG_ROOT}"
        fi
        say "launcher python: ${launch_python}  (conda activate ${LAUNCH_ENV})"
    else
        say "WARNING: no conda at ${CONDA_BIN}; the launchers need omegaconf + pyyaml (+ wandb) on the login node"
    fi
fi

# ============================================================ 4. images
if [[ "${DO_BUILD}" == "1" ]]; then
    hr "4. images"
    export RNK_HPC_DIR="${HERE}"
    export SHARE RNK_ROOT
    declare -A BUILD_JOBS=()
    for flavor in ${FLAVORS}; do
        img="${IMG_DIR}/general-${flavor}.sif"
        if [[ -f "${img}.verified" && "${REBUILD}" != "1" ]]; then
            say "${flavor}: verified image present ($(du -h "${img}" | cut -f1)), skipping (--rebuild to force)"
            continue
        fi
        parts="${BUILD_PARTITIONS}"
        constraint="${CONSTRAINT}"
        if [[ "${flavor}" == "cu126" ]]; then
            parts="dgx2,${BUILD_PARTITIONS}"   # verify on a V100 when one is free
        fi
        job_id="$(sbatch --parsable -A "${ACCOUNT}" -p "${parts}" --constraint="${constraint}" \
            -o "${RNK_ROOT}/build/build_${flavor}_%j.out" -e "${RNK_ROOT}/build/build_${flavor}_%j.out" \
            --export=ALL "${HERE}/build_image.sbatch" "${flavor}")"
        BUILD_JOBS["${flavor}"]="${job_id}"
        say "${flavor}: build job ${job_id} on ${parts} -> ${RNK_ROOT}/build/build_${flavor}_${job_id}.out"
    done
    if (( ${#BUILD_JOBS[@]} > 0 )); then
        ids="$(IFS=,; echo "${BUILD_JOBS[*]}")"
        deadline=$(( $(date +%s) + BUILD_MAX_MINUTES * 60 ))
        while squeue -h -j "${ids}" 2>/dev/null | grep -q .; do
            if (( $(date +%s) > deadline )); then
                echo "[install] builds still running after ${BUILD_MAX_MINUTES} min; re-run with --no-build once they finish" >&2
                exit 3
            fi
            say "$(date +%H:%M) building: $(squeue -h -j "${ids}" -o '%j/%P/%T/%M' | tr '\n' ' ')"
            sleep "${POLL_SECONDS}"
        done
        for flavor in "${!BUILD_JOBS[@]}"; do
            img="${IMG_DIR}/general-${flavor}.sif"
            log="${RNK_ROOT}/build/build_${flavor}_${BUILD_JOBS[${flavor}]}.out"
            if [[ -f "${img}.verified" ]]; then
                say "${flavor}: built and verified ($(du -h "${img}" | cut -f1))"
            else
                echo "[install] ${flavor}: no verified image. Log tail (${log}):" >&2
                tail -n 30 "${log}" >&2 || true
                exit 3
            fi
        done
    fi
fi
for flavor in ${FLAVORS}; do
    [[ -f "${IMG_DIR}/general-${flavor}.sif" ]] || say "WARNING: ${IMG_DIR}/general-${flavor}.sif is missing"
done

# ============================================================ 5. config
hr "5. config"
cat > "${RNK_ROOT}/hpc.env" <<EOF
# written by hpc/install_hpc.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
RNK_ROOT=${RNK_ROOT}
RNK_REPOS=${RNK_REPOS}
ACCOUNT=${ACCOUNT}
PARTITIONS=${PARTITIONS}
CONSTRAINT=${CONSTRAINT}
SIF=${IMG_DIR}/general-cu128.sif
SIF_SM70=${IMG_DIR}/general-cu126.sif
CACHE_HOME=${RNK_ROOT}/cache_home
BINDS=${RNK_ROOT}:${RNK_ROOT}
APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}
EOF
say "wrote ${RNK_ROOT}/hpc.env"
cat <<EOF

The \`hpc:\` block for a project's configs/base/hpc.yaml (hybrid_demos adds extra_repos):

hpc:
  account: ${ACCOUNT}
  partitions: ${PARTITIONS}
  constraint: ${CONSTRAINT}
  sif_image: ${IMG_DIR}/general-cu128.sif
  sif_image_sm70: ${IMG_DIR}/general-cu126.sif
  cache_home: ${RNK_ROOT}/cache_home
  # the clones' runs/ wandb/ exp_logs/ are symlinks into here; the container sees only what
  # is bound, so without this a job's first mkdir('runs') fails on a dangling link
  binds:
    - ${RNK_ROOT}:${RNK_ROOT}
  # extra_repos:
  #   - ${RNK_REPOS}/Robonuke_Furniture_Bench

EOF

# ============================================================ 6. verify
if [[ "${DO_VERIFY}" == "1" ]]; then
    hr "6. verify every partition"
    SIF="${IMG_DIR}/general-cu128.sif" SIF_SM70="${IMG_DIR}/general-cu126.sif" \
    CACHE_HOME="${RNK_ROOT}/cache_home" ACCOUNT="${ACCOUNT}" RNK_ROOT="${RNK_ROOT}" \
        bash "${HERE}/verify_partitions.sh" ${PARTITIONS//,/ }
fi
say "done"

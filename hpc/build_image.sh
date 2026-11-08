#!/usr/bin/env bash
#
# Build one flavor of the `general` image from hpc/general.def.in and the lock files.
#
#   hpc/build_image.sh cu128     # H100 / H200 / L40S / RTX 6000
#   hpc/build_image.sh cu126     # V100 (dgx2): torch rebuilt with sm_70 kernels
#
# This is a COMPUTE-NODE script: submit nodes refuse or OOM the build (mksquashfs), so
# `hpc/install_hpc.sh` submits it through `hpc/build_image.sbatch`. Running it by hand:
#
#   srun -A virl-grp -p dgxh,tiamat,eecs2 --gres=gpu:1 -c 16 --mem 64G -t 4:00:00 --pty bash
#   hpc/build_image.sh cu128
#
# The build needs no root: this cluster has no subuid mapping, and apptainer falls back to a
# root-mapped user namespace with the %post run under the fakeroot command, which works
# (probed, and it is how the earlier Isaac Lab image here was built).
#
# Disk: the sandbox and the squashfs need ~70 GB of LOCAL scratch (APPTAINER_TMPDIR), and
# the layer cache (APPTAINER_CACHEDIR) must not be the 25 GB home -- hpc-share, always.
#
# VERIFY_ONLY=1 re-runs only the checks on an image that is already there.
# Every knob is env-overridable.

set -Eeuo pipefail
trap 'echo "[build] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAVOR="${1:-}"
case "${FLAVOR}" in
    cu128|cu126) ;;
    *) echo "usage: $(basename "$0") <cu128|cu126>" >&2; exit 2 ;;
esac

: "${SHARE:=${HOME}/hpc-share}"
: "${RNK_ROOT:=${SHARE}/robonuke}"
: "${IMG_DIR:=${RNK_ROOT}/images}"
: "${BUILD_DIR:=${RNK_ROOT}/build}"
: "${APPTAINER_BIN:=apptainer}"
: "${APPTAINER_CACHEDIR:=${RNK_ROOT}/apptainer_cache}"
: "${MINIFORGE_VERSION:=24.11.3-0}"
: "${BUILD_FLAGS:=--fakeroot}"
: "${MIN_TMP_GB:=70}"
# verify with a GPU after the build (needs one on this node); 0 skips
: "${VERIFY_GPU:=1}"
# also boot Kit headless in the verify step (a minute or two); 0 skips
: "${VERIFY_KIT:=1}"

LOCK="${HERE}/requirements-${FLAVOR}.lock"
ENV_LOCK="${HERE}/env.lock.env"
DEF_IN="${HERE}/general.def.in"
DEF="${BUILD_DIR}/general-${FLAVOR}.def"
IMG="${IMG_DIR}/general-${FLAVOR}.sif"

say() { echo "[build] $*"; }

for f in "${LOCK}" "${ENV_LOCK}" "${DEF_IN}"; do
    [[ -f "${f}" ]] || { echo "[build] missing ${f}; run hpc/export_env.sh locally and commit" >&2; exit 2; }
done
command -v "${APPTAINER_BIN}" >/dev/null 2>&1 || {
    echo "[build] ${APPTAINER_BIN} is not on PATH" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${ENV_LOCK}"
: "${PYTHON_VERSION:?env.lock.env has no PYTHON_VERSION}"
: "${ISAACLAB_REPO:?}" "${ISAACLAB_COMMIT:?}" "${ISAACSIM_VERSION:?}" "${LOCK_SHA256_16:?}"

# ---------------------------------------------------------------- local scratch for the build
# TmpDisk is 0 in slurm's view on every GPU node, so probe the candidates for real.
pick_tmp() {
    local candidate free_gb
    for candidate in "${APPTAINER_TMPDIR:-}" "${TMPDIR:-}" "/scratch/${USER}" "/scratch" "/tmp"; do
        [[ -z "${candidate}" ]] && continue
        mkdir -p "${candidate}" 2>/dev/null || continue
        [[ -w "${candidate}" ]] || continue
        free_gb=$(df -BG --output=avail "${candidate}" | tail -1 | tr -dc '0-9')
        if (( free_gb >= MIN_TMP_GB )); then
            echo "${candidate}"
            return 0
        fi
        say "skipping ${candidate}: ${free_gb} GB free, need ${MIN_TMP_GB}" >&2
    done
    return 1
}
TMP_BASE="$(pick_tmp)" || { echo "[build] no local scratch with ${MIN_TMP_GB} GB free" >&2; exit 2; }
export APPTAINER_TMPDIR="${TMP_BASE}/apptainer-build-${FLAVOR}-$$"
export APPTAINER_CACHEDIR
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}" "${IMG_DIR}" "${BUILD_DIR}"
trap 'rm -rf "${APPTAINER_TMPDIR}"' EXIT

say "flavor     : ${FLAVOR}"
say "host       : $(hostname)"
say "image      : ${IMG}"
say "lock       : ${LOCK} (${LOCK_SHA256_16})"
say "isaac lab  : ${ISAACLAB_COMMIT}"
say "tmpdir     : ${APPTAINER_TMPDIR}"
say "cachedir   : ${APPTAINER_CACHEDIR}"

# ---------------------------------------------------------------- render the definition
sed \
    -e "s|@FLAVOR@|${FLAVOR}|g" \
    -e "s|@LOCKFILE@|$(basename "${LOCK}")|g" \
    -e "s|@LOCK_PATH@|${LOCK}|g" \
    -e "s|@ENV_LOCK_PATH@|${ENV_LOCK}|g" \
    -e "s|@LOCK_SHA@|${LOCK_SHA256_16}|g" \
    -e "s|@PYTHON_VERSION@|${PYTHON_VERSION}|g" \
    -e "s|@ISAACLAB_REPO@|${ISAACLAB_REPO}|g" \
    -e "s|@ISAACLAB_COMMIT@|${ISAACLAB_COMMIT}|g" \
    -e "s|@ISAACSIM_VERSION@|${ISAACSIM_VERSION}|g" \
    -e "s|@MINIFORGE_VERSION@|${MINIFORGE_VERSION}|g" \
    -e "s|@BUILT_ON@|$(date -u +%Y-%m-%dT%H:%M:%SZ) $(hostname)|g" \
    "${DEF_IN}" > "${DEF}"
if grep -n "@[A-Z_]*@" "${DEF}"; then
    echo "[build] unrendered token in ${DEF}" >&2
    exit 2
fi
say "definition : ${DEF}"

# ---------------------------------------------------------------- build
# Into a temporary name, so a half-written image is never mistaken for a good one.
# VERIFY_ONLY=1 skips the build and re-runs the checks on the image that is there.
if [[ "${VERIFY_ONLY:-0}" == "1" && -f "${IMG}" ]]; then
    say "VERIFY_ONLY=1: not building, verifying ${IMG} ($(du -h "${IMG}" | cut -f1))"
else
    say "building (pip pulls ~12 GB of wheels; mksquashfs is the slow part)"
    SECONDS=0
    # shellcheck disable=SC2086
    "${APPTAINER_BIN}" build ${BUILD_FLAGS} "${IMG}.building" "${DEF}"
    mv -f "${IMG}.building" "${IMG}"
    say "built in $((SECONDS / 60)) min: ${IMG} ($(du -h "${IMG}" | cut -f1))"
fi

# ---------------------------------------------------------------- verify
# A built image that cannot run the stack is worse than none: the failure would surface
# inside a queued job, hours later. %test covers the imports without a GPU.
say "verify: %test"
"${APPTAINER_BIN}" test "${IMG}"

if [[ "${VERIFY_GPU}" == "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        say "WARNING: no nvidia-smi on this node; skipping the GPU verify (VERIFY_GPU=1)"
    else
        VERIFY_HOME="${BUILD_DIR}/verify_home_${FLAVOR}"
        mkdir -p "${VERIFY_HOME}"
        say "verify: torch sees the GPU and has kernels for it"
        "${APPTAINER_BIN}" exec --nv --writable-tmpfs --home "${VERIFY_HOME}:/root" "${IMG}" python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False under --nv"
def kernels_for(capability, archs):
    """The wheel entry that runs on this GPU: a cubin of the same major architecture and a
    minor at or below the device's (sm_86 runs on an sm_89 L40S), else PTX it can JIT."""
    major, minor = capability
    cubins = []
    for arch in archs:
        kind, number = arch.split("_")
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            cubins.append((arch_minor, arch))
    if cubins:
        return max(cubins)[1]
    for arch in archs:
        kind, number = arch.split("_")
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return arch
    return None
capability = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
usable = kernels_for(capability, archs)
print("[verify] gpu", torch.cuda.get_device_name(0), f"sm_{capability[0]}{capability[1]}", "| wheel archs", archs, "| runs", usable)
if usable is None:
    raise SystemExit(f"[verify] this torch has no kernels for sm_{capability[0]}{capability[1]}; this flavor is wrong for this GPU")
x = torch.randn(512, 512, device="cuda")
print("[verify] matmul ok", float((x @ x).sum()))
PYEOF
        if [[ "${VERIFY_KIT}" == "1" ]]; then
            say "verify: Kit boots headless (this takes a minute or two the first time)"
            "${APPTAINER_BIN}" exec --nv --writable-tmpfs --home "${VERIFY_HOME}:/root" "${IMG}" python - <<'PYEOF'
import os, sys, time
t0 = time.time()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print(f"[verify] kit up in {time.time() - t0:.0f}s")
sys.stdout.flush()
os._exit(0)  # Kit's shutdown can hang; the boot is what we wanted to see
PYEOF
        fi
    fi
fi

{
    echo "image=${IMG}"
    echo "flavor=${FLAVOR}"
    echo "lock_sha256_16=${LOCK_SHA256_16}"
    echo "isaaclab_commit=${ISAACLAB_COMMIT}"
    echo "verified_on=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "verified_host=$(hostname)"
    echo "verified_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
} > "${IMG}.verified"
say "done: ${IMG}"
say "stamp: ${IMG}.verified"

#!/usr/bin/env bash
#
# Build the Apptainer image the HPC jobs run in.
#
# YOU PROBABLY DO NOT NEED THIS. The jobs install nothing into the image -- they bind both
# repos and put them on PYTHONPATH -- so any image carrying the stack works, and today that is
# the existing ghvic.sif built for generalized_hybrid_vic_action_space. Point
# hpc.sif_image at it (see examples/hpc.yaml) and skip this script.
#
# This exists for the day that image is no longer enough: a new Isaac Lab, or a dependency the
# package needs that it does not have.
#
# THE RULE: the image bakes the STACK, never the research code.
#
#   baked   Ubuntu 22.04 + CUDA 12.8, Python 3.11, torch 2.7.0+cu128, Isaac Sim 5.1.0
#           wheels, Isaac Lab at a pinned commit, and the package's DEPENDENCIES (wandb,
#           pandas, pyarrow, imageio, skrl, omegaconf...) via a throwaway clone.
#   bound   at runtime the job binds both repos at their own paths and puts them on
#           PYTHONPATH. Nothing here is what makes `import robonuke_rl_core` work.
#
# This is a compute-node script, not package code: login nodes OOM running mksquashfs.
#
#   srun -A <acct> -p <part> -c 16 --mem 64G -t 4:00:00 --pty bash
#   SHARE=$HOME/hpc-share ./hpc/build_image.sh
#
# Every knob is env-overridable.

set -Eeuo pipefail
trap 'echo "[build] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

: "${SHARE:=${HOME}/hpc-share}"
: "${IMG:=${SHARE}/robonuke_rl_core.sif}"
: "${DEF:=${SHARE}/robonuke_rl_core.def}"
: "${APPTAINER_BIN:=apptainer}"
# APPTAINER_TMPDIR must be LOCAL disk: on Lustre the build is glacial and sometimes corrupt
: "${APPTAINER_TMPDIR:=/tmp/apptainer-build-$$}"
: "${BASE_IMAGE:=docker://nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04}"
: "${PYTHON_VERSION:=3.11}"
: "${TORCH_SPEC:=torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128}"
: "${ISAACSIM_SPEC:=isaacsim[all,extscache]==5.1.0}"
: "${ISAACSIM_INDEX:=https://pypi.nvidia.com}"
: "${ISAACLAB_REPO:=https://github.com/isaac-sim/IsaacLab.git}"
# pin the commit: an Isaac Lab bump is a deliberate act, never a side effect of rebuilding
: "${ISAACLAB_COMMIT:=v2.3.0}"
: "${IMAGE_PKG_PATH:=/opt/robonuke_rl_core_deps}"  # throwaway clone: deps only
: "${PKG_REPO:=https://github.com/RoboNuke/robonuke_rl_core.git}"
: "${PKG_COMMIT:=main}"

say() { echo "[build] $*"; }

command -v "${APPTAINER_BIN}" >/dev/null 2>&1 || {
    echo "[build] ${APPTAINER_BIN} is not on PATH; load the module first" >&2
    exit 2
}
mkdir -p "${SHARE}" "${APPTAINER_TMPDIR}"
export APPTAINER_TMPDIR
say "image      : ${IMG}"
say "tmpdir     : ${APPTAINER_TMPDIR} (must be local disk, not Lustre)"
say "isaac lab  : ${ISAACLAB_COMMIT}"
say "pkg path   : ${IMAGE_PKG_PATH} (dependencies only; jobs use PYTHONPATH)"

cat > "${DEF}" <<DEFEOF
Bootstrap: docker
From: ${BASE_IMAGE##docker://}

%post
    set -eux
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \\
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \\
        python3-pip git curl ca-certificates build-essential \\
        libglu1-mesa libxi6 libxrandr2 libxinerama1 libxcursor1 libgl1 libglib2.0-0 \\
        libsm6 libice6 libxt6 libxrender1 libxext6 ffmpeg
    rm -rf /var/lib/apt/lists/*
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    ln -sf /usr/bin/python3 /usr/bin/python
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

    # torch first: the Isaac Sim wheels resolve against it
    python -m pip install --no-cache-dir ${TORCH_SPEC}

    # Isaac Sim wheels from the public NVIDIA index
    python -m pip install --no-cache-dir --extra-index-url ${ISAACSIM_INDEX} '${ISAACSIM_SPEC}'

    # Isaac Lab, editable at a pinned commit
    git clone ${ISAACLAB_REPO} /opt/IsaacLab
    cd /opt/IsaacLab
    git checkout ${ISAACLAB_COMMIT}
    ./isaaclab.sh --install none

    # a throwaway clone, purely to make pip resolve and install the package's DEPENDENCIES.
    # Jobs never import from here -- they bind the cluster's checkout and put it on
    # PYTHONPATH -- so this copy going stale does not matter.
    git clone ${PKG_REPO} ${IMAGE_PKG_PATH}
    cd ${IMAGE_PKG_PATH}
    git checkout ${PKG_COMMIT}
    python -m pip install --no-cache-dir -e .

%environment
    export OMNI_KIT_ACCEPT_EULA=YES
    export TORCHDYNAMO_DISABLE=1
    export PYTHONUNBUFFERED=1

%labels
    isaaclab_commit ${ISAACLAB_COMMIT}
    package_path ${IMAGE_PKG_PATH}
DEFEOF

say "definition written to ${DEF}"
say "building (this takes a while; mksquashfs is the slow part)"
"${APPTAINER_BIN}" build --fakeroot "${IMG}" "${DEF}"

# ------------------------------------------------------------------ verify
# A built image that cannot import the stack is worse than no image: the failure surfaces
# inside a queued job, hours later. So prove it here, before anyone submits anything.
say "verifying"
"${APPTAINER_BIN}" exec --nv "${IMG}" python - <<'PYEOF'
import sys

print("python        ", sys.version.split()[0])
import torch

print("torch         ", torch.__version__, "| cuda build", torch.version.cuda)
import isaaclab

print("isaaclab      ", getattr(isaaclab, "__version__", "<no __version__>"))
import robonuke_rl_core

print("robonuke_rl_core", robonuke_rl_core.__version__)
PYEOF

say "done: ${IMG}"
say "point hpc.sif_image at it, and hpc.cache_home at scratch with room for GBs of caches"

#!/usr/bin/env bash
#SBATCH --job-name=rnk_build_image
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#
# SLURM wrapper around hpc/build_image.sh: one job builds one flavor.
#
#   sbatch -A virl-grp -p dgxh,tiamat,eecs2 -o build_cu128_%j.out hpc/build_image.sbatch cu128
#
# Account and partition are the submitter's (install_hpc.sh passes them); the headers above
# are the resources the build needs. A GPU is requested only so the verify step can prove
# torch and Kit run; the build itself is CPU and disk.
#
# sbatch spools this file, so it cannot find build_image.sh by its own location: the
# submitter exports RNK_HPC_DIR (the package's hpc/ directory). Everything else is
# build_image.sh's own env knobs, carried through sbatch --export=ALL.

set -Eeuo pipefail
: "${RNK_HPC_DIR:?export RNK_HPC_DIR=<package>/hpc (install_hpc.sh does this)}"
echo "[build_image.sbatch] $(date) on $(hostname), flavor ${1:-?}, hpc dir ${RNK_HPC_DIR}"
exec bash "${RNK_HPC_DIR}/build_image.sh" "$@"

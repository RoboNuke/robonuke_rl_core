#!/usr/bin/env bash
#
# Train, then evaluate -- inside the container, as one job.
#
# A train-then-eval job cannot be a single exec'd python, so this wrapper runs the training
# command and then one eval per agent. It is bind-mounted into the image by hpc_job.bash and
# run with `bash`, so it needs no exec bit on a shared filesystem.
#
# Inputs (exported by the submitter, forwarded through hpc_job.bash):
#   RNK_PYTHON          python inside the image
#   RNK_EVAL_CONFIG     the --eval_config to run after training
#   RNK_EVAL_RUNS       comma-separated run paths, "entity/project/group_ai"
#   RNK_EVAL_CHECKPOINT best | <step> | <file name>   (default: best)
# argv: the training command, run as given.
#
# The eval half is deliberately NON-FATAL. Training that finished is the expensive thing; a
# wandb hiccup or a missing checkpoint in the eval step must not turn a completed run into a
# failed job. Each failure is warned about and the next eval is tried.

set -Eeuo pipefail
trap 'echo "[chain] FAILED at ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2' ERR

: "${RNK_PYTHON:=python}"
: "${RNK_EVAL_CHECKPOINT:=best}"
: "${RNK_EVAL_CONFIG:=}"
: "${RNK_EVAL_RUNS:=}"

say() { echo "[chain] $*"; }

child=""
forward() {
    # SLURM's grace signal has to reach the live child, not just this wrapper
    if [[ -n "${child}" ]]; then
        say "forwarding ${1} to pid ${child}"
        kill -"${1}" "${child}" 2>/dev/null || true
    fi
}
trap 'forward TERM' TERM
trap 'forward INT' INT

if [[ $# -eq 0 ]]; then
    echo "[chain] no training command given" >&2
    exit 2
fi

say "train: $*"
"$@" &
child=$!
set +e
wait "${child}"
status=$?
set -e
child=""

if [[ ${status} -ne 0 ]]; then
    say "training exited ${status}; skipping eval"
    exit "${status}"
fi
say "training finished"

if [[ -z "${RNK_EVAL_CONFIG}" || -z "${RNK_EVAL_RUNS}" ]]; then
    say "no eval configured"
    exit 0
fi

IFS=',' read -r -a runs <<< "${RNK_EVAL_RUNS}"
failures=0
for run in "${runs[@]}"; do
    [[ -z "${run}" ]] && continue
    say "eval: ${run} with ${RNK_EVAL_CONFIG}"
    set +e
    "${RNK_PYTHON}" scripts/eval.py \
        --run "${run}" \
        --eval_config "${RNK_EVAL_CONFIG}" \
        --checkpoint "${RNK_EVAL_CHECKPOINT}" \
        --headless
    eval_status=$?
    set -e
    if [[ ${eval_status} -ne 0 ]]; then
        say "WARNING: eval for ${run} exited ${eval_status}; continuing"
        failures=$((failures + 1))
    fi
done

if [[ ${failures} -gt 0 ]]; then
    say "${failures} of ${#runs[@]} evals failed; training succeeded, so exiting 0"
fi
exit 0

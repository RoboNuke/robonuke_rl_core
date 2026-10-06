"""`hpc_job_chain.bash`: train, then eval, and what happens when either goes wrong.

This is the "does it start the right processes" question for a train-then-eval job, and it
needs **no cluster and no container**: the wrapper's inputs are a command and some env vars,
so a stub standing in for python records exactly what it was asked to run. Everything the
wrapper decides -- run eval at all, run it per agent, with which checkpoint, and what to do
when something exits nonzero -- is settled here, on CPU, where a failure costs a second
instead of a queue wait.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from robonuke_rl_core.hpc import submit as S


def stub(tmp_path, name="python_stub", exit_code=0):
    """A fake interpreter that appends its argv to a log and exits with `exit_code`."""
    path = tmp_path / name
    log = tmp_path / f"{name}.log"
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {log}\n'
        f"exit {exit_code}\n"
    )
    path.chmod(0o755)
    return path, log


def run_chain(tmp_path, *, train_cmd, env_extra=None, expect=0):
    env = dict(os.environ)
    env.pop("RNK_EVAL_CONFIG", None)
    env.pop("RNK_EVAL_RUNS", None)
    env.update(env_extra or {})
    done = subprocess.run(
        ["bash", str(S.chain_script()), *train_cmd],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == expect, f"exit {done.returncode}\n{done.stdout}\n{done.stderr}"
    return done


def test_training_alone_runs_and_no_eval_is_attempted(tmp_path):
    done = run_chain(tmp_path, train_cmd=["true"])
    assert "no eval configured" in done.stdout
    assert "training finished" in done.stdout


def test_eval_runs_once_per_agent_after_a_clean_train(tmp_path):
    python, log = stub(tmp_path)
    done = run_chain(
        tmp_path,
        train_cmd=["true"],
        env_extra={
            "RNK_PYTHON": str(python),
            "RNK_EVAL_CONFIG": "configs/eval/quick.yaml",
            "RNK_EVAL_RUNS": "hur/P/g_a0,hur/P/g_a1,hur/P/g_a2",
            "RNK_EVAL_CHECKPOINT": "best",
        },
    )
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 3, f"expected one eval per agent, got {lines}"
    for index, line in enumerate(lines):
        assert "scripts/eval.py" in line
        assert f"--run hur/P/g_a{index}" in line
        assert "--eval_config configs/eval/quick.yaml" in line
        assert "--checkpoint best" in line
        assert "--headless" in line  # a queued job has no display
    assert "training finished" in done.stdout


def test_the_checkpoint_choice_is_forwarded(tmp_path):
    python, log = stub(tmp_path)
    run_chain(
        tmp_path,
        train_cmd=["true"],
        env_extra={
            "RNK_PYTHON": str(python),
            "RNK_EVAL_CONFIG": "e.yaml",
            "RNK_EVAL_RUNS": "hur/P/g_a0",
            "RNK_EVAL_CHECKPOINT": "2500",
        },
    )
    assert "--checkpoint 2500" in log.read_text()


def test_the_checkpoint_defaults_to_best(tmp_path):
    python, log = stub(tmp_path)
    env = {"RNK_PYTHON": str(python), "RNK_EVAL_CONFIG": "e.yaml", "RNK_EVAL_RUNS": "hur/P/g_a0"}
    run_chain(tmp_path, train_cmd=["true"], env_extra=env)
    assert "--checkpoint best" in log.read_text()


def test_a_failed_train_skips_eval_and_propagates_its_exit_code(tmp_path):
    """The expensive thing failed; running eval on a half-trained run wastes more time."""
    python, log = stub(tmp_path)
    done = run_chain(
        tmp_path,
        train_cmd=["bash", "-c", "exit 7"],
        env_extra={
            "RNK_PYTHON": str(python),
            "RNK_EVAL_CONFIG": "e.yaml",
            "RNK_EVAL_RUNS": "hur/P/g_a0",
        },
        expect=7,
    )
    assert "skipping eval" in done.stdout
    assert not log.exists(), "eval ran after a failed train"


def test_a_failed_eval_is_non_fatal_and_the_rest_still_run(tmp_path):
    """Training that finished is the expensive thing. A wandb hiccup in eval must not turn a
    completed run into a failed job -- and must not stop the other agents' evals."""
    python, log = stub(tmp_path, exit_code=3)
    done = run_chain(
        tmp_path,
        train_cmd=["true"],
        env_extra={
            "RNK_PYTHON": str(python),
            "RNK_EVAL_CONFIG": "e.yaml",
            "RNK_EVAL_RUNS": "hur/P/g_a0,hur/P/g_a1",
        },
        expect=0,  # the job still succeeds
    )
    assert len(log.read_text().strip().splitlines()) == 2, "it stopped at the first failure"
    assert done.stdout.count("WARNING") == 2
    assert "2 of 2 evals failed" in done.stdout


def test_an_empty_run_list_means_no_eval(tmp_path):
    python, log = stub(tmp_path)
    run_chain(
        tmp_path,
        train_cmd=["true"],
        env_extra={"RNK_PYTHON": str(python), "RNK_EVAL_CONFIG": "e.yaml", "RNK_EVAL_RUNS": ""},
    )
    assert not log.exists()


def test_no_training_command_is_an_error(tmp_path):
    done = subprocess.run(
        ["bash", str(S.chain_script())],
        capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False,
    )
    assert done.returncode == 2
    assert "no training command" in done.stderr


def test_the_wrapper_forwards_a_termination_signal_to_the_live_child(tmp_path):
    """SLURM's --signal=TERM@300 lands on the wrapper; the training process is what needs it.

    Without the forward, bash takes the signal, the job dies, and whatever the trainer was
    writing is left half-written -- which is exactly what the grace period exists to avoid.
    """
    started = tmp_path / "child_started"
    caught = tmp_path / "child_caught_term"
    child = tmp_path / "child.bash"
    child.write_text(
        "#!/usr/bin/env bash\n"
        f"trap 'echo caught > {caught}; exit 0' TERM\n"
        f"touch {started}\n"
        "while true; do sleep 0.05; done\n"
    )
    child.chmod(0o755)

    process = subprocess.Popen(
        ["bash", str(S.chain_script()), "bash", str(child)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
    )
    try:
        wait_until(started.exists, "the child never started")
        process.terminate()  # stands in for SLURM's grace signal
        wait_until(caught.exists, "the child never saw TERM; the wrapper did not forward it")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
    assert caught.read_text().strip() == "caught"


def wait_until(predicate, message, timeout=20.0, interval=0.05):
    """Poll `predicate` until it is true, or fail with `message`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    pytest.fail(f"{message} (waited {timeout:.0f}s)")

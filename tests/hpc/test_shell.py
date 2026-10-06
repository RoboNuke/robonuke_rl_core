"""The shell scripts parse, ship, and say what they need.

Their real behaviour needs a cluster, so it lives behind the `hpc` marker and the README's
manual checklist. What is worth checking here is cheap and catches the mistakes that would
otherwise surface inside a queued job: a syntax error, a script that did not get packaged, a
renamed env var that the submitter still exports under the old name.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from robonuke_rl_core.hpc import submit as S

SCRIPTS = {
    "hpc_job.bash": S.job_script(),
    "hpc_job_chain.bash": S.chain_script(),
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_the_script_is_shipped_with_the_package(name):
    path = SCRIPTS[name]
    assert path.is_file(), f"{name} is missing; it is submitted by path, so it must ship"
    assert path.parent.name == "hpc"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_the_script_parses(name):
    done = subprocess.run(
        ["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_the_script_fails_loud(name):
    """`set -Eeuo pipefail` plus an ERR trap naming file:line. A job that dies quietly in a
    container is the worst thing to debug from a queue."""
    text = SCRIPTS[name].read_text()
    assert "set -Eeuo pipefail" in text
    assert "trap" in text and "LINENO" in text


def test_the_job_script_reads_exactly_what_the_submitter_exports():
    """A renamed variable on one side and not the other is invisible until a job runs."""
    text = S.job_script().read_text()
    exported = set(
        S.job_env(
            S.HpcCfg(sif_image="/s.sif", cache_home="/c"),
            package_root=S.Path("/pkg"),
            project_root=S.Path("/proj"),
        )
    )
    for name in exported:
        assert name in text, f"the submitter exports {name}, which hpc_job.bash never reads"


def test_the_job_script_validates_its_inputs_before_launching_a_container():
    text = S.job_script().read_text()
    for name in ("RNK_SIF", "RNK_PKG_ROOT", "RNK_PROJECT_ROOT", "RNK_CACHE_HOME"):
        assert name in text
    assert "command -v" in text  # the apptainer binary is checked too


def test_the_job_script_binds_the_clone_over_the_image_install_path():
    """The whole image design: bake the stack, bind the code. Updating the package on the
    cluster must be a git pull, not an image rebuild."""
    text = S.job_script().read_text()
    assert "RNK_IMAGE_PKG_PATH" in text
    assert "/opt/robonuke_rl_core" in text
    assert '"${RNK_PKG_ROOT}:${RNK_IMAGE_PKG_PATH}"' in text


def test_the_job_script_execs_so_slurms_signal_reaches_python():
    """`--signal=TERM@300` has to land on the training process, not on bash."""
    text = S.job_script().read_text()
    assert re.search(r"^exec \"\$\{RNK_APPTAINER_BIN\}\"", text, re.MULTILINE)


def test_the_job_script_never_dies_over_wandb():
    """No key, or a bad paste, must go offline rather than kill an unattended job."""
    text = S.job_script().read_text()
    assert "WANDB_MODE=offline" in text
    assert "WANDB_API_KEY" in text


def test_the_chain_script_runs_eval_only_after_a_clean_train():
    text = S.chain_script().read_text()
    assert "skipping eval" in text
    assert "scripts/eval.py" in text


def test_the_chain_script_treats_eval_as_non_fatal():
    """Training that finished is the expensive thing; a wandb hiccup in eval must not turn a
    completed run into a failed job."""
    text = S.chain_script().read_text()
    assert "continuing" in text
    assert "exit 0" in text


def test_the_chain_script_forwards_the_termination_signal():
    text = S.chain_script().read_text()
    assert "trap 'forward TERM' TERM" in text
    assert "kill -" in text


def test_the_image_build_script_parses_and_pins_isaac_lab():
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "hpc" / "build_image.sh"
    assert script.is_file()
    done = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    text = script.read_text()
    assert "ISAACLAB_COMMIT" in text  # a version bump is a deliberate act
    assert "APPTAINER_TMPDIR" in text and "Lustre" in text  # the build trap V hit
    assert "pip install --no-cache-dir -e ." in text  # editable, so the bind works
    assert "verifying" in text  # and it proves the image imports before anyone queues work

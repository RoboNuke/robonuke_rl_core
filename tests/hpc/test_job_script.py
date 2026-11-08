"""`hpc_job.bash` on a fake node: what it asks nvidia-smi, which image it picks, what it binds.

Like `test_chain_script.py`, this needs **no cluster and no container**. The script's two
external commands -- `nvidia-smi` and `apptainer` -- are stubs on PATH: the first answers with
whatever GPU the test puts on the node, the second records the argv it was exec'd with and
exits. Everything the script decides from that -- refuse a MIG slice, switch to the sm_70
image on a V100, bind an extra repo and put it on the container PYTHONPATH, keep the login
shell's PYTHONPATH out -- is settled here, on CPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from robonuke_rl_core.hpc import submit as S

H100 = (
    "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaa)\n",
    "NVIDIA H100 80GB HBM3, 9.0\n",
)
V100 = (
    "GPU 0: Tesla V100-SXM3-32GB (UUID: GPU-bbbb)\nGPU 1: Tesla V100-SXM3-32GB (UUID: GPU-cccc)\n",
    "Tesla V100-SXM3-32GB, 7.0\nTesla V100-SXM3-32GB, 7.0\n",
)
MIG = (
    "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-dddd)\n"
    "  MIG 1g.10gb     Device  0: (UUID: MIG-eeee)\n",
    "NVIDIA H100 80GB HBM3, 9.0\n",
)


def node(tmp_path, gpu=H100, *, nvidia_smi=True):
    """A fake compute node: stub binaries on PATH, the three directories, two images.

    Returns the environment to run the job script with and the path the apptainer stub
    writes its argv to.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apptainer_log = tmp_path / "apptainer.log"
    apptainer = bin_dir / "apptainer"
    apptainer.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {apptainer_log}\n"
        "exit 0\n"
    )
    apptainer.chmod(0o755)
    if nvidia_smi:
        listing, caps = gpu
        smi = bin_dir / "nvidia-smi"
        smi.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == "-L" ]]; then\n'
            f"    printf '%s' {_quoted(listing)}\n"
            'elif [[ "$1" == --query-gpu=* ]]; then\n'
            f"    printf '%s' {_quoted(caps)}\n"
            "else\n"
            '    echo "unexpected nvidia-smi args: $*" >&2; exit 1\n'
            "fi\n"
        )
        smi.chmod(0o755)

    for name in ("pkg", "project", "cache", "furniture_lab", "third"):
        (tmp_path / name).mkdir()
    sif = tmp_path / "general-cu128.sif"
    sif.write_text("")
    sif_sm70 = tmp_path / "general-cu126.sif"
    sif_sm70.write_text("")

    # a whitelist PATH: the stubs, plus the few host tools the script itself uses. A real
    # nvidia-smi on the test machine must never be found, and `nvidia_smi=False` must mean
    # "not on PATH" rather than "a stub that fails".
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("bash", "grep", "head", "awk", "touch", "printf", "env"):
        found = shutil.which(tool)
        if found:
            (tools / tool).symlink_to(found)

    env = {
        "PATH": f"{bin_dir}:{tools}",
        "HOME": str(tmp_path),
        "RNK_PKG_ROOT": str(tmp_path / "pkg"),
        "RNK_PROJECT_ROOT": str(tmp_path / "project"),
        "RNK_CACHE_HOME": str(tmp_path / "cache"),
        "RNK_SIF": str(sif),
        "RNK_APPTAINER_BIN": "apptainer",
        "RNK_PYTHON": "python",
        "WANDB_MODE": "offline",
    }
    return env, apptainer_log


def _quoted(text: str) -> str:
    """A bash single-quoted literal that reproduces `text` byte for byte."""
    return "'" + text.replace("'", "'\\''") + "'"


def run_job(env, *argv, expect=0):
    done = subprocess.run(
        ["bash", str(S.job_script()), *(argv or ("python", "scripts/train.py"))],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert done.returncode == expect, f"exit {done.returncode}\n{done.stdout}\n{done.stderr}"
    return done


def apptainer_argv(log) -> list:
    return log.read_text().splitlines()


def env_value(argv, name) -> str:
    """The value of `--env NAME=value` in the apptainer argv."""
    for index, part in enumerate(argv):
        if part == "--env" and argv[index + 1].startswith(f"{name}="):
            return argv[index + 1].split("=", 1)[1]
    raise AssertionError(f"no --env {name}= in {argv}")


def binds(argv) -> list:
    return [argv[index + 1] for index, part in enumerate(argv) if part == "--bind"]


def image(argv) -> str:
    """The .sif apptainer was given: the argument after the last `--env NAME=value` pair."""
    last_env = max(index for index, part in enumerate(argv) if part == "--env")
    return argv[last_env + 2]


# ------------------------------------------------------------------ the GPU
def test_the_gpu_is_logged_and_a_whole_h100_runs_the_default_image(tmp_path):
    env, log = node(tmp_path, H100)
    env["RNK_SIF_SM70"] = str(tmp_path / "general-cu126.sif")  # set, but not for this GPU
    done = run_job(env)
    assert "NVIDIA H100 80GB HBM3 (UUID: GPU-aaaa)" in done.stdout
    assert "9.0" in done.stdout
    argv = apptainer_argv(log)
    assert image(argv) == env["RNK_SIF"], argv
    assert argv[-2:] == ["python", "scripts/train.py"]


def test_a_mig_slice_is_refused_naming_the_constraint_fix(tmp_path):
    """A MIG device has no graphics, so Kit cannot start on it. Fail before the container
    does, and say which config field keeps the job off such a node."""
    env, log = node(tmp_path, MIG)
    done = run_job(env, expect=2)
    assert "MIG" in done.stderr
    assert "hpc.constraint" in done.stderr
    assert not log.exists(), "apptainer ran on a MIG slice"


def test_a_v100_switches_to_the_sm70_image_and_says_why(tmp_path):
    env, log = node(tmp_path, V100)
    env["RNK_SIF_SM70"] = str(tmp_path / "general-cu126.sif")
    done = run_job(env)
    assert "7.0" in done.stdout and "sm_70" in done.stdout
    assert "general-cu126.sif" in done.stdout  # says which image it chose
    argv = apptainer_argv(log)
    assert image(argv) == env["RNK_SIF_SM70"], argv


def test_a_v100_without_an_sm70_image_warns_and_continues(tmp_path):
    env, log = node(tmp_path, V100)
    done = run_job(env)
    assert "WARNING" in done.stdout and "sm_70" in done.stdout
    assert image(apptainer_argv(log)) == env["RNK_SIF"]


def test_a_missing_nvidia_smi_warns_and_continues(tmp_path):
    env, log = node(tmp_path, nvidia_smi=False)
    done = run_job(env)
    assert "WARNING" in done.stdout and "nvidia-smi" in done.stdout
    assert image(apptainer_argv(log)) == env["RNK_SIF"]


def test_the_chosen_image_is_touched_so_the_share_does_not_purge_it(tmp_path):
    env, log = node(tmp_path, V100)
    sm70 = tmp_path / "general-cu126.sif"
    env["RNK_SIF_SM70"] = str(sm70)
    old = time.time() - 100 * 86400  # 100 days idle: past hpc-share's 90-day purge
    os.utime(sm70, (old, old))
    os.utime(tmp_path / "general-cu128.sif", (old, old))
    run_job(env)
    assert sm70.stat().st_mtime > old + 86400, "the chosen image was not touched"
    assert (tmp_path / "general-cu128.sif").stat().st_mtime < old + 1, "the other image was"


# ------------------------------------------------------------------ binds and PYTHONPATH
def test_extra_repos_are_bound_at_their_own_paths_and_appended_to_pythonpath(tmp_path):
    """Bind the code, install nothing: a third clone follows the same pattern as the two
    repos, and lands on PYTHONPATH after them so the package still wins."""
    env, log = node(tmp_path)
    repos = [str(tmp_path / "furniture_lab"), str(tmp_path / "third")]
    env["RNK_EXTRA_REPOS"] = ",".join(repos)
    done = run_job(env)
    argv = apptainer_argv(log)
    for repo in repos:
        assert f"{repo}:{repo}" in binds(argv)
    assert env_value(argv, "PYTHONPATH") == ":".join(
        [env["RNK_PKG_ROOT"], env["RNK_PROJECT_ROOT"], *repos]
    )
    assert "extra repos" in done.stdout


def test_an_extra_repo_that_is_not_a_directory_fails_before_the_container(tmp_path):
    env, log = node(tmp_path)
    env["RNK_EXTRA_REPOS"] = str(tmp_path / "nowhere")
    done = run_job(env, expect=2)
    assert "RNK_EXTRA_REPOS" in done.stderr and "nowhere" in done.stderr
    assert not log.exists()


def test_an_extra_repo_already_bound_is_bound_once(tmp_path):
    """Through the same dedup as the two repos: apptainer warns on a repeated destination,
    and a warning nobody needs hides a real one."""
    env, log = node(tmp_path)
    env["RNK_EXTRA_REPOS"] = env["RNK_PROJECT_ROOT"]
    run_job(env)
    mounted = binds(apptainer_argv(log))
    assert mounted.count(f"{env['RNK_PROJECT_ROOT']}:{env['RNK_PROJECT_ROOT']}") == 1


def test_the_login_shell_pythonpath_is_not_appended(tmp_path):
    """The container PYTHONPATH is exactly package:project[:extra...]. A host PYTHONPATH names
    installs the container cannot see, or shadows the image's stack through a bind."""
    env, log = node(tmp_path)
    env["PYTHONPATH"] = "/home/me/.local/lib/python3.11/site-packages:/host/only"
    run_job(env)
    argv = apptainer_argv(log)
    assert env_value(argv, "PYTHONPATH") == f"{env['RNK_PKG_ROOT']}:{env['RNK_PROJECT_ROOT']}"
    assert "/host/only" not in " ".join(argv)


def test_the_container_gets_the_same_pythonpath_on_the_chain_path(tmp_path):
    env, log = node(tmp_path)
    env["RNK_EXTRA_REPOS"] = str(tmp_path / "third")
    env["RNK_CHAIN_SCRIPT"] = str(S.chain_script())
    env["PYTHONPATH"] = "/host/only"
    run_job(env)
    argv = apptainer_argv(log)
    assert "/opt/hpc_job_chain.bash" in argv
    assert env_value(argv, "PYTHONPATH") == ":".join(
        [env["RNK_PKG_ROOT"], env["RNK_PROJECT_ROOT"], str(tmp_path / "third")]
    )
    assert "/host/only" not in " ".join(argv)

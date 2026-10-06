"""Does SLURM accept our jobs, and do they start the right process?

Run on a **login node**:

    RNK_TEST_CONFIG=configs/base/hpc.yaml pytest -m hpc tests/hpc/HPC

These are about the **SLURM path**, not the container. Whether the image can import the stack
is the build script's `verify` step, which runs once at build time; re-testing it here would
mean running `apptainer exec` by hand for no new information. So apptainer appears below only
where it always does in real use — on the compute node, inside a job SLURM started.

Marked `hpc` and deselected by default, like the Isaac Sim tests are marked `gpu`. And like
those, **nothing here skips silently**: a cluster test that cannot find its cluster fails,
because a green suite that quietly tested nothing is worse than a red one.

`RNK_TEST_CONFIG` is any config whose chain carries a complete `hpc` section — the project's
`configs/base/hpc.yaml`, or an experiment that chains onto it. Everything is read from it, so
these follow your cluster's real settings instead of hard-coding them.

Nothing here records video: the recorder camera is a separate path with its own problems, and
a launcher test has no business depending on it.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from robonuke_rl_core.hpc import launch_train
from robonuke_rl_core.hpc import submit as S

pytestmark = pytest.mark.hpc

#: how long to wait for a trivial job to get through the queue and run
JOB_TIMEOUT = float(os.environ.get("RNK_TEST_JOB_TIMEOUT", 900))
#: how long any one command may take
CMD_TIMEOUT = 120
#: printed by the probe job, so finding it in the .out proves the job ran OUR command.
#: The probe BUILDS it by concatenation, so the string only ever appears in the log if python
#: really evaluated the expression -- `hpc_job.bash` echoes the command it was given, and a
#: literal marker in that echo would make these assertions pass without python running at all.
MARKER = "RNK_SLURM_SELFTEST_OK"
MARKER_EXPR = "'RNK_SLURM' + '_SELFTEST' + '_OK'"


@pytest.fixture(scope="module")
def config() -> S.SubmitConfig:
    """The config under test, from `RNK_TEST_CONFIG`."""
    raw = os.environ.get("RNK_TEST_CONFIG", "")
    if not raw:
        pytest.fail(
            "set RNK_TEST_CONFIG to a config whose chain carries a complete hpc section, e.g. "
            "RNK_TEST_CONFIG=configs/base/hpc.yaml pytest -m hpc tests/hpc/HPC"
        )
    path = Path(raw)
    if not path.is_file():
        pytest.fail(f"RNK_TEST_CONFIG={raw} is not a file")
    submit = S.read_submit_config(path)
    S.require_submit_fields(submit)  # fail here, not three tests later
    return submit


def run(command, **kwargs):
    """A subprocess with a timeout, so a wedged cluster command fails rather than hangs."""
    return subprocess.run(
        command, capture_output=True, text=True, timeout=CMD_TIMEOUT, check=False, **kwargs
    )


# ------------------------------------------------------------------ 1. the tools and paths
def test_sbatch_is_available():
    assert shutil.which("sbatch"), "no sbatch on PATH; these tests run on a login node"


def test_squeue_is_available():
    """The probe job below polls for completion, so this is needed to interpret it."""
    assert shutil.which("squeue"), "no squeue on PATH"


def test_the_image_the_jobs_will_use_exists(config):
    """A path check, not a container run: the job cannot start without this file."""
    path = Path(config.hpc.sif_image).expanduser()
    assert path.is_file(), (
        f"hpc.sif_image={path} is not a file. The intended image is the existing "
        "~/hpc-share/isaac/ghvic.sif (see examples/hpc.yaml) -- these tests reuse it, they do "
        "not build or inspect one. hpc/build_image.sh is only for making a fresh image if "
        "that one ever stops being enough."
    )


def test_the_cache_home_is_writable(config):
    """Kit and shader caches land here and run to GBs, so a quota'd NFS home fails a job
    partway through, which is the worst time to find out."""
    path = Path(config.hpc.cache_home).expanduser()
    assert path.is_dir(), f"hpc.cache_home={path} is not a directory"
    probe = path / ".rnk_write_probe"
    probe.write_text("ok")
    assert probe.read_text() == "ok"
    probe.unlink()


def test_every_extra_bind_exists(config):
    for bind in config.hpc.binds:
        host = bind.split(":", 1)[0]
        assert Path(host).expanduser().exists(), f"hpc.binds host path {host!r} does not exist"


# ------------------------------------------------------------------ 2. SLURM accepts the job
def probe_sbatch(config: S.SubmitConfig, tmp_path: Path, *, argv, binds=()) -> list:
    """The sbatch command for a trivial probe job, built exactly as a launcher builds one."""
    out_path, err_path = S.log_paths(config.hpc, "hpc_selftest", "selftest", create=True)
    hpc = dataclasses.replace(config.hpc, binds=list(config.hpc.binds) + list(binds))
    return S.sbatch_command(
        hpc=hpc,
        job_name="rnk_hpc_selftest",
        out_path=out_path,
        err_path=err_path,
        env=S.job_env(
            hpc,
            package_root=launch_train.package_root(),
            project_root=launch_train.project_root(),
        ),
        argv=argv,
    )


def write_probe(tmp_path: Path) -> Path:
    """The probe as a FILE, not a `python -c` string.

    sbatch word-splits the arguments it passes to a job script, so a multi-line snippet
    arrives as `-c import` and dies with a SyntaxError inside the container. A path is one
    word, and it is also how production invokes the job: `python scripts/train.py ...`.

    The import of the package is LAST, so a failure still leaves us the environment that
    caused it instead of an empty log. MARKER is assembled from pieces, so the string can only
    appear in the log if python really evaluated it -- `hpc_job.bash` echoes the command it was
    given, and a literal marker there would make the assertion pass without python running.
    """
    probe = tmp_path / "rnk_probe.py"
    probe.write_text(
        "import os, pathlib, sys\n"
        f"print('PROBE MARK', {MARKER_EXPR})\n"
        "print('PROBE CWD', os.getcwd())\n"
        "print('PROBE HOME', os.environ.get('HOME'))\n"
        "print('PROBE PYTHONPATH', os.environ.get('PYTHONPATH', '<unset>'))\n"
        "print('PROBE SYSPATH', [p for p in sys.path if 'robonuke' in p] or '<no robonuke entry>')\n"
        "import robonuke_rl_core as r\n"
        "print('PROBE PKG', pathlib.Path(r.__file__).resolve())\n"
    )
    return probe


def test_slurm_accepts_a_submission_without_queueing_it(config, tmp_path):
    """`sbatch --test-only` validates account, partitions and resources and queues nothing.

    This is the check that catches a partition the account may not use, which otherwise shows
    up as a job that sits pending forever with no explanation.
    """
    command = probe_sbatch(
        config, tmp_path, argv=[config.hpc.container_python, "-c", "print('ok')"]
    )
    command.insert(1, "--test-only")
    done = run(command)
    assert done.returncode == 0, (
        f"SLURM rejected the submission:\n{done.stdout}\n{done.stderr}\n"
        f"Check hpc.account={config.hpc.account!r} and "
        f"hpc.partitions={config.hpc.partitions!r}."
    )


def test_the_resource_flags_are_the_configs(config, tmp_path):
    """What SLURM is asked for comes from this config's hpc section, not a shell file."""
    command = probe_sbatch(config, tmp_path, argv=["true"])
    text = " ".join(command)
    assert f"-A {config.hpc.account}" in text
    assert f"-p {config.hpc.partitions}" in text
    assert f"--time {config.hpc.time}" in text
    assert f"--gres=gpu:{config.hpc.gpus}" in text
    assert f"--mem {config.hpc.mem}" in text
    assert f"--signal {config.hpc.signal}" in text


# ------------------------------------------------------------------ 3. a job really starts
def job_id(stdout: str) -> str:
    match = re.search(r"(\d+)", stdout)
    assert match, f"could not find a job id in sbatch output: {stdout!r}"
    return match.group(1)


def wait_for(job: str, timeout: float) -> None:
    """Poll until the job leaves the queue; scancel and fail on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        done = run(["squeue", "-h", "-j", job])
        if done.returncode != 0 or not done.stdout.strip():
            return
        time.sleep(10)
    run(["scancel", job])
    pytest.fail(
        f"job {job} did not finish within {timeout:.0f}s and was cancelled. Raise "
        "RNK_TEST_JOB_TIMEOUT if the queue is just busy."
    )


def test_a_real_job_runs_the_process_we_asked_for(config, tmp_path):
    """Submit a trivial job and prove from its log that OUR command ran, in the right place.

    This is the one test that exercises the whole chain the way production does: sbatch flags,
    the spooled `hpc_job.bash`, the exported RNK_* variables, the binds, and the container
    starting the python we named. It costs a few seconds of one GPU.

    What the log must show: the marker (so it was our command), the project root as cwd (so
    configs resolve the way train.py expects), and the package resolving to the bound clone
    (so the job runs current code rather than whatever the image baked).
    """
    probe = write_probe(tmp_path)
    command = probe_sbatch(
        config,
        tmp_path,
        argv=[config.hpc.container_python, str(probe)],
        binds=[f"{tmp_path}:{tmp_path}"],  # so the probe file exists inside the container
    )
    submitted = run(command)
    assert submitted.returncode == 0, f"sbatch failed:\n{submitted.stderr}"
    job = job_id(submitted.stdout)
    print(f"[selftest] submitted job {job}", flush=True)

    wait_for(job, JOB_TIMEOUT)

    out_path, err_path = S.log_paths(config.hpc, "hpc_selftest", "selftest", create=False)
    log = Path(str(out_path).replace("%j", job))
    errors = Path(str(err_path).replace("%j", job))
    assert log.is_file(), (
        f"no job log at {log}. hpc.exp_log_dir={config.hpc.exp_log_dir!r} must be writable "
        "from the compute node."
    )
    text = log.read_text()
    tail = errors.read_text()[-4000:] if errors.is_file() else "<no .err>"
    # EVERY assertion below shows both streams: a job that failed inside the container writes
    # its traceback to .err, and hiding that is how a green-looking failure wastes an hour
    where = f"\n--- {log} ---\n{text}\n--- {errors} ---\n{tail}"

    assert f"PROBE MARK {MARKER}" in text, f"python never ran, or died before its first print.{where}"
    assert f"PROBE CWD {launch_train.project_root()}" in text, (
        f"the job did not cd to the project root, so project-relative configs would not "
        f"resolve.{where}"
    )
    assert f"PROBE PKG {launch_train.package_root()}/" in text, (
        "the package did not resolve to the bound clone, so jobs would not be running the "
        f"cluster's checkout. The PROBE PYTHONPATH and PROBE SYSPATH lines say why.{where}"
    )


def test_a_job_whose_image_is_missing_fails_fast(config, tmp_path):
    """A bad path must fail in seconds with a named variable, not after the queue wait."""
    command = probe_sbatch(config, tmp_path, argv=["true"])
    # swap the image path in the --export list for one that does not exist
    for index, part in enumerate(command):
        if part.startswith("--export="):
            command[index] = re.sub(
                r"RNK_SIF=[^,]*", "RNK_SIF=/nonexistent/image.sif", part
            )
    submitted = run(command)
    assert submitted.returncode == 0, submitted.stderr
    job = job_id(submitted.stdout)
    wait_for(job, JOB_TIMEOUT)

    _, err_path = S.log_paths(config.hpc, "hpc_selftest", "selftest", create=False)
    errors = Path(str(err_path).replace("%j", job))
    assert errors.is_file(), f"no .err at {errors}"
    assert "RNK_SIF" in errors.read_text(), (
        "the job script did not name the missing image; it must validate its inputs before "
        f"launching a container.\n{errors.read_text()[-2000:]}"
    )


# ------------------------------------------------------------------ 4. the launcher itself
def test_the_launcher_composes_a_job_for_this_config(config, capsys):
    """The last gate before a real submit: `--dry_run` on the project's own config."""
    code = launch_train.main(
        [
            str(config.path),
            "--project", "hpc_selftest",
            "--group_prefix", "selftest",
            "--dry_run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    line = next(l for l in out.splitlines() if l.startswith("sbatch "))
    assert f"-A {config.hpc.account}" in line
    assert str(Path(config.hpc.sif_image).expanduser()) in line
    assert "scripts/train.py" in line

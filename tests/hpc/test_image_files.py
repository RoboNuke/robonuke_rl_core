"""The image's data files: the lock files and the definition template, checked on CPU.

The image is the `general` env reproduced from `hpc/requirements-*.lock` by
`hpc/general.def.in`; the build runs on a cluster compute node, an hour from the first sign of
a typo. So the cheap properties are proven here: the locks are exact pins the `--no-deps`
install can act on, the two flavors differ only where torch's CUDA build does, and the
template renders with no token left behind.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

HPC = Path(__file__).resolve().parents[2] / "hpc"
PIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9.+_-]+$")


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert PIN_RE.match(line), f"{path.name}: not an exact pin: {line!r}"
        name, version = line.split("==", 1)
        assert name.lower() not in pins, f"{path.name}: {name} pinned twice"
        pins[name.lower()] = version
    return pins


def test_the_locks_are_exact_pins_of_the_general_env():
    cu128 = _pins(HPC / "requirements-cu128.lock")
    cu126 = _pins(HPC / "requirements-cu126.lock")
    assert cu128["torch"].endswith("+cu128") and cu126["torch"].endswith("+cu126")
    assert cu128["torchvision"].endswith("+cu128") and cu126["torchvision"].endswith("+cu126")
    assert cu128["isaacsim"] == cu126["isaacsim"]
    for name in ("pip", "setuptools", "wheel", "isaacsim", "warp-lang", "numpy", "omegaconf", "wandb", "skrl", "diffusers", "pyarrow", "imageio-ffmpeg"):
        assert name in cu128, f"{name} missing from the cu128 lock"
    # no editable install survives: Isaac Lab is cloned by the def, the repos are bound
    for name in ("isaaclab", "isaaclab_tasks", "robonuke_rl_core", "robonuke-furniture-bench"):
        assert name not in cu128 and name not in cu126


def test_the_two_flavors_differ_only_in_torch_and_the_nvidia_libraries():
    cu128 = _pins(HPC / "requirements-cu128.lock")
    cu126 = _pins(HPC / "requirements-cu126.lock")

    def rest(pins):
        return {n: v for n, v in pins.items() if not (n.startswith("nvidia-") or n in ("torch", "torchvision"))}

    assert rest(cu128) == rest(cu126)
    # the cu126 torch wheel's nvidia pins are the 12.6 line, and every one is named
    assert cu126["nvidia-cuda-runtime-cu12"].startswith("12.6")
    assert cu128["nvidia-cuda-runtime-cu12"].startswith("12.8")
    assert set(n for n in cu126 if n.startswith("nvidia-")) >= {
        "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12", "nvidia-nccl-cu12"
    }


def test_env_lock_names_what_the_def_needs():
    text = (HPC / "env.lock.env").read_text()
    values = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))
    assert re.fullmatch(r"3\.11\.\d+", values["PYTHON_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{40}", values["ISAACLAB_COMMIT"])
    assert values["ISAACLAB_REPO"].startswith("https://")
    assert values["ISAACSIM_VERSION"].startswith("5.1.0")
    assert values["TORCH_VERSION"].endswith("+cu128")


def test_the_definition_template_renders_with_no_token_left():
    template = (HPC / "general.def.in").read_text()
    tokens = set(re.findall(r"@([A-Z_]+)@", template))
    expected = {
        "FLAVOR", "LOCKFILE", "LOCK_PATH", "ENV_LOCK_PATH", "LOCK_SHA", "PYTHON_VERSION",
        "ISAACLAB_REPO", "ISAACLAB_COMMIT", "ISAACSIM_VERSION", "MINIFORGE_VERSION", "BUILT_ON",
    }
    assert tokens == expected, f"template tokens {tokens ^ expected} are not what build_image.sh renders"
    rendered = template
    for token in expected:
        rendered = rendered.replace(f"@{token}@", "x")
    assert not re.search(r"@[A-Z_]+@", rendered)
    # the decisions the build depends on are in the text, not only in a comment
    assert "pip install --no-cache-dir --no-deps --no-build-isolation" in rendered
    assert "buildtools.txt" in rendered
    assert "nvidia_icd.json" in rendered and "10_nvidia.json" in rendered
    assert "isaacsim-kit" in rendered  # kit/{cache,data,logs} out of the read-only image
    assert "OMNI_KIT_ACCEPT_EULA=YES" in rendered


def test_the_scripts_parse():
    for name in ("build_image.sh", "build_image.sbatch", "export_env.sh", "install_hpc.sh", "verify_partitions.sh"):
        done = subprocess.run(["bash", "-n", str(HPC / name)], capture_output=True, text=True, check=False)
        assert done.returncode == 0, f"{name}: {done.stderr}"
    for name in ("smoke/smoke.py", "smoke/report.py"):
        compile((HPC / name).read_text(), name, "exec")

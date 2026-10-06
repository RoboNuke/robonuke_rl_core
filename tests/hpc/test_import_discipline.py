"""`robonuke_rl_core.hpc` must import without torch, wandb or Isaac Lab.

The submitters run on a cluster **login node**: a light python, none of the Isaac
environment, no GPU. A stray heavy import at module level turns "submit a job" into "install
Isaac Lab on the login node", and the failure shows up as an ImportError in front of someone
who only wanted to queue work.

This is checked in a subprocess, because by the time the rest of the suite has run, torch is
already in `sys.modules` and an in-process assert would pass no matter what.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: the modules a login node is not expected to have
FORBIDDEN = ("torch", "wandb", "isaaclab", "isaacsim", "skrl", "pandas")

MODULES = [
    "robonuke_rl_core.hpc",
    "robonuke_rl_core.hpc.cfg",
    "robonuke_rl_core.hpc.submit",
    "robonuke_rl_core.hpc.launch_train",
    "robonuke_rl_core.hpc.launch_sweep",
    "robonuke_rl_core.hpc.launch_eval",
    "robonuke_rl_core.configfile",
]


def imported_modules(target: str) -> set:
    """Import `target` in a fresh interpreter; return which FORBIDDEN names it pulled in."""
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({target!r})\n"
        f"print(','.join(n for n in {FORBIDDEN!r} if n in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"importing {target} failed:\n{done.stderr}"
    return {name for name in done.stdout.strip().split(",") if name}


@pytest.mark.parametrize("module", MODULES)
def test_the_module_imports_light(module):
    pulled = imported_modules(module)
    assert not pulled, (
        f"{module} imported {sorted(pulled)} at module level. The submitters run on a login "
        "node without the Isaac environment; import heavy things lazily, inside the one "
        "function that needs them (see find_runs)."
    )


def test_the_launchers_run_their_help_without_the_isaac_environment():
    """A whole `--help` exercises argparse and every module-level import on the path."""
    for module in ("launch_train", "launch_sweep", "launch_eval"):
        code = (
            "import sys\n"
            f"from robonuke_rl_core.hpc.{module} import build_parser\n"
            "build_parser().format_help()\n"
            f"print(','.join(n for n in {FORBIDDEN!r} if n in sys.modules))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, f"{module} --help failed:\n{done.stderr}"
        assert done.stdout.strip() == "", f"{module} pulled {done.stdout.strip()}"


def test_wandb_is_imported_lazily_not_at_module_level():
    """`find_runs` is the only path that needs wandb, so it imports it itself."""
    import inspect

    from robonuke_rl_core.hpc.launch_eval import find_runs

    source = inspect.getsource(find_runs)
    assert "import wandb" in source


def test_config_py_is_deliberately_not_import_safe():
    """The reason configfile.py exists, asserted so nobody "simplifies" it away.

    `config.py` registers the `eval` section, which pulls `evaluation.py`, which imports
    torch. That is fine for a training run and fatal on a login node -- hence the split.
    """
    assert "torch" in imported_modules("robonuke_rl_core.config")

"""Prove a project's imports resolve inside the job: Kit up, then import each module.

    python hpc/smoke/import_check.py my_project.tasks furniture_lab.tasks [--task <gym id>]

Run through ``hpc_job.bash`` with the project as ``RNK_PROJECT_ROOT`` (and its further clones in
``RNK_EXTRA_REPOS``): what this checks is exactly the bind + PYTHONPATH layout the real jobs
get, not the image. Each module is imported after Isaac Sim is up (task modules import Isaac
Lab) and its file is printed, so a module silently taken from the wrong place shows. With
``--task`` the env is also built with 2 envs and stepped once, so ``gym.register`` and the
task's own assets are proven too.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="+", help="modules to import, e.g. my_project.tasks furniture_lab.tasks")
    parser.add_argument("--task", default=None, help="a registered gym id to build with 2 envs and step once")
    args = parser.parse_args(argv)

    print(f"[import_check] cwd {os.getcwd()}", flush=True)
    print(f"[import_check] PYTHONPATH {os.environ.get('PYTHONPATH', '')}", flush=True)

    sys.argv = sys.argv[:1]  # Kit parses argv
    from isaaclab.app import AppLauncher

    t0 = time.time()
    AppLauncher(headless=True)
    print(f"[import_check] kit up in {time.time() - t0:.0f}s", flush=True)

    failed = False
    for name in args.modules:
        try:
            module = importlib.import_module(name)
            print(f"[import_check] ok   {name} -> {getattr(module, '__file__', '?')}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report every module, then fail
            failed = True
            print(f"[import_check] FAIL {name}: {type(exc).__name__}: {exc}", flush=True)

    if args.task and not failed:
        import gymnasium as gym
        import torch

        spec = gym.spec(args.task)
        cfg_entry = spec.kwargs.get("env_cfg_entry_point")
        print(f"[import_check] task {args.task}: {spec.entry_point}, cfg {cfg_entry}", flush=True)
        if isinstance(cfg_entry, str):
            module_name, attr = cfg_entry.split(":")
            cfg = getattr(importlib.import_module(module_name), attr)()
        else:
            cfg = cfg_entry()
        cfg.scene.num_envs = 2
        env = gym.make(args.task, cfg=cfg)
        env.reset()
        action = torch.zeros((2, env.unwrapped.single_action_space.shape[0]), device=env.unwrapped.device)
        env.step(action)
        print(f"[import_check] ok   {args.task} built with 2 envs and stepped", flush=True)

    print("[import_check] RESULT " + ("FAIL" if failed else "PASS"), flush=True)
    sys.stdout.flush()
    os._exit(1 if failed else 0)  # Kit's shutdown can hang


if __name__ == "__main__":
    sys.exit(main())

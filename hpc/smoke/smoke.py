"""Smoke test of the image on the GPU it landed on, in tiers.

Runs INSIDE the container, started by ``hpc_job.bash`` with the package clone as the project
root (so the working directory is the package and ``tests/gpu_forge.yaml`` is at hand)::

    python hpc/smoke/smoke.py --out <dir> [--tiers abcde] [--required abcd]

``hpc/verify_partitions.sh`` submits one of these per partition and tabulates the results.

Tiers, each in its own process (Kit can be launched once per process, and a Kit with
cameras is a different launch from one without):

    a   torch sees the GPU and the wheel carries kernels for its architecture
    b   Isaac Sim's Kit boots headless
    c   the package's Forge test env builds and steps
    d   a short training run through scripts/train.py finishes (wandb offline)
    e   a tiled camera renders and an mp4 is written through the package's VideoWriter

``e`` is the recording path. It needs the RTX renderer, so on a GPU without RT cores (H100,
V100) it is expected to fail; it is reported, and only the ``--required`` tiers decide the
exit status. The summary lands in ``<out>/summary.json``, each tier's output in
``<out>/tier_<x>.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

TIERS = {
    "a": ("torch sees the GPU, wheel has its kernels", 300),
    "b": ("Kit boots headless", 900),
    "c": ("Forge env builds and steps", 1200),
    "d": ("short training run (scripts/train.py)", 1800),
    "e": ("camera renders, mp4 written (needs RT cores)", 1200),
}
GPU_CONFIG = Path("tests/gpu_forge.yaml")


def _say(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def _gpu_summary() -> dict:
    info = {"node": platform.node(), "partition": os.environ.get("SLURM_JOB_PARTITION", "")}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout.strip().splitlines()
        info["gpu"] = out[0] if out else "none"
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        info["gpu"] = f"nvidia-smi failed: {exc}"
    return info


# ----------------------------------------------------------------------------------- tiers
def kernels_for(capability, archs):
    """The wheel entry that runs on this GPU: a cubin of the same major architecture and a
    minor at or below the device's (sm_86 runs on an sm_89 L40S), else PTX it can JIT."""
    major, minor = capability
    cubins = []
    for arch in archs:
        kind, number = arch.split("_")
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            cubins.append((arch_minor, arch))
    if cubins:
        return max(cubins)[1]
    for arch in archs:
        kind, number = arch.split("_")
        arch_major, arch_minor = int(number[:-1]), int(number[-1])
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return arch
    return None


def tier_a(out: Path) -> None:
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    capability = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    usable = kernels_for(capability, archs)
    _say(f"torch {torch.__version__} | gpu {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]} "
         f"| archs {archs} | runs {usable}")
    if usable is None:
        raise SystemExit(f"this torch build has no kernels for sm_{capability[0]}{capability[1]}: "
                         "wrong image flavor for this GPU")
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    torch.cuda.synchronize()
    _say(f"matmul ok ({y:.3e})")


def _launch_kit(enable_cameras: bool):
    # Kit parses sys.argv itself and dies on flags it does not know
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    t0 = time.time()
    launcher = AppLauncher(headless=True, enable_cameras=enable_cameras)
    _say(f"kit up in {time.time() - t0:.0f}s (enable_cameras={enable_cameras})")
    return launcher.app


def tier_b(out: Path) -> None:
    _launch_kit(enable_cameras=False)


def _build_env(out: Path, num_envs: int, extra_overrides: list[str]):
    import gymnasium as gym

    from robonuke_rl_core.config import load_config
    from robonuke_rl_core.envs.build import wrap_for_skrl

    cfg = load_config(
        GPU_CONFIG,
        [f"trainer.output_dir={out / 'runs'}", f"task.cfg.scene.num_envs={num_envs}", *extra_overrides],
    )
    t0 = time.time()
    env = wrap_for_skrl(gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.experiment.seed)
    _say(f"env {cfg.task_name} with {num_envs} envs built in {time.time() - t0:.0f}s")
    return cfg, env


def _random_steps(env, steps: int) -> None:
    import torch

    env.reset()
    act_dim = env.action_space.shape[-1]
    t0 = time.time()
    for _ in range(steps):
        action = 2.0 * torch.rand((env.num_envs, act_dim), device=env.device) - 1.0
        env.step(action)
    _say(f"{steps} random steps in {time.time() - t0:.1f}s")


def tier_c(out: Path) -> None:
    _launch_kit(enable_cameras=False)
    _, env = _build_env(out, num_envs=4, extra_overrides=[])
    _random_steps(env, steps=20)


def tier_d(out: Path) -> None:
    runs = out / "runs"
    env = dict(os.environ, WANDB_MODE="offline")
    cmd = [
        sys.executable, "scripts/train.py", "--config", str(GPU_CONFIG), "--headless",
        f"trainer.output_dir={runs}", "wandb.mode=offline", "task.cfg.scene.num_envs=8",
    ]
    _say("running: " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, env=env, timeout=TIERS["d"][1] - 60)
    _say(f"train.py exited {result.returncode} after {time.time() - t0:.0f}s")
    if result.returncode != 0:
        raise SystemExit(f"scripts/train.py exited {result.returncode}")
    produced = [p for p in runs.rglob("*") if p.is_file()]
    if not produced:
        raise SystemExit(f"training wrote nothing under {runs}")
    _say(f"{len(produced)} files under {runs}")


def tier_e(out: Path) -> None:
    try:
        vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=120)
        head = "\n".join(vk.stdout.splitlines()[:40])
        _say(f"vulkaninfo --summary (exit {vk.returncode}):\n{head}\n{vk.stderr[:2000]}")
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        _say(f"vulkaninfo unavailable: {exc}")

    _launch_kit(enable_cameras=True)
    from robonuke_rl_core.recording import (
        VideoWriter, install_recorder_camera, read_camera_rgb, recorder_camera, set_camera_active,
    )
    from robonuke_rl_core.config import load_config

    # the camera must be installed before gym.make, so resolve the cfg first
    cfg = load_config(
        GPU_CONFIG,
        [f"trainer.output_dir={out / 'runs_e'}", "task.cfg.scene.num_envs=2",
         "task.cfg.scene.clone_in_fabric=false"],
    )
    install_recorder_camera(
        cfg.task_name, cfg.task_cfg, width=cfg.eval.video_width, height=cfg.eval.video_height
    )
    import gymnasium as gym
    import torch

    from robonuke_rl_core.envs.build import wrap_for_skrl

    env = wrap_for_skrl(gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.experiment.seed)
    camera = recorder_camera(env)
    set_camera_active(camera, True)
    env.reset()
    path = out / "smoke_camera.mp4"
    writer = VideoWriter(path, fps=10)
    act_dim = env.action_space.shape[-1]
    nonblank = 0
    for _ in range(15):
        env.step(2.0 * torch.rand((env.num_envs, act_dim), device=env.device) - 1.0)
        frame = read_camera_rgb(camera)[0].numpy()
        if int(frame.std()) > 0:
            nonblank += 1
        writer.append(frame)
    writer.close()
    size = path.stat().st_size if path.exists() else 0
    _say(f"{writer.frames} frames, {nonblank} non-blank, {size} bytes -> {path}")
    if writer.frames == 0 or size < 5_000:
        raise SystemExit("no usable mp4 was written")
    if nonblank == 0:
        raise SystemExit("every frame is blank: the camera rendered nothing")


TIER_FUNCS = {"a": tier_a, "b": tier_b, "c": tier_c, "d": tier_d, "e": tier_e}


# ---------------------------------------------------------------------------------- driver
def run_tier(letter: str, out: Path) -> dict:
    label, timeout = TIERS[letter]
    log = out / f"tier_{letter}.log"
    cmd = [sys.executable, os.path.abspath(__file__), "--tier", letter, "--out", str(out)]
    t0 = time.time()
    with open(log, "w") as handle:
        try:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            status = "pass" if proc.returncode == 0 else "fail"
            note = "" if proc.returncode == 0 else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            status, note = "timeout", f"> {timeout}s"
    seconds = round(time.time() - t0, 1)
    _say(f"tier {letter} [{status:7}] {seconds:7.1f}s  {label} {note}")
    return {"tier": letter, "label": label, "status": status, "seconds": seconds, "note": note,
            "log": str(log)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="directory for logs, runs and the summary")
    parser.add_argument("--tiers", default="abcde", help="tiers to run, in order (default abcde)")
    parser.add_argument("--required", default="abcd", help="tiers that must pass (default abcd)")
    parser.add_argument("--tier", default=None, help=argparse.SUPPRESS)  # one tier, in-process
    args = parser.parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.tier:
        if args.tier not in TIER_FUNCS:
            parser.error(f"unknown tier {args.tier!r}")
        TIER_FUNCS[args.tier](out)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # Kit's shutdown can hang; a finished tier has nothing left to do

    unknown = [t for t in args.tiers + args.required if t not in TIERS]
    if unknown:
        parser.error(f"unknown tier(s) {unknown}; known: {''.join(TIERS)}")

    summary = {**_gpu_summary(), "image_flavor": os.environ.get("RNK_IMAGE_FLAVOR", ""),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "tiers": []}
    _say(f"node {summary['node']} partition {summary['partition']!r} gpu {summary['gpu']} "
         f"flavor {summary['image_flavor']!r}")
    for letter in args.tiers:
        summary["tiers"].append(run_tier(letter, out))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    failed_required = [t["tier"] for t in summary["tiers"]
                       if t["tier"] in args.required and t["status"] != "pass"]
    summary["ok"] = not failed_required
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _say("RESULT " + ("PASS" if summary["ok"] else f"FAIL required tier(s) {failed_required}")
         + f" -> {out / 'summary.json'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

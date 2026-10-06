"""Evaluate a trained policy under the conditions an eval config names.

    # the usual way: the policy and its config come from a wandb run
    python scripts/eval.py --run entity/project/run_id \\
        --eval_config examples/eval/quick.yaml --headless \\
        [--checkpoint best|<step>|<file name>] [--record] [overrides...]

    # no wandb: a local run directory with the layout training writes
    python scripts/eval.py --local runs/proj/group/group_a0 \\
        --eval_config examples/eval/quick.yaml --headless

``--run`` takes ``entity/project/<run id or exact run name>`` and pulls the run's
``resolved_config.yaml`` and the requested checkpoint out of its ordinary run files (never the
Artifacts API) into a local cache, which the local path then treats as a run directory.
``--checkpoint`` defaults to the best-return checkpoint and also accepts a step number or an
exact file name.

The run says what was trained; the eval config says what to test it under (``task.cfg.*``, the
env count and the ``eval`` section), and anything it leaves out keeps the trained run's value;
CLI overrides still win. Episodes are collected in rounds: one global reset with fresh random
initial conditions, up to ``max_episode_length`` steps, and each env contributes only its
FIRST episode of the round. Everything lands under
``<run_dir>/eval/<eval-config-stem>_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from robonuke_rl_core.config import dump, load_from_run  # noqa: E402
from robonuke_rl_core.envs.build import build_env, describe, prepare_task  # noqa: E402
from robonuke_rl_core.recording import EvalRecorder, install_recorder_camera  # noqa: E402
from robonuke_rl_core.evaluation import (  # noqa: E402
    SINGLE_AGENT_OVERRIDE,
    build_eval_policy,
    fetch_wandb_run,
    find_checkpoint,
    resolved_config_path,
    run_eval,
    upload_eval_results,
)


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run",
        type=str,
        default=None,
        help="wandb run as entity/project/<run id or name> (the usual source)",
    )
    source.add_argument(
        "--local",
        type=str,
        default=None,
        help="skip wandb: an agent's local run dir (holds checkpoints/)",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        required=True,
        help="eval config YAML layered on the run's resolved config (required)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="'best', a step number, or a checkpoint file name (default: best)",
    )
    parser.add_argument(
        "--record", action="store_true", help="force eval.record on for this run"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="where --run downloads land (default: ~/.cache/robonuke_rl_core)",
    )
    return parser


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_eval_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    # Rendering is enabled unconditionally: a recorder camera has to be in the scene before
    # the env is built, but whether we record is only known once the config is loaded, which
    # needs the task registry, which needs the app already running.
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    if setup is not None:
        # Isaac Sim is up: a project imports its tasks (gym.register) and registers its
        # config sections, losses, overlays and model architectures here, before the
        # config loads.
        setup()


    # Isaac Sim is up: the task registry and the env can be imported now
    import gymnasium as gym
    import torch
    from skrl.envs.wrappers.torch import wrap_env

    if args.run:
        print(f"[eval] fetching {args.run} from wandb", flush=True)
        run_dir = fetch_wandb_run(args.run, args.checkpoint, args.cache_dir)
        print(f"[eval] run files: {run_dir}", flush=True)
    else:
        run_dir = Path(args.local).expanduser()
    config_path = resolved_config_path(run_dir)
    checkpoint = find_checkpoint(run_dir, args.checkpoint)

    # eval runs ONE agent whatever the run trained, so the config has to say so
    cfg = load_from_run(
        config_path,
        list(overrides) + [SINGLE_AGENT_OVERRIDE],
        extra_files=[args.eval_config],
    )
    record = bool(cfg.eval.record or args.record)

    stem = Path(args.eval_config).stem
    out_dir = run_dir / "eval" / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"

    if record:
        # before gym.make: the camera has to exist in the scene the task clones
        install_recorder_camera(
            cfg.task_name,
            cfg.task_cfg,
            width=cfg.eval.video_width,
            height=cfg.eval.video_height,
        )
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    # the eval's own resolved config, written from the live env cfg: this also raises if the
    # env discarded any task.cfg value the eval config set (check_env_kept_overrides)
    print(f"[eval] config: {dump(cfg, out_dir, env.unwrapped.cfg)}", flush=True)
    max_episode_length = int(env.unwrapped.max_episode_length)
    env = build_env(cfg, env, cfg.task_name)
    if describe(cfg):
        print(f"[eval] env wrappers: {' -> '.join(describe(cfg))}", flush=True)
    env = wrap_env(env, wrapper="isaaclab")

    torch.manual_seed(cfg.experiment.seed)
    policy, learner, metadata = build_eval_policy(
        cfg, env, checkpoint, deterministic=cfg.eval.deterministic
    )

    print(
        f"[eval] {cfg.trainer.learner} from {checkpoint.name} (step {metadata['step']}, "
        f"mean_return {metadata['mean_return']:.3f})",
        flush=True,
    )
    print(
        f"[eval] {cfg.eval.num_rollouts} episodes over {env.num_envs} envs x "
        f"{max_episode_length} steps/round, "
        f"{'deterministic' if cfg.eval.deterministic else 'sampled'} actions",
        flush=True,
    )
    recorder = None
    if record:
        recorder = EvalRecorder(
            env,
            output_dir=out_dir / "videos",
            fps=cfg.eval.video_fps,
            overlays=cfg.eval.overlays,
        )
        print(
            f"[eval] recording {cfg.eval.video_width}x{cfg.eval.video_height} @ "
            f"{cfg.eval.video_fps} fps, overlays={list(cfg.eval.overlays)}",
            flush=True,
        )

    try:
        result = run_eval(
            env=env,
            policy=policy,
            num_rollouts=cfg.eval.num_rollouts,
            max_episode_length=max_episode_length,
            seed=cfg.experiment.seed,
            save_state=cfg.eval.save_state,
            recorder=recorder,
            device=env.device,
        )
    finally:
        if recorder is not None:
            recorder.close()  # never leave an mp4 half-written

    write_outputs(out_dir, result, cfg, checkpoint, metadata, stem)
    if recorder is not None:
        print(
            f"[eval] videos: {len(recorder.paths)} files in {out_dir / 'videos'} "
            f"({sum(recorder.frame_counts.values())} frames)",
            flush=True,
        )
    print(f"[eval] wrote {out_dir}", flush=True)
    for name, value in sorted(result.summary.items()):
        print(f"[eval]   {name}: {value}", flush=True)

    if args.run:
        names = upload_eval_results(args.run, out_dir, run_dir)
        print(f"[eval] uploaded to {args.run}: {', '.join(names)}", flush=True)
    else:
        print("[eval] --local: results stay on disk, nothing uploaded", flush=True)

    env.close()
    app_launcher.app.close()
    return 0


def write_outputs(out_dir, result, cfg, checkpoint, metadata, stem: str) -> None:
    """``summary.yaml`` plus the gathered per-step trace, both plain files on disk."""
    from omegaconf import OmegaConf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "meta": {
            "checkpoint": str(checkpoint),
            "checkpoint_step": metadata["step"],
            "trained_agent": metadata["agent_idx"],
            "eval_config": stem,
            "learner": cfg.trainer.learner,
            "num_rollouts": cfg.eval.num_rollouts,
            "deterministic": cfg.eval.deterministic,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": result.summary,
        "episodes": result.episodes,
    }
    (out_dir / "summary.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(document)))
    if result.state is not None:
        path = result.state.write_parquet(out_dir / f"{stem}.parquet")
        print(f"[eval] trace: {path} ({result.state.rows} rows)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

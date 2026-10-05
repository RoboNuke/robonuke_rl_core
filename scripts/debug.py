"""Watch a trained policy, or watch what the env spawns. One env, no data collection.

    # live viewer (GUI): the policy acts, you watch
    python scripts/debug.py --run entity/project/run_id
    python scripts/debug.py --local runs/proj/group/group_a0 --checkpoint 39

    # reset viewer: hold each sampled initial condition still and look at it
    python scripts/debug.py --run entity/project/run_id --resets --hold_seconds 2

    # the same, to an mp4, no window
    python scripts/debug.py --run entity/project/run_id --resets --headless \\
        --num_resets 10 --out resets.mp4

Keys are **j = pause/resume, k = reset, l = quit** in both viewers, and the mapping is printed
when the viewer starts. They avoid the letters Omniverse binds in the viewport (``p`` is
Parent Prim, ``w/e/r`` the gizmos, ``f`` frame-selected, and so on): Kit consumes those before
this script sees them, so the viewer would appear to ignore the key while Kit popped up a
toast. Change any of them with ``--key_pause`` / ``--key_reset`` / ``--key_quit`` if one still
collides on your setup. Ctrl+C also exits cleanly.

The policy, the config and the override layering are exactly eval's (an ``--eval_config`` is
optional here), except that debug always runs **one** env.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robonuke_rl_core.config import load_from_run  # noqa: E402
from robonuke_rl_core.evaluation import (  # noqa: E402
    build_eval_policy,
    fetch_wandb_run,
    find_checkpoint,
    force_env_reset,
    resolved_config_path,
)
from robonuke_rl_core.envs.build import build_env, describe, prepare_task  # noqa: E402
from robonuke_rl_core.recording import (  # noqa: E402
    install_recorder_camera,
    render_resets,
)


def add_debug_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", type=str, default=None, help="wandb run as entity/project/<run id or name>"
    )
    source.add_argument(
        "--local", type=str, default=None, help="skip wandb: an agent's local run dir"
    )
    parser.add_argument(
        "--eval_config", type=str, default=None, help="optional config layer, as in eval.py"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="best", help="'best', a step, or a file name"
    )
    parser.add_argument(
        "--resets", action="store_true", help="reset viewer instead of the live viewer"
    )
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=2.0,
        help="reset viewer: how long to hold each sampled condition (default 2s)",
    )
    parser.add_argument(
        "--num_resets", type=int, default=10, help="headless reset viewer: resets to render"
    )
    parser.add_argument(
        "--out", type=str, default="resets.mp4", help="headless reset viewer: the mp4 to write"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="where --run downloads land"
    )
    parser.add_argument("--key_pause", default="j", help="live viewer: pause/resume (default j)")
    parser.add_argument("--key_reset", default="k", help="reset now (default k)")
    parser.add_argument("--key_quit", default="l", help="quit (default l)")
    return parser


class Keys:
    """Keyboard presses as flags, through the carb input interface.

    Isaac Lab's own devices subscribe the same way (see
    ``isaaclab/devices/keyboard/se2_keyboard.py``): acquire the input interface, take the app
    window's keyboard, and subscribe. Nothing here runs headless — the caller skips it.
    """

    def __init__(self, bindings: dict) -> None:
        import carb
        import omni

        self._carb = carb
        self.pressed = {name: False for name in bindings.values()}
        self._bindings = {key.upper(): name for key, name in bindings.items()}
        self._seen_any = False
        self._window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._window.get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )

    def _on_event(self, event, *args) -> bool:
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if not self._seen_any:
                # proof that key events reach this script at all: if a mapped key seems dead
                # but this line never appeared, Kit is consuming the event, not us
                self._seen_any = True
                print(f"[debug] keyboard connected (first key: {event.input.name})", flush=True)
            name = self._bindings.get(event.input.name)
            if name is not None:
                self.pressed[name] = True
        # True = handled. Kit's own hotkey helper does the same; our keys are chosen so they
        # are not ones the viewport already claims.
        return True

    def take(self, name: str) -> bool:
        """True once per press."""
        if self.pressed.get(name):
            self.pressed[name] = False
            return True
        return False

    def close(self) -> None:
        # the interface spells it `unsubscribe_to_...`; Isaac Lab's own device calls
        # `unsubscribe_from_...` in a __del__, where the AttributeError is swallowed
        unsubscribe = getattr(self._input, "unsubscribe_to_keyboard_events", None) or getattr(
            self._input, "unsubscribe_from_keyboard_events"
        )
        unsubscribe(self._keyboard, self._subscription)

    @staticmethod
    def announce(bindings: dict, what: str) -> None:
        """Print the key map, so it never has to be looked up in the source."""
        width = max(len(name) for name in bindings.values())
        print(f"[debug] {what} keys:", flush=True)
        for key, name in bindings.items():
            print(f"[debug]   {key.lower()}  {name:<{width}}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_debug_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    headless = bool(getattr(args, "headless", False))
    if headless and not args.resets:
        parser.error("the live viewer needs a window; drop --headless or pass --resets")
    if headless:
        args.enable_cameras = True  # the reset mp4 is rendered through a camera
    app_launcher = AppLauncher(args)

    import gymnasium as gym
    import torch
    from skrl.envs.wrappers.torch import wrap_env

    if args.run:
        print(f"[debug] fetching {args.run} from wandb", flush=True)
        run_dir = fetch_wandb_run(args.run, args.checkpoint, args.cache_dir)
    else:
        run_dir = Path(args.local).expanduser()
    config_path = resolved_config_path(run_dir)
    checkpoint = find_checkpoint(run_dir, args.checkpoint)

    extra = [args.eval_config] if args.eval_config else []
    # debug watches one env; an eval config's env count would only confuse the view
    injected = ["task.cfg.scene.num_envs=1"]
    if headless:
        # the reset video needs a real per-env camera prim. Unlike eval -- where recording is
        # a deliberate choice the eval config states -- debug has no config of its own, so it
        # sets this as a CLI-layer override, which is recorded like any other.
        injected.append("task.cfg.scene.clone_in_fabric=false")
    cfg = load_from_run(config_path, list(overrides) + injected, extra_files=extra)
    print(f"[debug] overrides: {' '.join(injected)}", flush=True)

    if headless:
        install_recorder_camera(
            cfg.task_name,
            cfg.task_cfg,
            width=cfg.eval.video_width,
            height=cfg.eval.video_height,
        )
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    env = gym.make(cfg.task_name, cfg=cfg.task_cfg, render_mode=None if headless else "human")
    # the same stack training uses, so what you watch is what trains
    env = build_env(cfg, env, cfg.task_name)
    if describe(cfg):
        print(f"[debug] env wrappers: {' -> '.join(describe(cfg))}", flush=True)
    env = wrap_env(env, wrapper="isaaclab")
    torch.manual_seed(cfg.experiment.seed)

    policy, _, metadata = build_eval_policy(
        cfg, env, checkpoint, deterministic=cfg.eval.deterministic
    )
    print(
        f"[debug] {cfg.trainer.learner} from {checkpoint.name} (step {metadata['step']})",
        flush=True,
    )

    try:
        if args.resets and headless:
            writer = render_resets(
                env,
                args.out,
                num_resets=args.num_resets,
                hold_seconds=args.hold_seconds,
                fps=cfg.eval.video_fps,
                overlays=cfg.eval.overlays,
            )
            print(
                f"[debug] wrote {writer.path} ({writer.frames} frames = {args.num_resets} "
                f"resets x {args.hold_seconds:g}s at {cfg.eval.video_fps} fps)",
                flush=True,
            )
        elif args.resets:
            reset_viewer(
                env,
                app_launcher.app,
                {args.key_reset: "reset", args.key_quit: "quit"},
                hold_seconds=args.hold_seconds,
            )
        else:
            live_viewer(
                env,
                app_launcher.app,
                policy,
                {
                    args.key_pause: "pause",
                    args.key_reset: "reset",
                    args.key_quit: "quit",
                },
            )
    finally:
        env.close()
        app_launcher.app.close()
    return 0


def live_viewer(env, app, policy, bindings: dict) -> None:
    """Step the policy forever, pausing and resetting on the mapped keys."""
    keys = Keys(bindings)
    Keys.announce(bindings, "live viewer")

    observations, states = force_env_reset(env)
    simulation = env.unwrapped.sim
    paused = False
    try:
        while app.is_running():
            if keys.take("quit"):
                print("[debug] quit", flush=True)
                break
            if keys.take("reset"):
                observations, states = force_env_reset(env)
                print("[debug] reset", flush=True)
            if keys.take("pause"):
                paused = not paused
                print(f"[debug] {'paused' if paused else 'resumed'}", flush=True)
            if paused:
                # sim.render() is app.update() with "/app/player/playSimulations" off, so the
                # viewport stays live and interactive while physics does NOT advance. A bare
                # app.update() here would keep stepping physics — Isaac Lab says as much in
                # the FIXME inside SimulationContext.step().
                simulation.render()
                continue
            actions = policy(observations, states)
            observations, _, _, _, _ = env.step(actions)
            states = env.state() if hasattr(env, "state") else None
    except KeyboardInterrupt:
        print("[debug] interrupted", flush=True)
    finally:
        keys.close()


def reset_viewer(env, app, bindings: dict, *, hold_seconds: float) -> None:
    """Resample initial conditions on a timer. ``r`` resets now, ``q`` quits.

    Between resets the sim is rendered but never stepped, so the pose on screen is exactly the
    one the env sampled.
    """
    keys = Keys(bindings)
    print(f"[debug] reset viewer: a reset every {hold_seconds:g}s", flush=True)
    Keys.announce(bindings, "reset viewer")
    simulation = env.unwrapped.sim
    # wall-clock, not a render count: how long a render takes is not a fixed unit of time
    held_since = None
    count = 0
    try:
        while app.is_running():
            if keys.take("quit"):
                print(f"[debug] quit after {count} resets", flush=True)
                break
            due = held_since is None or (time.monotonic() - held_since) >= hold_seconds
            if keys.take("reset") or due:
                force_env_reset(env)
                count += 1
                held_since = time.monotonic()
                print(f"[debug] reset {count}", flush=True)
            # renders and pumps the UI without stepping physics (see live_viewer)
            simulation.render()
    except KeyboardInterrupt:
        print(f"[debug] interrupted after {count} resets", flush=True)
    finally:
        keys.close()


if __name__ == "__main__":
    raise SystemExit(main())

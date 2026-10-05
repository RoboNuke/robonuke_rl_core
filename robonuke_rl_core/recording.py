"""Per-env video recording for eval: one mp4 per (round, env), with overlays.

Cut down from the recorder in RoboNuke/generalized_hybrid_vic_action_space
(``wrappers/recording.py`` + ``wrappers/recording_grid.py``). What is kept: the per-env
``TiledCamera`` and its placement, the pre-clone spawn shim that makes the camera exist in
every env, the RGB read, and the H.264 writer settings that actually play back in a browser.
What is dropped: the 3x4 best/median/worst grid, the critic Q-value overlay, the
every-K-resets training cadence, and the TensorBoard video — eval records whole rounds and
writes one file per env, so none of that applies.

Frames are written as they arrive (one open writer per env), so memory stays at one frame per
env rather than a whole round. An env's video holds exactly the steps its episode contributed
to the data: the recorder is driven by the same valid mask as the accounting and the trace.

Overlays are a registry, like the losses: subclass :class:`Overlay`, set ``name``, decorate
with :func:`register_overlay`, and name it in ``eval.overlays``. The package ships ``hud``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CAMERA_KEY",
    "CameraPlacement",
    "install_recorder_camera",
    "set_camera_active",
    "read_camera_rgb",
    "StepData",
    "Overlay",
    "OVERLAYS",
    "register_overlay",
    "build_overlays",
    "HudOverlay",
    "EvalRecorder",
    "VideoWriter",
    "render_resets",
]

#: scene-sensor name the camera is registered under
CAMERA_KEY = "recorder_camera"


@dataclass
class CameraPlacement:
    """Where the per-env camera sits and how it sees. Defaults are the ported ones."""

    pos: tuple = (1.0, 0.0, 0.35)
    #: (w, x, y, z) under the ROS convention
    quat: tuple = (-0.3535534, 0.6123724, 0.6123724, -0.3535534)
    focal_length: float = 24.0
    focus_distance: float = 0.05
    horizontal_aperture: float = 20.955
    clipping_range: tuple = (0.1, 20.0)


# ------------------------------------------------------------------------------- the camera
def install_recorder_camera(
    task_name: str,
    env_cfg: Any,
    *,
    width: int,
    height: int,
    placement: Optional[CameraPlacement] = None,
) -> None:
    """Attach a per-env ``TiledCamera`` to the task's scene. Call **before** ``gym.make``.

    The camera is a rendering sensor, so two things have to be true and neither is the
    default. Under ``clone_in_fabric=True`` the cloned cameras are Fabric-only — the view then
    resolves env 0 alone and ``scene.reset(env_ids)`` indexes an env_0-sized buffer out of
    bounds (a device-side assert) — so the config must turn fabric cloning off. This function
    *checks* that rather than silently flipping it: the resolved config has to say what the
    env ran with, and ``dump`` raises when the env holds a value a layer did not set. Second,
    the camera is spawned *between* the task's own asset spawns and its ``clone_environments``
    call, by shimming that call once, so the clone replicates the camera into every env.
    """
    import importlib

    import gymnasium as gym
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaaclab.sim.spawners.sensors import PinholeCameraCfg

    placement = placement or CameraPlacement()

    if getattr(env_cfg.scene, "clone_in_fabric", False):
        raise ValueError(
            "recording needs task.cfg.scene.clone_in_fabric: false — a TiledCamera needs real "
            "per-env prims, and under Fabric cloning the per-env cameras resolve env 0 alone "
            "(scene.reset then indexes an env_0-sized buffer out of bounds). Add it to the "
            "eval config:\n"
            "  task:\n    cfg:\n      scene:\n        clone_in_fabric: false"
        )

    camera_cfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/{CAMERA_KEY}",
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(placement.pos), rot=tuple(placement.quat), convention="ros"
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=float(placement.focal_length),
            focus_distance=float(placement.focus_distance),
            horizontal_aperture=float(placement.horizontal_aperture),
            clipping_range=tuple(placement.clipping_range),
        ),
        width=int(width),
        height=int(height),
        update_period=0.0,  # every step
    )

    entry_point = gym.spec(task_name).entry_point
    module_name, _, class_name = str(entry_point).partition(":")
    if not class_name:
        raise RuntimeError(
            f"cannot attach the recorder camera: task {task_name!r} has entry point "
            f"{entry_point!r}, which is not 'module:Class'"
        )
    env_class = getattr(importlib.import_module(module_name), class_name)
    original_setup_scene = env_class._setup_scene

    def patched_setup_scene(self):
        original_clone = self.scene.clone_environments

        def shim_clone(*args, **kwargs):
            self.scene.clone_environments = original_clone  # fires exactly once
            camera = TiledCamera(camera_cfg)
            result = original_clone(*args, **kwargs)
            self.scene._sensors[CAMERA_KEY] = camera
            print(
                f"[recorder] TiledCamera {width}x{height} registered as "
                f"scene.sensors[{CAMERA_KEY!r}] after env clone",
                flush=True,
            )
            return result

        self.scene.clone_environments = shim_clone
        try:
            return original_setup_scene(self)
        finally:
            env_class._setup_scene = original_setup_scene  # un-patch the class

    env_class._setup_scene = patched_setup_scene


def recorder_camera(env: Any) -> Any:
    """The installed camera sensor, or a loud error saying how to install it."""
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    sensors = getattr(scene, "sensors", None) or {}
    if CAMERA_KEY not in sensors:
        raise RuntimeError(
            f"no camera at scene.sensors[{CAMERA_KEY!r}]: call install_recorder_camera(...) "
            "before gym.make() when recording is on"
        )
    return sensors[CAMERA_KEY]


def set_camera_active(camera: Any, active: bool) -> None:
    """Toggle rasterization by moving ``update_period`` (0 = every step, huge = off)."""
    target = 0.0 if active else 1.0e9
    for owner in (camera, getattr(camera, "cfg", None)):
        if owner is not None and hasattr(owner, "update_period"):
            owner.update_period = target


def read_camera_rgb(camera: Any) -> Any:
    """The camera's latest RGB as a CPU uint8 ``(num_envs, H, W, 3)`` tensor."""
    import torch

    rgb = camera.data.output["rgb"]
    if rgb.dim() != 4:
        raise RuntimeError(
            f"the recorder camera returned shape {tuple(rgb.shape)}; expected (N, H, W, C)"
        )
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]  # drop alpha
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return rgb.detach().to("cpu")


# ------------------------------------------------------------------------------- the writer
class VideoWriter:
    """One open mp4. Append ``(H, W, 3)`` uint8 frames, then close.

    H.264 in ``yuv420p`` needs even frame dimensions to stay playable in a browser or
    QuickTime, and ``macro_block_size=1`` stops ffmpeg padding small frames on its own; a
    stray odd row or column is trimmed instead.
    """

    def __init__(self, path: Any, fps: int = 30) -> None:
        from pathlib import Path

        import imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self._writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            fps=max(1, int(fps)),
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pixelformat="yuv420p",
        )

    def append(self, frame: Any) -> None:
        self._writer.append_data(_even(frame))
        self.frames += 1

    def close(self) -> None:
        self._writer.close()


# ------------------------------------------------------------------------------- overlays
@dataclass
class StepData:
    """What an overlay knows about one env at one step."""

    env: int
    step: int
    reward: float
    #: return so far, this episode
    total_reward: float
    valid: bool
    terminated: bool
    truncated: bool
    #: that env's value for each per-step env metric
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def outcome(self) -> Optional[str]:
        """``terminal``, ``timeout``, or None while the episode is still running."""
        if not self.done:
            return None
        return "timeout" if self.truncated else "terminal"


class Overlay:
    """Base class for a frame overlay. Constructed with the config's kwargs."""

    name: str = ""

    def __init__(self, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} takes no kwargs, got {sorted(kwargs)}"
            )

    def apply(self, frame: Any, step: StepData) -> Any:
        """Return the frame to write: an ``(H, W, 3)`` uint8 numpy array."""
        raise NotImplementedError


#: registered overlays, keyed by ``Overlay.name``
OVERLAYS: Dict[str, type] = {}


def register_overlay(cls: type) -> type:
    """Class decorator: register an :class:`Overlay` subclass under its ``name``."""
    if not issubclass(cls, Overlay):
        raise TypeError(f"{cls.__name__} must subclass Overlay")
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    if cls.name in OVERLAYS:
        taken = OVERLAYS[cls.name]
        raise ValueError(
            f"overlay name {cls.name!r} is already registered as {taken.__module__}:"
            f"{taken.__qualname__}; refusing to replace it with {cls.__qualname__}"
        )
    OVERLAYS[cls.name] = cls
    return cls


def build_overlays(names: Sequence[str]) -> List[Overlay]:
    """Instantiate ``names`` in order. An unknown name raises, listing what is registered."""
    built = []
    for name in names:
        if name not in OVERLAYS:
            raise ValueError(
                f"unknown overlay {name!r} in eval.overlays; registered: "
                f"{sorted(OVERLAYS) or '<none>'}. Projects register their own with "
                "@register_overlay before loading the config."
            )
        built.append(OVERLAYS[name]())
    return built


@register_overlay
class HudOverlay(Overlay):
    """Step, reward and return in a banner across the top; the outcome when it ends."""

    name = "hud"
    #: rows the banner owns; nothing outside it is touched
    banner_height = 18

    def apply(self, frame: Any, step: StepData) -> Any:
        import numpy as np
        from PIL import Image, ImageDraw

        image = Image.fromarray(np.ascontiguousarray(frame))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), (image.width - 1, self.banner_height - 1)], fill=(0, 0, 0))
        text = f"t={step.step:03d} r={step.reward:+.2f} R={step.total_reward:+.2f}"
        outcome = step.outcome
        if outcome:
            text = f"{text} {outcome.upper()}"
        draw.text((3, 3), text, fill=(0, 255, 0))
        return np.asarray(image)


# ------------------------------------------------------------------------------- the recorder
class EvalRecorder:
    """One mp4 per (round, env), written as the round runs.

    Follows the same ``start_round`` / ``capture`` / ``end_round`` protocol as the state
    writer and sees the same valid mask, so a video holds exactly the frames behind that
    env's episode. Memory is one frame per env: files are appended to, not buffered.
    """

    def __init__(
        self,
        env: Any,
        *,
        output_dir: Any,
        fps: int = 30,
        overlays: Sequence[str] = ("hud",),
        camera: Any = None,
    ) -> None:
        from pathlib import Path

        self.env = env
        self.num_envs = int(env.num_envs)
        self.camera = camera if camera is not None else recorder_camera(env)
        self.output_dir = Path(output_dir)
        self.fps = int(fps)
        self.overlays = build_overlays(list(overlays))
        #: ``{(round, env): frames written}``
        self.frame_counts: Dict[tuple, int] = {}
        self.paths: Dict[tuple, Any] = {}
        self._round: Optional[int] = None
        self._writers: Dict[int, Any] = {}
        self._step = 0
        self._returns: Dict[int, float] = {}

    # ------------------------------------------------------------------ the round
    def start_round(self, round_index: int) -> None:
        if self._round is not None:
            raise RuntimeError(f"round {self._round} is still open; call end_round() first")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._round = int(round_index)
        self._writers = {}
        self._step = 0
        self._returns = {}
        set_camera_active(self.camera, True)

    def capture(
        self,
        valid: Any,
        *,
        rewards: Any = None,
        terminated: Any = None,
        truncated: Any = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write this step's frame for every env still on its first episode."""
        if self._round is None:
            raise RuntimeError("call start_round() before capture()")
        frames = read_camera_rgb(self.camera)
        if frames.shape[0] != self.num_envs:
            raise RuntimeError(
                f"the camera returned {frames.shape[0]} frames for {self.num_envs} envs; the "
                "camera must be spawned per env (see install_recorder_camera)"
            )
        keep = [index for index, flag in enumerate(valid.reshape(-1).tolist()) if flag]
        if keep:
            rewards_list = _flat(rewards, self.num_envs)
            terminated_list = _flat(terminated, self.num_envs)
            truncated_list = _flat(truncated, self.num_envs)
            metric_lists = {
                name: _flat(value, self.num_envs) for name, value in (metrics or {}).items()
            }
            numpy_frames = frames.numpy()
            for env in keep:
                reward = float(rewards_list[env]) if rewards_list else 0.0
                self._returns[env] = self._returns.get(env, 0.0) + reward
                step = StepData(
                    env=env,
                    step=self._step,
                    reward=reward,
                    total_reward=self._returns[env],
                    valid=True,
                    terminated=bool(terminated_list[env]) if terminated_list else False,
                    truncated=bool(truncated_list[env]) if truncated_list else False,
                    metrics={
                        name: float(values[env]) for name, values in metric_lists.items()
                    },
                )
                frame = numpy_frames[env]
                for overlay in self.overlays:
                    frame = overlay.apply(frame, step)
                self._writer(env).append(frame)
                key = (self._round, env)
                self.frame_counts[key] = self.frame_counts.get(key, 0) + 1
        self._step += 1

    def end_round(self) -> None:
        if self._round is None:
            raise RuntimeError("call start_round() before end_round()")
        for writer in self._writers.values():
            writer.close()
        set_camera_active(self.camera, False)
        self._writers = {}
        self._round = None

    def close(self) -> None:
        """Close an open round, if any — safe to call on an error path."""
        if self._round is not None:
            self.end_round()

    # ------------------------------------------------------------------ internals
    def _writer(self, env: int) -> VideoWriter:
        """The open mp4 writer for this env, created on its first frame."""
        if env not in self._writers:
            path = self.output_dir / f"round{self._round}_env{env}.mp4"
            self._writers[env] = VideoWriter(path, self.fps)
            self.paths[(self._round, env)] = path
        return self._writers[env]


def _flat(value: Any, num_envs: int) -> list:
    """A per-env value as a flat python list, or [] when it was not given."""
    if value is None:
        return []
    try:
        flat = value.reshape(-1).tolist()
    except AttributeError:
        flat = list(value)
    if len(flat) != num_envs:
        raise ValueError(f"expected {num_envs} per-env values, got {len(flat)}")
    return flat


def _even(frame: Any) -> Any:
    """Trim a stray odd row or column: H.264 in yuv420p needs even dimensions."""
    import numpy as np

    height, width = frame.shape[0], frame.shape[1]
    trimmed = frame[: height - height % 2, : width - width % 2]
    return np.ascontiguousarray(trimmed)


def render_resets(
    env: Any,
    out: Any,
    *,
    num_resets: int,
    hold_seconds: float,
    fps: int = 30,
    overlays: Sequence[str] = (),
    env_index: int = 0,
    camera: Any = None,
) -> "VideoWriter":
    """Reset, hold still, repeat — into one mp4. Used by ``debug.py --resets --headless``.

    Between resets the sim is **rendered without being stepped**, so what the frames show is
    exactly the sampled initial condition rather than whatever the physics drifted into. Each
    reset is held for ``hold_seconds`` of video, which at ``fps`` is
    ``round(hold_seconds * fps)`` frames — the playing time is what was asked for, exactly.
    """
    import torch

    from .evaluation import force_env_reset

    if num_resets < 1:
        raise ValueError(f"num_resets must be >= 1, got {num_resets}")
    if hold_seconds <= 0:
        raise ValueError(f"hold_seconds must be > 0, got {hold_seconds}")
    frames_per_reset = max(1, round(float(hold_seconds) * max(1, int(fps))))
    sensor = camera if camera is not None else recorder_camera(env)
    drawings = build_overlays(list(overlays))
    writer = VideoWriter(out, fps)
    simulation = getattr(env, "unwrapped", env).sim
    set_camera_active(sensor, True)
    try:
        for index in range(num_resets):
            force_env_reset(env)
            for tick in range(frames_per_reset):
                simulation.render()  # no physics step: the scene holds its reset pose
                sensor.update(0.0, force_recompute=True)
                frame = read_camera_rgb(sensor).numpy()[env_index]
                step = StepData(
                    env=env_index,
                    step=tick,
                    reward=0.0,
                    total_reward=0.0,
                    valid=True,
                    terminated=False,
                    truncated=False,
                    metrics={"reset": float(index)},
                )
                for overlay in drawings:
                    frame = overlay.apply(frame, step)
                writer.append(frame)
    finally:
        writer.close()
        set_camera_active(sensor, False)
    return writer

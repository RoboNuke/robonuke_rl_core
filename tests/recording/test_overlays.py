"""Overlays and the mp4 writer, on synthetic frames. No Isaac Lab, no camera.

What is pinned here: the registry's rules, that `hud` keeps to its banner, that overlays draw
in the order the config lists, and that the recorder turns valid frames into one playable mp4
per (round, env) with the right frame counts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robonuke_rl_core.recording import (
    OVERLAYS,
    EvalRecorder,
    HudOverlay,
    Overlay,
    StepData,
    build_overlays,
    register_overlay,
)

HEIGHT, WIDTH, NUM_ENVS = 180, 240, 3  # the eval config's video defaults


def frame(value: int = 7) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def step_data(**overrides) -> StepData:
    fields = dict(
        env=0, step=3, reward=1.5, total_reward=4.5, valid=True, terminated=False,
        truncated=False,
    )
    fields.update(overrides)
    return StepData(**fields)


@pytest.fixture
def clean_registry():
    """Tests that register overlays must not leak them into the rest of the suite."""
    known = dict(OVERLAYS)
    yield
    OVERLAYS.clear()
    OVERLAYS.update(known)


# ------------------------------------------------------------------ the registry
def test_an_unknown_overlay_raises_listing_what_is_registered():
    with pytest.raises(ValueError) as err:
        build_overlays(["hud", "nope"])
    assert "nope" in str(err.value) and "hud" in str(err.value)


def test_a_duplicate_name_is_refused(clean_registry):
    class First(Overlay):
        name = "mine"

        def apply(self, frame, step):
            return frame

    register_overlay(First)

    class Second(Overlay):
        name = "mine"

        def apply(self, frame, step):
            return frame

    with pytest.raises(ValueError) as err:
        register_overlay(Second)
    assert "already registered" in str(err.value)


def test_a_bad_overlay_class_is_refused(clean_registry):
    class NotAnOverlay:
        name = "x"

    with pytest.raises(TypeError):
        register_overlay(NotAnOverlay)

    class Nameless(Overlay):
        def apply(self, frame, step):
            return frame

    with pytest.raises(ValueError):
        register_overlay(Nameless)

    class Unimplemented(Overlay):
        name = "todo"

    register_overlay(Unimplemented)
    with pytest.raises(NotImplementedError):
        Unimplemented().apply(frame(), step_data())


def test_overlay_order_follows_the_config_list(clean_registry):
    marks: list[str] = []

    def make(tag: str) -> type:
        class Tagged(Overlay):
            name = tag

            def apply(self, frame, step):
                marks.append(tag)
                return frame

        Tagged.__name__ = f"Overlay{tag}"
        return Tagged

    for tag in ("a", "b"):
        register_overlay(make(tag))

    for overlay in build_overlays(["b", "a", "hud"]):
        overlay.apply(frame(), step_data())
    assert marks == ["b", "a"]
    assert [type(o).name for o in build_overlays(["a", "b"])] == ["a", "b"]


# ------------------------------------------------------------------ the hud
def test_the_hud_draws_only_inside_its_banner():
    original = frame(7)
    drawn = HudOverlay().apply(original, step_data())

    assert drawn.shape == original.shape and drawn.dtype == np.uint8
    banner = HudOverlay.banner_height
    assert not np.array_equal(drawn[:banner], original[:banner])  # it drew something
    assert np.array_equal(drawn[banner:], original[banner:])  # and nothing below
    assert np.array_equal(original, frame(7))  # the input is not modified in place


def test_the_hud_marks_the_outcome_only_when_the_episode_ends():
    running = HudOverlay().apply(frame(), step_data())
    timeout = HudOverlay().apply(frame(), step_data(terminated=True, truncated=True))
    terminal = HudOverlay().apply(frame(), step_data(terminated=True))

    assert step_data().outcome is None
    assert step_data(terminated=True, truncated=True).outcome == "timeout"
    assert step_data(terminated=True).outcome == "terminal"
    # the three banners differ, so the tag really reaches the pixels
    assert not np.array_equal(running, timeout)
    assert not np.array_equal(timeout, terminal)


# ------------------------------------------------------------------ the recorder
class FakeCamera:
    """Stands in for the TiledCamera: per-env RGB plus the update_period toggle."""

    def __init__(self, num_envs: int = NUM_ENVS, alpha: bool = False, float_rgb: bool = False):
        channels = 4 if alpha else 3
        data = torch.arange(num_envs, dtype=torch.float32).view(num_envs, 1, 1, 1)
        rgb = data.expand(num_envs, HEIGHT, WIDTH, channels).clone()
        self.data = type("Data", (), {"output": {"rgb": rgb / 255.0 if float_rgb else rgb.to(torch.uint8)}})()
        self.update_period = 1.0e9


class FakeEnv:
    def __init__(self, num_envs: int = NUM_ENVS):
        self.num_envs = num_envs


def recorder(tmp_path, camera=None, overlays=("hud",)) -> EvalRecorder:
    return EvalRecorder(
        FakeEnv(),
        output_dir=tmp_path / "videos",
        fps=10,
        overlays=overlays,
        camera=camera or FakeCamera(),
    )


def test_one_mp4_per_round_and_env_holding_only_valid_frames(tmp_path):
    rec = recorder(tmp_path)
    rec.start_round(0)
    assert rec.camera.update_period == 0.0  # rasterization on while recording

    # env 0 runs 4 steps, env 1 stops after 2, env 2 never takes part
    masks = [[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]]
    for step, mask in enumerate(masks):
        rec.capture(
            torch.tensor(mask, dtype=torch.bool),
            rewards=torch.full((NUM_ENVS, 1), 0.5),
            terminated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            metrics={"env/force": torch.arange(NUM_ENVS, dtype=torch.float32)},
        )
    rec.end_round()
    assert rec.camera.update_period > 0.0  # and off again afterwards

    assert rec.frame_counts == {(0, 0): 4, (0, 1): 2}
    assert sorted(p.name for p in rec.paths.values()) == ["round0_env0.mp4", "round0_env1.mp4"]
    for path in rec.paths.values():
        assert path.is_file() and path.stat().st_size > 0

    import imageio

    reader = imageio.get_reader(str(rec.paths[(0, 0)]), format="FFMPEG")
    try:
        frames = [f for f in reader]
    finally:
        reader.close()
    assert len(frames) == 4  # the file really holds the four valid steps
    assert frames[0].shape[:2] == (HEIGHT, WIDTH)


def test_rounds_write_separate_files(tmp_path):
    rec = recorder(tmp_path)
    for index in (0, 1):
        rec.start_round(index)
        rec.capture(torch.ones(NUM_ENVS, dtype=torch.bool), rewards=torch.zeros(NUM_ENVS, 1))
        rec.end_round()
    assert sorted(key for key in rec.frame_counts) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert (tmp_path / "videos" / "round1_env2.mp4").is_file()


def test_the_overlay_sees_each_envs_own_values(tmp_path, clean_registry):
    seen: list[StepData] = []

    class Spy(Overlay):
        name = "spy"

        def apply(self, frame, step):
            seen.append(step)
            return frame

    register_overlay(Spy)
    rec = recorder(tmp_path, overlays=("spy",))
    rec.start_round(2)
    rec.capture(
        torch.tensor([True, False, True]),
        rewards=torch.tensor([[1.0], [9.0], [3.0]]),
        terminated=torch.tensor([[False], [False], [True]]),
        truncated=torch.tensor([[False], [False], [True]]),
        metrics={"env/force": torch.tensor([10.0, 20.0, 30.0])},
    )
    rec.capture(torch.tensor([True, False, False]), rewards=torch.tensor([[2.0], [9.0], [9.0]]))
    rec.end_round()

    assert [s.env for s in seen] == [0, 2, 0]
    assert seen[0].reward == pytest.approx(1.0) and seen[0].metrics == {"env/force": 10.0}
    assert seen[1].outcome == "timeout"  # env 2's own flags
    assert seen[2].total_reward == pytest.approx(3.0)  # env 0's running return, not env 2's
    assert all(s.valid for s in seen)


def test_frames_from_the_camera_are_normalized(tmp_path):
    """RGBA and float frames both reach the writer as uint8 RGB."""
    from robonuke_rl_core.recording import read_camera_rgb

    rgba = read_camera_rgb(FakeCamera(alpha=True))
    assert rgba.shape == (NUM_ENVS, HEIGHT, WIDTH, 3) and rgba.dtype == torch.uint8
    floats = read_camera_rgb(FakeCamera(float_rgb=True))
    assert floats.dtype == torch.uint8

    bad = FakeCamera()
    bad.data.output["rgb"] = torch.zeros(HEIGHT, WIDTH, 3)
    with pytest.raises(RuntimeError) as err:
        read_camera_rgb(bad)
    assert "(N, H, W, C)" in str(err.value)


def test_the_recorder_polices_its_protocol_and_the_camera(tmp_path):
    rec = recorder(tmp_path)
    with pytest.raises(RuntimeError):
        rec.capture(torch.ones(NUM_ENVS, dtype=torch.bool))  # no round open
    with pytest.raises(RuntimeError):
        rec.end_round()
    rec.start_round(0)
    with pytest.raises(RuntimeError):
        rec.start_round(1)  # still open
    rec.close()

    # a camera that returns the wrong env count is a misconfigured spawn, not a warning
    wrong = recorder(tmp_path, camera=FakeCamera(num_envs=NUM_ENVS + 1))
    wrong.start_round(0)
    with pytest.raises(RuntimeError) as err:
        wrong.capture(torch.ones(NUM_ENVS, dtype=torch.bool), rewards=torch.zeros(NUM_ENVS, 1))
    assert "per env" in str(err.value)
    wrong.close()


def test_a_missing_camera_says_how_to_install_it():
    from robonuke_rl_core.recording import recorder_camera

    with pytest.raises(RuntimeError) as err:
        recorder_camera(FakeEnv())
    assert "install_recorder_camera" in str(err.value)

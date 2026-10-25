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


def test_full_length_videos_hold_the_last_frame_with_the_cause(tmp_path):
    """pad_to: every video is the episode limit long; after an env's episode ends its last
    frame is held, darkened, with the cause written across it."""
    import imageio

    rec = EvalRecorder(
        FakeEnv(), output_dir=tmp_path / "videos", fps=10, overlays=(), camera=FakeCamera(),
        pad_to=6, causes=lambda: ["success", None, "out of bounds"],
    )
    rec.start_round(0)
    no = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
    # env 0 succeeds at step 1, env 2 terminates at step 0, env 1 runs to step 3 and times out
    rec.capture(torch.tensor([1, 1, 1], dtype=torch.bool), terminated=torch.tensor([[0], [0], [1]], dtype=torch.bool), truncated=no)
    rec.capture(torch.tensor([1, 1, 0], dtype=torch.bool), terminated=torch.tensor([[1], [0], [0]], dtype=torch.bool), truncated=no)
    rec.capture(torch.tensor([0, 1, 0], dtype=torch.bool), terminated=no, truncated=no)
    rec.capture(torch.tensor([0, 1, 0], dtype=torch.bool), terminated=torch.tensor([[0], [1], [0]], dtype=torch.bool),
                truncated=torch.tensor([[0], [1], [0]], dtype=torch.bool))
    rec.end_round()
    assert rec.frame_counts == {(0, 0): 6, (0, 1): 6, (0, 2): 6}
    assert rec._ended == {0: "success", 1: "timeout", 2: "out of bounds"}  # env 1: no term label -> the outcome
    reader = imageio.get_reader(str(rec.paths[(0, 0)]), format="FFMPEG")
    try:
        frames = [f for f in reader]
    finally:
        reader.close()
    assert len(frames) == 6
    # held frames differ from the live one (darkened + text) and are all the same
    assert np.abs(frames[2].astype(int) - frames[5].astype(int)).max() <= 3
    assert np.abs(frames[0].astype(int) - frames[2].astype(int)).max() > 10


def test_the_end_card_labels_and_colours(tmp_path):
    from robonuke_rl_core.recording import end_card

    base = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    card = end_card(base, "success")
    assert card.shape == base.shape and card.dtype == np.uint8
    middle = card[HEIGHT // 3: 2 * HEIGHT // 3]
    assert (middle[..., 1] > 180).any() and (middle[..., 0] < 100).any()  # green text on black
    assert card[0, 0, 0] == 120  # the rest darkened to 60 %


def test_termination_causes_reads_the_fired_term_and_prefers_it_over_the_timeout():
    from robonuke_rl_core.recording import termination_causes

    class Manager:
        active_terms = ["time_out", "success", "part_out_of_limits"]
        terms = {
            "time_out": torch.tensor([True, True, False, False]),
            "success": torch.tensor([False, True, False, False]),
            "part_out_of_limits": torch.tensor([False, False, True, False]),
        }

        def get_term(self, name):
            return self.terms[name]

    env = type("E", (), {"num_envs": 4, "termination_manager": Manager(),
                         "termination_labels": {"part_out_of_limits": "out of bounds", "time_out": "timeout"}})()
    assert termination_causes(env)() == ["timeout", "success", "out of bounds", None]
    assert termination_causes(FakeEnv()) is None  # a Direct env: the recorder labels by outcome


def test_video_fps_defaults_to_the_control_rate():
    from robonuke_rl_core.recording import video_fps

    env = type("E", (), {"step_dt": 0.05})()
    assert video_fps(None, env) == 20 and video_fps(30, env) == 30


def test_video_fps_counts_the_frames_per_control_step():
    from robonuke_rl_core.recording import video_fps

    env = type("E", (), {"step_dt": 0.1})()
    assert video_fps(None, env, 3) == 30 and video_fps(12, env, 3) == 12


def test_render_every_sets_the_render_interval_and_must_divide_the_decimation():
    from types import SimpleNamespace

    from robonuke_rl_core.recording import render_substeps

    def cfg(interval=12):
        return SimpleNamespace(decimation=12, sim=SimpleNamespace(render_interval=interval))

    untouched = cfg()
    assert render_substeps(untouched, None) == 1 and untouched.sim.render_interval == 12
    env_cfg = cfg(12)  # the run's resolved value: render_every wins
    assert render_substeps(env_cfg, 4) == 3 and env_cfg.sim.render_interval == 4
    for bad in (5, 24):
        with pytest.raises(ValueError, match="divide"):
            render_substeps(cfg(), bad)


class SubstepEnv:
    """Isaac Lab's step loop in miniature: decimation physics steps, a render every
    render_interval, a scene update after each; the camera shows the physics step count."""

    def __init__(self, camera, decimation=12, render_interval=4):
        self.num_envs = NUM_ENVS
        self.camera, self.decimation, self.render_interval = camera, decimation, render_interval
        self.count = 0
        env = self

        class Sim:
            def step(self, render=False):
                env.count += 1

            def render(self):
                rgb = env.camera.data.output["rgb"]
                rgb[:] = env.count

        class Scene:
            def update(self, dt=0.0):
                pass

        self.sim, self.scene = Sim(), Scene()

    def step(self):
        for _ in range(self.decimation):
            self.sim.step(render=False)
            if self.count % self.render_interval == 0:
                self.sim.render()
            self.scene.update()

    def reset(self):
        self.sim.render()  # a reset render, after the step's last update


def test_substep_frames_record_every_render_inside_the_step(tmp_path):
    camera = FakeCamera()
    env = SubstepEnv(camera)
    rec = EvalRecorder(env, output_dir=tmp_path / "videos", fps=30, overlays=(), camera=camera, substeps=3, pad_to=9)
    env.step()  # before the round: not recorded
    rec.start_round(0)
    yes = torch.ones(NUM_ENVS, dtype=torch.bool)
    seen = []
    original = rec._writer

    def spy(index):
        writer = original(index)
        if index == 0 and not hasattr(writer, "_spied"):
            append = writer.append
            writer.append = lambda frame: (seen.append(int(frame[0, 0, 0])), append(frame))[1]
            writer._spied = True
        return writer

    rec._writer = spy
    for _ in range(2):
        env.step()
        env.reset()  # the reset render must not add a frame
        rec.capture(yes)
    assert seen == [16, 20, 24, 28, 32, 36]  # physics steps 4, 8, 12 of each control step
    rec.end_round()
    assert rec.frame_counts[(0, 0)] == 9  # padded to the episode limit in frames
    env.step()
    with pytest.raises(RuntimeError, match="start_round"):
        rec.capture(yes)


def test_substep_frames_fail_loud_when_the_step_renders_a_different_count(tmp_path):
    camera = FakeCamera()
    env = SubstepEnv(camera, render_interval=6)  # 2 renders per step, the recorder wants 3
    rec = EvalRecorder(env, output_dir=tmp_path / "videos", fps=30, overlays=(), camera=camera, substeps=3)
    rec.start_round(0)
    env.step()
    with pytest.raises(RuntimeError, match="rendered 2 frames, expected 3"):
        rec.capture(torch.ones(NUM_ENVS, dtype=torch.bool))


def test_the_overlay_sees_each_envs_own_eval_state(tmp_path, clean_registry):
    seen: list[StepData] = []

    class Spy(Overlay):
        name = "spy"

        def apply(self, frame, step):
            seen.append(step)
            return frame

    register_overlay(Spy)
    rec = recorder(tmp_path, overlays=("spy",))
    rec.start_round(0)
    rec.capture(
        torch.tensor([True, False, True]),
        rewards=torch.zeros(NUM_ENVS, 1),
        eval_state={"phase": torch.tensor([1.0, 2.0, 3.0]), "wrench": torch.arange(18.0).view(NUM_ENVS, 6)},
    )
    rec.capture(torch.tensor([True, False, False]), rewards=torch.zeros(NUM_ENVS, 1))
    rec.end_round()

    assert [s.env for s in seen] == [0, 2, 0]
    assert seen[0].state["phase"] == pytest.approx(1.0) and isinstance(seen[0].state["phase"], float)
    np.testing.assert_array_equal(seen[1].state["wrench"], np.arange(12.0, 18.0))  # env 2's row, flat
    assert seen[2].state == {}  # a step without the channel


def test_the_overlays_see_the_reward_and_return_as_the_env_delivers_them(tmp_path, clean_registry):
    seen: list[StepData] = []

    class Spy(Overlay):
        name = "spy"

        def apply(self, frame, step):
            seen.append(step)
            return frame

    register_overlay(Spy)
    rec = EvalRecorder(FakeEnv(), output_dir=tmp_path / "videos", fps=10, overlays=("spy",), camera=FakeCamera())
    rec.start_round(0)
    for _ in range(2):
        rec.capture(torch.tensor([True, False, False]), rewards=torch.full((NUM_ENVS, 1), 0.1))
    rec.end_round()
    assert [s.reward for s in seen] == pytest.approx([0.1, 0.1])
    assert seen[-1].total_reward == pytest.approx(0.2)


def test_a_whole_episode_overlay_sees_every_step_before_drawing_the_raw_frames(tmp_path, clean_registry):
    events: list = []

    class Timeline(Overlay):
        name = "timeline"
        whole_episode = True

        def start_episode(self, steps):
            events.append(("start", [s.step for s in steps], [s.state.get("phase") for s in steps]))

        def apply(self, frame, step):
            events.append(("frame", step.env, step.step, int(frame[0, 0, 0])))
            out = frame.copy()
            out[:4, :4] = 255
            return out

    register_overlay(Timeline)
    rec = EvalRecorder(FakeEnv(), output_dir=tmp_path / "videos", fps=10, overlays=("timeline",),
                       camera=FakeCamera(), pad_to=4)
    rec.start_round(0)
    for step, phase in enumerate((0.0, 1.0, 2.0)):
        rec.capture(torch.tensor([True, step < 2, False]), rewards=torch.zeros(NUM_ENVS, 1),
                    eval_state={"phase": torch.full((NUM_ENVS,), phase)})
    assert events == []  # nothing is drawn while the round runs
    rec.end_round()

    # env 0: its three steps first, then its frames -- the camera's own pixels (env 0's value 0),
    # bit for bit through the scratch file; then env 1's two
    assert events == [
        ("start", [0, 1, 2], [0.0, 1.0, 2.0]),
        ("frame", 0, 0, 0), ("frame", 0, 1, 0), ("frame", 0, 2, 0),
        ("start", [0, 1], [0.0, 1.0]),
        ("frame", 1, 0, 1), ("frame", 1, 1, 1),
    ]
    assert rec.frame_counts == {(0, 0): 4, (0, 1): 4}  # padded to the episode limit as before
    assert sorted(p.name for p in (tmp_path / "videos").iterdir()) == ["round0_env0.mp4", "round0_env1.mp4"]

    import imageio

    reader = imageio.get_reader(str(rec.paths[(0, 1)]), format="FFMPEG")
    try:
        frames = [f for f in reader]
    finally:
        reader.close()
    assert len(frames) == 4 and frames[0][:4, :4].min() > 200  # the overlay's mark made it in

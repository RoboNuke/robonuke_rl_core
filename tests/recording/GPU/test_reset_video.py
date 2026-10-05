"""The headless reset viewer: `debug.py --resets --headless` writes one mp4 of N resets.

Run on the GPU machine: `pytest -m gpu`. It drives `render_resets` against the session env
(one env per process, so it is shared) rather than launching debug.py itself; the keyboard
modes cannot be tested automatically and have a manual checklist in CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robonuke_rl_core.recording import render_resets

pytestmark = pytest.mark.gpu

RESETS, HOLD_SECONDS = 2, 0.1  # 0.1s at the config's fps


def test_the_reset_video_holds_each_sampled_initial_condition(gpu_cfg, gpu_env, tmp_path):
    out = tmp_path / "resets.mp4"
    writer = render_resets(
        gpu_env,
        out,
        num_resets=RESETS,
        hold_seconds=HOLD_SECONDS,
        fps=gpu_cfg.eval.video_fps,
        overlays=gpu_cfg.eval.overlays,
    )

    expected = RESETS * max(1, round(HOLD_SECONDS * gpu_cfg.eval.video_fps))
    assert writer.frames == expected  # the hold lasts the asked-for playing time
    assert out.is_file() and out.stat().st_size > 0

    import imageio

    reader = imageio.get_reader(str(out), format="FFMPEG")
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    assert abs(len(frames) - expected) <= 1  # encoder slack
    assert frames[0].shape[:2] == (gpu_cfg.eval.video_height, gpu_cfg.eval.video_width)
    assert frames[0].std() > 0  # the camera rendered the scene, not a blank buffer

    # held still: the ticks within one reset are the same pose, and the sim never stepped
    assert int(gpu_env.unwrapped.episode_length_buf.max()) == 0


@pytest.mark.parametrize("bad", [{"num_resets": 0}, {"hold_seconds": 0.0}])
def test_a_degenerate_request_raises(gpu_env, tmp_path, bad):
    kwargs = {"num_resets": 1, "hold_seconds": 0.1, **bad}
    with pytest.raises(ValueError):
        render_resets(gpu_env, tmp_path / "x.mp4", **kwargs)

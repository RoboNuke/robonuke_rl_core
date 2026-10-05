"""The eval runner on a real Isaac Lab task: rounds, counts, and the trace it writes.

Run on the GPU machine: `pytest -m gpu`. The config and the env come from the session
fixtures in `tests/conftest.py` (one env per process), so this test drives `run_eval` against
that env rather than building its own — the config layering that `scripts/eval.py` does is
covered on CPU in `tests/config/test_config.py`.

What it proves: a round really resets, every round contributes exactly one episode per active
env, a partial final round uses only the envs it needs, and the parquet trace holds exactly
the valid steps the accountant counted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from robonuke_rl_core.evaluation import build_eval_policy, find_checkpoint, run_eval

pytestmark = pytest.mark.gpu

# keep the GPU budget small: 2 rounds (one of them partial) of one episode length
ROLLOUTS_OVER_ENVS = 1  # num_rollouts = num_envs + 1 -> rounds [num_envs, 1]


@pytest.fixture(scope="module")
def eval_policy(gpu_cfg, gpu_env, tmp_path_factory):
    """A 1-agent policy loaded from a checkpoint, the way `scripts/eval.py` builds it."""
    from robonuke_rl_core.learners.sac import SAC
    from robonuke_rl_core.models.factory import build_models

    device = gpu_env.device
    directory = tmp_path_factory.mktemp("eval_ckpt")
    # a 1-agent learner writes the checkpoint eval will load (untrained: this is wiring,
    # not learning)
    models = build_models(
        "sac",
        gpu_cfg.model,
        gpu_env.observation_space,
        gpu_env.state_space,
        gpu_env.action_space,
        1,
        device,
    )
    writer = SAC(
        models=models,
        memory=None,
        observation_space=gpu_env.observation_space,
        state_space=gpu_env.state_space,
        action_space=gpu_env.action_space,
        device=device,
        cfg=gpu_cfg.sac,
        trainer_cfg=gpu_cfg.trainer,
        num_agents=1,
        num_envs=gpu_env.num_envs,
        model_cfg=gpu_cfg.model,
        run_dirs=[directory],
    )
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    writer.save_checkpoints(step=7)

    checkpoint = find_checkpoint(directory, "7")
    policy, learner, metadata = build_eval_policy(gpu_cfg, gpu_env, checkpoint)
    assert metadata["step"] == 7 and metadata["agent_idx"] == 0
    return policy, learner


def test_eval_collects_exactly_the_requested_episodes(gpu_env, eval_policy):
    policy, learner = eval_policy
    num_envs = gpu_env.num_envs
    num_rollouts = num_envs + ROLLOUTS_OVER_ENVS  # forces a partial final round
    max_episode_length = int(gpu_env.unwrapped.max_episode_length)

    before = {name: tensor.clone() for name, tensor in learner.policy.state_dict().items()}
    normalizer = learner.observation_normalizer
    stats_before = normalizer.running_mean.clone() if normalizer is not None else None

    result = run_eval(
        env=gpu_env,
        policy=policy,
        num_rollouts=num_rollouts,
        max_episode_length=max_episode_length,
        seed=gpu_cfg_seed(gpu_env),
        save_state=True,
        device=gpu_env.device,
    )

    summary = result.summary
    assert summary["episodes"] == num_rollouts
    assert summary["rounds"] == 2
    assert summary["terminal"] + summary["timeout"] == num_rollouts
    assert 1 <= summary["length/mean"] <= max_episode_length
    assert [e.env for e in result.accounting.episodes if e.round == 1] == [0]

    # every episode is one env's first of its round, within the step budget
    for episode in result.accounting.episodes:
        assert 1 <= episode.length <= max_episode_length
        assert episode.terminal or episode.timeout
        # Forge raises terminated AND truncated at the budget: that is a timeout, not terminal
        if episode.length == max_episode_length and episode.truncated:
            assert episode.timeout

    # nothing trained: eval leaves the weights and the normalizer statistics alone
    for name, tensor in learner.policy.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    if stats_before is not None:
        assert torch.equal(normalizer.running_mean, stats_before)


def test_the_trace_holds_exactly_the_counted_steps(gpu_env, eval_policy, tmp_path):
    policy, _ = eval_policy
    max_episode_length = int(gpu_env.unwrapped.max_episode_length)
    result = run_eval(
        env=gpu_env,
        policy=policy,
        num_rollouts=gpu_env.num_envs,
        max_episode_length=max_episode_length,
        seed=gpu_cfg_seed(gpu_env),
        save_state=True,
        device=gpu_env.device,
    )
    state = result.state
    assert state is not None

    import pandas as pd

    path = state.write_parquet(tmp_path / "trace.parquet")
    assert Path(path).is_file()
    frame = pd.read_parquet(path)

    assert len(frame) == state.rows
    assert sum(e.length for e in result.accounting.episodes) == len(frame)
    for episode in result.accounting.episodes:
        rows = frame[(frame["round"] == episode.round) & (frame.env == episode.env)]
        assert len(rows) == episode.length
        assert rows.step.tolist() == list(range(episode.length))
    # the full state is there: observations and actions expanded per element
    assert any(name.startswith("observations_") for name in frame.columns)
    assert any(name.startswith("actions_") for name in frame.columns)
    assert {"reward", "rewards", "terminated"} & set(frame.columns)


def test_a_round_really_resets_the_envs(gpu_env, eval_policy):
    """Without clearing skrl's _reset_once guard, round 2 would start mid-rollout."""
    from robonuke_rl_core.evaluation import force_env_reset

    force_env_reset(gpu_env)
    for _ in range(3):
        gpu_env.step(torch.zeros(gpu_env.num_envs, *gpu_env.action_space.shape, device=gpu_env.device))
    assert int(gpu_env.unwrapped.episode_length_buf.max()) > 0

    force_env_reset(gpu_env)  # raises itself if the reset was swallowed
    assert int(gpu_env.unwrapped.episode_length_buf.max()) == 0


def gpu_cfg_seed(env) -> int:
    """A fixed seed for the rounds; the env's own seed is already set by the fixture."""
    return 11


def test_recording_writes_one_playable_mp4_per_env(gpu_cfg, gpu_env, eval_policy, tmp_path):
    """The record path end to end: real camera, real frames, one file per (round, env)."""
    import imageio

    from robonuke_rl_core.recording import EvalRecorder

    policy, _ = eval_policy
    recorder = EvalRecorder(
        gpu_env,
        output_dir=tmp_path / "videos",
        fps=gpu_cfg.eval.video_fps,
        overlays=gpu_cfg.eval.overlays,
    )
    result = run_eval(
        env=gpu_env,
        policy=policy,
        num_rollouts=gpu_env.num_envs,  # one round, every env
        max_episode_length=int(gpu_env.unwrapped.max_episode_length),
        seed=gpu_cfg_seed(gpu_env),
        save_state=False,
        recorder=recorder,
        device=gpu_env.device,
    )

    assert len(recorder.paths) == gpu_env.num_envs
    for episode in result.accounting.episodes:
        key = (episode.round, episode.env)
        # a video holds exactly the frames behind that env's episode
        assert recorder.frame_counts[key] == episode.length
        path = recorder.paths[key]
        assert path.is_file() and path.stat().st_size > 0

        reader = imageio.get_reader(str(path), format="FFMPEG")
        try:
            frames = [frame for frame in reader]
        finally:
            reader.close()
        assert len(frames) == episode.length
        assert frames[0].shape[:2] == (gpu_cfg.eval.video_height, gpu_cfg.eval.video_width)
        # the camera really rendered the scene (not a blank buffer) and the hud drew on it
        assert frames[0].std() > 0
        assert frames[0][: 2].max() == 0 or frames[0][:18].mean() < frames[0][18:].mean() * 2

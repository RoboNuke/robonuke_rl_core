"""Eval accounting: one episode per env per round, the first one, and nothing else.

Pure torch — no Isaac Lab, no env. The rule under test is the invariant the whole eval
pipeline rests on: an env's steps after its first done are not data.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.evaluation import (
    EvalAccounting,
    EvalCfg,
    EvalStateWriter,
    eval_rounds,
    per_env_channel,
)

NUM_ENVS = 4
LENGTH = 5  # max_episode_length


def flags(*envs_done: int) -> torch.Tensor:
    """A ``(num_envs, 1)`` done flag with the named envs set."""
    out = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
    for env in envs_done:
        out[env] = True
    return out


def rewards(value: float = 1.0) -> torch.Tensor:
    return torch.full((NUM_ENVS, 1), value)


def run_round(accounting, dones, *, active=NUM_ENVS, reward=1.0, metrics=None):
    """One round of ``LENGTH`` steps; ``dones[step]`` names the envs that terminate."""
    accounting.start_round(active)
    for step in range(LENGTH):
        accounting.step(
            rewards=rewards(reward),
            terminated=dones[step],
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            metrics=None if metrics is None else metrics(step),
        )
    accounting.end_round()


# ------------------------------------------------------------------ 1. first episode only
def test_an_env_that_terminates_early_counts_once():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    # env 0 ends at step 1 and would start a second episode that ends at step 3
    dones = [flags(), flags(0), flags(), flags(0), flags()]
    run_round(accounting, dones)

    assert len(accounting.episodes) == NUM_ENVS  # one per env, never two
    first = next(e for e in accounting.episodes if e.env == 0)
    assert first.length == 2 and first.ret == pytest.approx(2.0)  # steps 0 and 1 only
    assert first.terminal and not first.timeout
    # the other envs ran the whole budget
    for episode in accounting.episodes:
        if episode.env != 0:
            assert episode.length == LENGTH and episode.timeout


def test_the_second_episode_of_a_round_contributes_nothing():
    """The masked tail must not reach returns, metrics or the outcome flag."""
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    accounting.start_round(1)
    accounting.step(
        rewards=rewards(1.0),
        terminated=flags(0),
        truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        metrics={"env/force": torch.full((NUM_ENVS,), 2.0)},
        episode_metrics={"success": torch.ones(NUM_ENVS)},
    )
    for _ in range(LENGTH - 1):  # the env keeps running after its auto-reset
        accounting.step(
            rewards=rewards(100.0),
            terminated=flags(0),
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            metrics={"env/force": torch.full((NUM_ENVS,), 50.0)},
            episode_metrics={"success": torch.zeros(NUM_ENVS)},
        )
    accounting.end_round()

    (episode,) = accounting.episodes
    assert episode.length == 1
    assert episode.ret == pytest.approx(1.0)  # not 1 + 4*100
    assert episode.metrics["env/force"] == pytest.approx(2.0)  # not 50
    assert episode.episode_metrics["success"] == pytest.approx(1.0)  # the first done's value
    assert episode.terminated


# ------------------------------------------------------------------ 2. the timeout episode
def test_an_env_that_never_terminates_is_one_full_length_timeout():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    run_round(accounting, [flags()] * LENGTH, reward=0.5)

    assert len(accounting.episodes) == NUM_ENVS
    for episode in accounting.episodes:
        assert episode.length == LENGTH
        assert episode.ret == pytest.approx(LENGTH * 0.5)
        assert episode.timeout and not episode.terminated
    summary = accounting.summary()
    assert summary["episodes"] == NUM_ENVS and summary["timeout"] == NUM_ENVS
    assert summary["terminal"] == 0


def test_truncation_closes_the_episode_as_a_timeout():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    accounting.start_round(NUM_ENVS)
    for step in range(LENGTH):
        accounting.step(
            rewards=rewards(),
            terminated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            truncated=flags(*range(NUM_ENVS)) if step == LENGTH - 1 else flags(),
        )
    accounting.end_round()
    assert all(e.timeout and e.length == LENGTH for e in accounting.episodes)


def test_both_flags_at_the_time_limit_is_a_timeout():
    """Isaac Lab (Factory, Forge) raises terminated AND truncated at the step budget."""
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    accounting.start_round(NUM_ENVS)
    for step in range(LENGTH):
        last = step == LENGTH - 1
        accounting.step(
            rewards=rewards(),
            terminated=flags(*range(NUM_ENVS)) if last else flags(),
            truncated=flags(*range(NUM_ENVS)) if last else flags(),
        )
    accounting.end_round()

    assert all(e.timeout and not e.terminal for e in accounting.episodes)
    assert all(e.terminated and e.truncated for e in accounting.episodes)  # both, as observed
    summary = accounting.summary()
    assert summary["timeout"] == NUM_ENVS and summary["terminal"] == 0
    # a mid-episode terminal is still terminal
    accounting.start_round(NUM_ENVS)
    for step in range(LENGTH):
        accounting.step(
            rewards=rewards(),
            terminated=flags(0) if step == 1 else flags(),
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        )
    accounting.end_round()
    env_0 = next(e for e in accounting.episodes if e.round == 1 and e.env == 0)
    assert env_0.terminal and env_0.length == 2


def test_a_round_that_is_cut_short_raises():
    """An unfinished env is only a clean timeout if the round really ran its budget."""
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    accounting.start_round(NUM_ENVS)
    accounting.step(
        rewards=rewards(),
        terminated=flags(),
        truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
    )
    with pytest.raises(RuntimeError) as err:
        accounting.end_round()
    assert "max_episode_length" in str(err.value)


# ------------------------------------------------------------------ 3. the partial round
@pytest.mark.parametrize(
    "num_rollouts, num_envs, expected",
    [(10, 4, [4, 4, 2]), (8, 4, [4, 4]), (3, 8, [3]), (1, 1, [1]), (7, 1, [1] * 7)],
)
def test_round_sizes_sum_to_the_requested_rollouts(num_rollouts, num_envs, expected):
    assert eval_rounds(num_rollouts, num_envs) == expected
    assert sum(expected) == num_rollouts


def test_a_final_partial_round_uses_exactly_the_envs_it_needs():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    sizes = eval_rounds(6, NUM_ENVS)  # [4, 2]
    assert sizes == [4, 2]
    for size in sizes:
        run_round(accounting, [flags()] * LENGTH, active=size)

    assert len(accounting.episodes) == 6
    assert [e.env for e in accounting.episodes if e.round == 1] == [0, 1]
    assert accounting.summary()["rounds"] == 2
    with pytest.raises(ValueError):
        accounting.start_round(NUM_ENVS + 1)


def test_an_inactive_env_is_not_data():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    accounting.start_round(2)
    for _ in range(LENGTH):
        valid = accounting.step(
            rewards=rewards(),
            terminated=flags(),
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        )
        assert valid.tolist() == [True, True, False, False]
    accounting.end_round()
    assert sorted(e.env for e in accounting.episodes) == [0, 1]


# ------------------------------------------------------------------ 4. masked steps
def test_masked_steps_contribute_nothing_to_returns_or_metrics():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    # env 1 ends immediately; envs 0, 2, 3 run on with a huge reward and metric
    dones = [flags(1)] + [flags()] * (LENGTH - 1)
    accounting.start_round(NUM_ENVS)
    for step in range(LENGTH):
        accounting.step(
            rewards=rewards(3.0),
            terminated=dones[step],
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            metrics={"env/force": torch.full((NUM_ENVS,), 4.0)},
        )
    accounting.end_round()

    done_early = next(e for e in accounting.episodes if e.env == 1)
    assert done_early.length == 1 and done_early.ret == pytest.approx(3.0)
    assert done_early.metrics["env/force"] == pytest.approx(4.0)  # mean of one valid step
    for episode in accounting.episodes:
        if episode.env != 1:
            assert episode.ret == pytest.approx(LENGTH * 3.0)
    summary = accounting.summary()
    assert summary["env/force/mean"] == pytest.approx(4.0)  # the mask never skewed it


def test_the_summary_aggregates_episode_metrics_and_reports_partial_counts():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    # envs 0 and 1 terminate and publish success; 2 and 3 time out and publish nothing
    accounting.start_round(NUM_ENVS)
    for step in range(LENGTH):
        done = flags(0, 1) if step == 2 else flags()
        accounting.step(
            rewards=rewards(),
            terminated=done,
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
            episode_metrics={"success": torch.tensor([1.0, 0.0, 9.0, 9.0])},
        )
    accounting.end_round()

    summary = accounting.summary()
    assert summary["success/mean"] == pytest.approx(0.5)  # the success rate
    assert summary["success/episodes"] == 2  # only the two that ended with a done
    assert summary["terminal"] == 2 and summary["timeout"] == 2
    assert summary["return/mean"] == pytest.approx((3 + 3 + 5 + 5) / 4)
    table = accounting.episode_table()
    assert {row["outcome"] for row in table} == {"terminal", "timeout"}
    assert table[0]["episode_metric/success"] == pytest.approx(1.0)


def test_stepping_outside_a_round_raises():
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    with pytest.raises(RuntimeError):
        accounting.step(
            rewards=rewards(),
            terminated=flags(),
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        )


# ------------------------------------------------------------------ 5. the state writer
def test_the_state_writer_stores_exactly_the_unmasked_steps(tmp_path):
    accounting = EvalAccounting(NUM_ENVS, LENGTH)
    writer = EvalStateWriter(NUM_ENVS)
    dones = [flags(), flags(1), flags(), flags(), flags()]  # env 1 ends at step 1

    accounting.start_round(3)  # env 3 sits this round out
    writer.start_round(0)
    for step in range(LENGTH):
        observations = torch.arange(NUM_ENVS * 2, dtype=torch.float32).view(NUM_ENVS, 2) + step
        valid = accounting.step(
            rewards=rewards(),
            terminated=dones[step],
            truncated=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        )
        writer.capture(
            valid,
            observations=observations,
            actions=torch.full((NUM_ENVS, 1), float(step)),
            rewards=rewards(),
            terminated=dones[step],
            eval_state={"peg_z": torch.full((NUM_ENVS,), 0.1 * step)},
        )
    accounting.end_round()
    writer.end_round()

    table = writer.table()
    # one row per valid (round, env, step); env 3 took no part, env 1 stopped after step 1
    assert writer.rows == LENGTH + 2 + LENGTH
    assert sorted(set(table["env"].tolist())) == [0, 1, 2]
    assert set(table["round"].tolist()) == {0}

    rows = {env: (table["env"] == env) for env in (0, 1, 2)}
    assert table["step"][rows[1]].tolist() == [0, 1]  # masked tail never became rows
    assert table["actions"][rows[1]].tolist() == pytest.approx([0.0, 1.0])
    assert table["eval_state/peg_z"][rows[1]].tolist() == pytest.approx([0.0, 0.1])
    # a vector signal is expanded per element, not collapsed
    assert "observations_0" in table and "observations_1" in table
    assert table["observations_0"][rows[2]].tolist() == pytest.approx([4.0 + s for s in range(LENGTH)])
    assert table["terminated"][rows[1]].tolist() == [0, 1]  # the flag reads as a small int

    # the writer's row counts match the accountant's episode lengths exactly
    for episode in accounting.episodes:
        assert int(rows[episode.env].sum()) == episode.length

    # and it round trips through parquet
    import pandas as pd

    path = writer.write_parquet(tmp_path / "trace.parquet")
    frame = pd.read_parquet(path)
    assert len(frame) == writer.rows
    assert list(frame.columns)[:3] == ["round", "env", "step"]
    assert frame[frame.env == 1].actions.tolist() == pytest.approx([0.0, 1.0])


def test_rounds_are_kept_apart_and_backfilled_rectangularly():
    writer = EvalStateWriter(NUM_ENVS)
    writer.start_round(0)
    writer.capture(torch.ones(NUM_ENVS, dtype=torch.bool), rewards=torch.zeros(NUM_ENVS, 1))
    with pytest.raises(RuntimeError):
        writer.start_round(1)
    with pytest.raises(RuntimeError):
        writer.table()
    writer.end_round()

    # round 1 publishes a column round 0 never had
    writer.start_round(1)
    writer.capture(
        torch.ones(NUM_ENVS, dtype=torch.bool),
        rewards=torch.ones(NUM_ENVS, 1),
        eval_state={"late": torch.full((NUM_ENVS,), 5.0)},
    )
    writer.end_round()

    table = writer.table()
    assert writer.rows == 2 * NUM_ENVS
    round_0 = table["round"] == 0
    assert all(value != value for value in table["eval_state/late"][round_0])  # NaN back-fill
    assert table["eval_state/late"][~round_0].tolist() == pytest.approx([5.0] * NUM_ENVS)
    with pytest.raises(ValueError):
        writer.start_round(0)  # already written


def test_a_round_with_nothing_valid_adds_no_rows():
    writer = EvalStateWriter(NUM_ENVS)
    writer.start_round(0)
    writer.capture(torch.zeros(NUM_ENVS, dtype=torch.bool), rewards=torch.zeros(NUM_ENVS, 1))
    writer.end_round()
    assert writer.rows == 0
    assert writer.table()["round"].tolist() == []


def test_the_writer_polices_shapes_and_channel_consistency():
    writer = EvalStateWriter(NUM_ENVS)
    writer.start_round(0)
    valid = torch.ones(NUM_ENVS, dtype=torch.bool)
    with pytest.raises(ValueError) as err:
        writer.capture(valid, observations=torch.zeros(NUM_ENVS + 1, 2))
    assert "observations" in str(err.value)
    with pytest.raises(TypeError):
        writer.capture(valid, observations=1.0)

    writer.capture(valid, observations=torch.zeros(NUM_ENVS, 2))
    writer.capture(valid, observations=torch.zeros(NUM_ENVS, 2), extra=torch.zeros(NUM_ENVS))
    with pytest.raises(ValueError) as err:
        writer.end_round()
    assert "extra" in str(err.value)


# ------------------------------------------------------------------ the config section
def test_the_eval_section_validates_its_fields():
    EvalCfg().validate(None)  # the defaults are usable as they stand
    EvalCfg(num_rollouts=4).validate(None)
    with pytest.raises(ValueError) as err:
        EvalCfg(num_rollouts=0).validate(None)
    assert "eval.num_rollouts" in str(err.value)
    for field, value in (("video_fps", 0), ("video_height", -1), ("video_width", 0)):
        with pytest.raises(ValueError) as err:
            EvalCfg(num_rollouts=1, **{field: value}).validate(None)
        assert f"eval.{field}" in str(err.value)
    with pytest.raises(ValueError) as err:
        EvalCfg(num_rollouts=1, overlays=["hud", "hud"]).validate(None)
    assert "more than once" in str(err.value)


def test_the_eval_state_channel_is_shape_policed():
    infos = {"eval_state": {"peg_z": torch.zeros(NUM_ENVS), "pose": torch.zeros(NUM_ENVS, 7)}}
    channel = per_env_channel(infos, "eval_state", NUM_ENVS)
    assert sorted(channel) == ["peg_z", "pose"]  # trailing dims are allowed here
    assert per_env_channel({}, "eval_state", NUM_ENVS) is None
    assert per_env_channel(None, "eval_state", NUM_ENVS) is None

    with pytest.raises(TypeError) as err:
        per_env_channel({"eval_state": {"peg_z": torch.zeros(NUM_ENVS + 1)}}, "eval_state", NUM_ENVS)
    assert "peg_z" in str(err.value) and "eval_state" in str(err.value)
    with pytest.raises(TypeError):
        per_env_channel({"eval_state": {"peg_z": 1.0}}, "eval_state", NUM_ENVS)
    with pytest.raises(TypeError):
        per_env_channel({"eval_state": [1, 2]}, "eval_state", NUM_ENVS)

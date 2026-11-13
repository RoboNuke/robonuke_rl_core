"""sac.phase_logging (learners/phase_logging.py): per-phase attempts, step counters, cadence and names."""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.learners.phase_logging import PhaseLogger, flat_name
from robonuke_rl_core.logging import MetricAccumulator, WandbLogger

NAMES = ["corner", "grasp", "insert", "screw", "success"]


class Sink:
    """The logger stand-in: interval metrics through __call__, per-phase points through log_now."""

    def __init__(self):
        self.interval: list[tuple[int, dict, int]] = []
        self.points: list[tuple[int, dict, int]] = []

    def __call__(self, agent, metrics, step):
        self.interval.append((agent, {k: float(v.float().mean()) for k, v in metrics.items()}, step))

    def log_now(self, agent, values, step):
        self.points.append((agent, dict(values), step))


def make(write_interval=3, success_window=None, sink=None):
    sink = sink or Sink()
    return PhaseLogger(sink, names=NAMES, phases=NAMES[:4], routes=["insert", "screw"], num_agents=1, envs_per_agent=1,
                       write_interval=write_interval, success_window=success_window), sink


def step(log, now, nxt, t, done=False, success=0.0, reward=0.0, oob=0.0):
    infos = {"episode_metrics_to_log": {"episode/success": torch.tensor([success]),
                                        "episode/out_of_bounds": torch.tensor([oob])}}
    log.observe(torch.tensor([NAMES.index(now)]), torch.tensor([NAMES.index(nxt)]), torch.tensor([reward]),
                torch.tensor([done]), torch.tensor([False]), infos, t)


def test_flat_names_keep_one_slash():
    assert flat_name("episode/length/max") == "episode/length_max"
    assert flat_name("q/target_mean") == "q/target_mean"
    assert flat_name("plain") == "plain"


def test_a_routed_phase_publishes_its_attempts_every_write_interval_of_its_own_steps():
    log, sink = make(write_interval=3)
    t = 0
    for now, nxt in [("corner", "corner"), ("corner", "grasp"), ("grasp", "insert")]:
        step(log, now, nxt, t); t += 1
    assert sink.points == []  # neither corner nor grasp is routed; insert has no steps yet
    step(log, "insert", "insert", t, reward=0.5); t += 1
    step(log, "insert", "insert", t); t += 1
    step(log, "insert", "screw", t, reward=1.0); t += 1  # the insert attempt passes, on insert's 3rd step
    (agent, values, at), = sink.points
    assert at == t - 1 and values["insert_step"] == 3.0
    assert values["insert_episode/success"] == 1.0 and values["insert_episode/count"] == 1.0
    assert values["insert_episode/length"] == 3.0 and values["insert_episode/return"] == 1.5
    assert values["insert_episode/out_of_bounds"] == 0.0 and values["insert_episode/best_success_rate"] == 1.0
    assert not any(k.startswith("screw") for k in values)


def test_an_episode_ending_in_a_phase_is_its_success_only_with_the_tasks():
    log, sink = make(write_interval=2)
    step(log, "screw", "screw", 0)
    step(log, "screw", "corner", 1, done=True, success=1.0)  # assembled on screw's 2nd step
    step(log, "insert", "insert", 2)
    step(log, "insert", "corner", 3, done=True, success=0.0, oob=1.0)  # timed out / out of bounds in insert
    screw = next(v for _, v, _ in sink.points if "screw_step" in v)
    insert = next(v for _, v, _ in sink.points if "insert_step" in v)
    assert screw["screw_episode/success"] == 1.0
    assert insert["insert_episode/success"] == 0.0 and insert["insert_episode/out_of_bounds"] == 1.0


def test_the_success_window_and_the_best_rate_carry_across_publishes():
    log, sink = make(write_interval=1, success_window=2)
    step(log, "insert", "screw", 0)               # pass
    step(log, "insert", "corner", 1, done=True)   # fail
    step(log, "insert", "corner", 2, done=True)   # fail
    rates = [v["insert_episode/success"] for _, v, _ in sink.points]
    best = [v["insert_episode/best_success_rate"] for _, v, _ in sink.points]
    assert rates == [1.0, 0.5, 0.0] and best == [1.0, 1.0, 1.0]


def test_unrouted_phases_go_to_the_episode_family_at_the_interval():
    log, sink = make(write_interval=100)
    step(log, "corner", "corner", 0)
    step(log, "corner", "grasp", 1)
    step(log, "grasp", "insert", 2)
    log.on_global_flush(2)
    (agent, values, at), = sink.interval
    assert values["episode/corner_success"] == 1.0 and values["episode/corner_length"] == 2.0
    assert values["episode/corner_steps"] == 2.0 and values["episode/grasp_steps"] == 1.0
    assert values["episode/grasp_length"] == 1.0
    assert not any(k.startswith(("episode/insert", "episode/screw")) for k in values)


def test_the_hooks_rename_hold_and_drop():
    log, sink = make(write_interval=1)
    insert = log.hook_for("insert", first=True)
    screw = log.hook_for("screw", first=False)
    one = torch.tensor(1.0)
    insert(0, {"q/target_mean": one, "policy/std_insert": one, "policy/std_screw": one,
               "phase/reached": one, "phase/end": one, "phase/passed_insert": one, "phase/insert_step": one,
               "phase/stalled": one, "phase/corner_release_mm": one, "episode/return": one,
               "residual/position_mm_insert": one, "residual/position_mm_corner": one, "residual/position_mm": one}, 0)
    screw(0, {"loss/critic": one, "policy/std_insert": one, "episode/return": one, "route/steps": one}, 0)
    (_, now, _), (_, route, _) = sink.interval  # the env's metrics, from the first residual only
    assert set(now) == {"episode/phase_reached", "episode/stalled", "episode/corner_release_mm", "episode/return"}
    assert set(route) == {"route/screw_steps"}  # a residual's progress: every env interval, not held for its phase
    step(log, "insert", "insert", 0)
    step(log, "screw", "screw", 1)
    published = {k for _, v, _ in sink.points for k in v}
    assert {"insert_q/target_mean", "insert_policy/std", "insert_residual/position_mm", "screw_loss/critic"} <= published
    assert "screw_policy/std" not in published and not any("corner" in k for k in published)
    assert not any("route/" in k for k in published)


def test_the_accumulator_flushes_chosen_agents_and_joins_with_its_separator():
    acc = MetricAccumulator(2, distributions=["a/x"], sep="_")
    acc.add(0, {"a/x": torch.tensor([1.0, 3.0])})
    acc.add(1, {"a/x": torch.tensor([5.0])})
    out = acc.flush(agents=[0])
    assert out[0] == {"a/x": 2.0, "a/x_min": 1.0, "a/x_max": 3.0} and out[1] == {}
    assert acc.flush()[1]["a/x"] == 5.0  # agent 1 kept accumulating


def test_log_now_and_the_interval_flush_publish_one_row_per_step():
    from tests.logging.test_logging import FakeWandb

    backend = FakeWandb()
    logger = WandbLogger(run_names=["r_a0"], project="p", mode="disabled", backend=backend, flat_names=True)
    run = backend.runs[0]
    logger.log_now(0, {"insert_step": 3.0}, 7)
    logger(0, {"episode/length": torch.tensor([10.0, 20.0])}, 7)
    logger.flush(7)
    assert run.logged == [({"insert_step": 3.0, "episode/length": 15.0, "episode/length_min": 10.0, "episode/length_max": 20.0}, 7)]
    logger.log_now(0, {"screw_step": 1.0}, 8)
    logger.log_now(0, {"screw_step": 2.0}, 9)  # a later step commits the earlier row
    assert run.logged[-1] == ({"screw_step": 1.0}, 8)
    logger.close()
    assert run.logged[-1] == ({"screw_step": 2.0}, 9)


def test_routed_phases_must_be_tracked():
    with pytest.raises(ValueError, match="routed phases"):
        PhaseLogger(Sink(), names=NAMES, phases=["corner"], routes=["insert"], num_agents=1, envs_per_agent=1,
                    write_interval=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_the_accumulator_takes_values_from_another_device():
    acc = MetricAccumulator(1, device="cuda")
    acc.add(0, {"a/x": torch.tensor([1.0], device="cuda")})
    acc.add(0, {"b/y": torch.tensor(2.0)})  # a host-side value, as phase logging's counts are
    assert acc.flush()[0] == {"a/x": 1.0, "b/y": 2.0}


def test_a_pooled_phase_counts_every_envs_step_for_every_agent():
    """While a residual pools its rows (shared_warmup), <phase>_step grows by every env's step in the phase for every
    agent -- the steps the agent's memory received rows for; the other phases and the attempts stay the agent's own."""
    sink = Sink()
    log = PhaseLogger(sink, names=NAMES, phases=NAMES[:4], routes=["insert", "screw"], num_agents=2, envs_per_agent=1,
                      write_interval=100, success_window=None)
    insert, screw, corner = (NAMES.index(p) for p in ("insert", "screw", "corner"))

    def observe(now, nxt, pooled=()):
        infos = {"episode_metrics_to_log": {"episode/success": torch.zeros(2), "episode/out_of_bounds": torch.zeros(2)}}
        log.observe(torch.tensor(now), torch.tensor(nxt), torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                    torch.zeros(2, dtype=torch.bool), infos, 0, pooled=pooled)

    observe([insert, insert], [insert, insert], pooled=["insert"])  # both envs in insert, pooled: 2 for each agent
    assert log.steps[:, log.phases.index("insert")].tolist() == [2.0, 2.0]
    observe([insert, corner], [insert, corner], pooled=["insert"])  # one env in insert: still 1 for each agent
    assert log.steps[:, log.phases.index("insert")].tolist() == [3.0, 3.0]
    assert log.steps[:, log.phases.index("corner")].tolist() == [0.0, 1.0]  # corner is not pooled: agent 1's own
    observe([insert, screw], [insert, screw])  # pooling over: each agent its own envs
    assert log.steps[:, log.phases.index("insert")].tolist() == [4.0, 3.0]
    assert log.steps[:, log.phases.index("screw")].tolist() == [0.0, 1.0]
    with pytest.raises(ValueError, match="not a routed phase"):
        observe([insert, insert], [insert, insert], pooled=["corner"])

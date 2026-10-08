"""The logging layer: the accumulator's arithmetic and the wandb wiring, without wandb.

Every test here runs against a fake backend, so the package's logging behavior is pinned
whether or not wandb is installed.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.logging import MIN_WANDB_VERSION, MetricAccumulator, WandbLogger

RUN_NAMES = ["exp_a0", "exp_a1"]


# ------------------------------------------------------------------ the fake backend
class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logged: list[tuple[dict, int | None]] = []
        self.commits: list = []
        self.saved: list[tuple[str, str, str]] = []
        self.finished = False

    def log(self, data, step=None, commit=None):
        self.logged.append((dict(data), step))
        self.commits.append(commit)

    def save(self, path, base_path=None, policy=None):
        # plain run files only; an Artifacts call would be an AttributeError here
        self.saved.append((path, base_path, policy))

    def finish(self):
        self.finished = True


class FakeWandb:
    """A wandb stand-in: records every ``init`` and hands back independent run handles."""

    def __init__(self, version: str = ".".join(str(p) for p in MIN_WANDB_VERSION)):
        self.__version__ = version
        self.runs: list[FakeRun] = []

    def init(self, **kwargs):
        run = FakeRun(**kwargs)
        self.runs.append(run)
        return run


def logger(num_agents: int = 2, **overrides) -> tuple[WandbLogger, FakeWandb]:
    backend = overrides.pop("backend", None) or FakeWandb()
    kwargs = dict(
        run_names=RUN_NAMES[:num_agents],
        project="proj",
        entity="ent",
        group="exp",
        mode="disabled",
        backend=backend,
    )
    kwargs.update(overrides)
    return WandbLogger(**kwargs), backend


# ------------------------------------------------------------------ 1. per-step means
def test_a_per_step_mean_over_an_interval_is_exact():
    accumulator = MetricAccumulator(2)
    # agent 0 sees two envs per step for three steps: 1,2 / 3,4 / 5,6 -> mean 3.5
    for step in range(3):
        accumulator.add(0, {"env/force": torch.tensor([1.0, 2.0]) + 2.0 * step})
    accumulator.add(1, {"env/force": torch.tensor([10.0, 20.0])})

    flushed = accumulator.flush()
    assert flushed[0]["env/force"] == pytest.approx(3.5)
    assert flushed[1]["env/force"] == pytest.approx(15.0)
    # the interval is cleared
    assert accumulator.flush() == [{}, {}]


def test_a_zero_dim_learner_metric_is_its_own_mean():
    accumulator = MetricAccumulator(1)
    accumulator.add(0, {"loss/policy": torch.tensor(0.25)})
    accumulator.add(0, {"loss/policy": torch.tensor(0.75)})
    assert accumulator.flush()[0]["loss/policy"] == pytest.approx(0.5)


# ------------------------------------------------------------------ 2. episode semantics
def test_an_episode_mean_uses_only_the_values_it_was_given():
    accumulator = MetricAccumulator(1)
    accumulator.add(0, {"episode/return": torch.tensor([2.0, 4.0])})  # two episodes
    accumulator.add(0, {"episode/return": torch.tensor([6.0])})  # one more
    assert accumulator.flush()[0]["episode/return"] == pytest.approx(4.0)


def test_an_interval_with_no_finished_episode_produces_no_entry():
    accumulator = MetricAccumulator(2)
    accumulator.add(0, {"episode/return": torch.zeros(0)})  # masked to nothing
    accumulator.add(0, {"env/force": torch.ones(2)})
    flushed = accumulator.flush()
    assert "episode/return" not in flushed[0]  # a gap, not a zero
    assert flushed[0]["env/force"] == pytest.approx(1.0)
    assert flushed[1] == {}


def test_a_gap_does_not_leak_into_the_next_interval():
    accumulator = MetricAccumulator(1)
    accumulator.add(0, {"episode/return": torch.tensor([5.0])})
    assert accumulator.flush()[0]["episode/return"] == pytest.approx(5.0)
    accumulator.add(0, {"episode/return": torch.zeros(0)})
    assert accumulator.flush() == [{}]


# ------------------------------------------------------------------ 3. per-agent isolation
def test_one_agents_values_never_reach_another():
    accumulator = MetricAccumulator(3)
    accumulator.add(1, {"env/force": torch.tensor([100.0, 200.0])})
    flushed = accumulator.flush()
    assert flushed[0] == {} and flushed[2] == {}
    assert flushed[1]["env/force"] == pytest.approx(150.0)

    with pytest.raises(ValueError):
        accumulator.add(3, {"env/force": torch.ones(1)})


def test_each_run_only_sees_its_own_agents_metrics():
    log, backend = logger(num_agents=2)
    log(0, {"env/force": torch.tensor([1.0, 3.0])}, step=4)
    log(1, {"env/force": torch.tensor([10.0])}, step=4)
    log.flush(4)

    assert backend.runs[0].logged == [({"env/force": 2.0}, 4)]
    assert backend.runs[1].logged == [({"env/force": 10.0}, 4)]


# ------------------------------------------------------------------ 4. no sync before flush
def test_nothing_is_converted_to_python_before_flush(monkeypatch):
    accumulator = MetricAccumulator(2)

    def boom(*args, **kwargs):
        raise AssertionError("converted a tensor to Python before flush")

    for name in ("item", "tolist", "__float__", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, boom, raising=False)
    for _ in range(4):
        accumulator.add(0, {"env/force": torch.ones(2), "episode/return": torch.tensor([3.0])})
        accumulator.add(1, {"env/force": torch.zeros(2)})
    # the state is still device-side tensors
    assert torch.is_tensor(accumulator._sums[0]["env/force"])
    monkeypatch.undo()

    flushed = accumulator.flush()
    assert flushed[0]["env/force"] == pytest.approx(1.0)
    assert flushed[0]["episode/return"] == pytest.approx(3.0)


def test_a_non_tensor_metric_raises_naming_the_key():
    accumulator = MetricAccumulator(1)
    with pytest.raises(TypeError) as err:
        accumulator.add(0, {"env/force": 1.0})
    assert "env/force" in str(err.value)


# ------------------------------------------------------------------ 5. the wandb wiring
def test_one_run_per_agent_carries_the_name_group_and_config():
    config = {"experiment": {"num_agents": 2}}
    log, backend = logger(num_agents=2, config=config, tags=["a", "b"])
    assert [run.kwargs["name"] for run in backend.runs] == RUN_NAMES
    for run in backend.runs:
        assert run.kwargs["group"] == "exp"
        assert run.kwargs["project"] == "proj"
        assert run.kwargs["entity"] == "ent"
        assert run.kwargs["tags"] == ["a", "b"]
        assert run.kwargs["config"] is config
        # independent handles, never the global wandb.run
        assert run.kwargs["reinit"] == "create_new"

    log(0, {"loss/policy": torch.tensor(1.0)}, step=10)
    log.close()
    assert backend.runs[0].logged == [({"loss/policy": 1.0}, 10)]  # flushed at the last step
    assert all(run.finished for run in backend.runs)


def test_a_flush_with_nothing_pending_logs_nothing():
    log, backend = logger()
    log.flush(3)
    assert backend.runs[0].logged == []


def test_a_wandb_too_old_for_concurrent_runs_stops_the_run():
    with pytest.raises(RuntimeError) as err:
        logger(num_agents=2, backend=FakeWandb(version="0.18.7"))
    assert "create_new" in str(err.value) and "0.18.7" in str(err.value)
    # a usable version is accepted, including a suffixed one
    logger(num_agents=2, backend=FakeWandb(version="0.19.10rc1"))
    logger(num_agents=2, backend=FakeWandb(version="0.27.0"))


def test_the_installed_wandb_can_hold_one_run_per_agent():
    """Guards the version floor against the env this package actually runs in."""
    wandb = pytest.importorskip("wandb")
    from robonuke_rl_core.logging import _version_tuple

    assert _version_tuple(wandb.__version__) >= MIN_WANDB_VERSION, (
        f"installed wandb {wandb.__version__} is below the floor "
        f"{'.'.join(str(p) for p in MIN_WANDB_VERSION)}"
    )


# ------------------------------------------------------------------ 6. config validation
def test_wandb_mode_rejects_a_bad_value():
    from robonuke_rl_core.config import WandbCfg

    cfg = WandbCfg(entity="e", project="p", group="g", mode="sideways")
    with pytest.raises(ValueError) as err:
        cfg.validate(None)
    assert "wandb.mode" in str(err.value)
    WandbCfg(entity="e", project="p", group="g", mode="offline").validate(None)

    with pytest.raises(ValueError):
        logger(mode="sideways")


# ------------------------------------------------------------------ the learner end to end
def test_a_learner_feeds_both_channels_through_the_logger():
    from helpers import ACT_DIM, OBS_DIM, build_learner  # tests/learners is on the path

    learner = build_learner(
        "sac", num_agents=2, envs_per_agent=2, trainer_overrides={"write_interval": 2}
    )
    log, backend = logger(num_agents=2)
    learner.on_log.append(log)
    learner.on_flush.append(log.flush)

    num_envs = learner.num_envs
    observations = torch.zeros(num_envs, OBS_DIM)
    terminated = torch.zeros(num_envs, 1, dtype=torch.bool)
    terminated[0] = True  # agent 0's first env finishes on step 0

    learner.record_transition(
        observations=observations,
        states=None,
        actions=torch.zeros(num_envs, ACT_DIM),
        rewards=torch.ones(num_envs, 1),
        next_observations=observations,
        next_states=None,
        terminated=terminated,
        truncated=torch.zeros(num_envs, 1, dtype=torch.bool),
        infos={
            "metrics_to_log": {"env/force": torch.tensor([1.0, 3.0, 5.0, 7.0])},
            "episode_metrics_to_log": {"env/success": torch.tensor([1.0, 0.0, 0.0, 0.0])},
        },
        timestep=0,
        timesteps=100,
    )
    learner.post_interaction(timestep=1, timesteps=100)  # step 2 -> the write interval closes

    first = backend.runs[0].logged[0][0]
    second = backend.runs[1].logged[0][0]
    assert first["env/force"] == pytest.approx(2.0)  # agent 0's two envs
    assert second["env/force"] == pytest.approx(6.0)  # agent 1's two envs
    assert first["env/success"] == pytest.approx(1.0)  # only the env that finished
    assert first["episode/return"] == pytest.approx(1.0)
    assert first["episode/count"] == pytest.approx(1.0)
    # agent 1 finished nothing: a gap, not a zero
    assert "env/success" not in second
    assert "episode/return" not in second
    assert backend.runs[0].logged[0][1] == 1  # x-axis: the global env timestep


# ------------------------------------------------------------------ 7. plain run files
def test_the_resolved_config_is_uploaded_once_to_every_run(tmp_path):
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text("experiment:\n  seed: 3\n")
    log, backend = logger(num_agents=2, config_path=config_path)

    for run in backend.runs:
        assert [name for name, _, _ in run.saved] == [str(config_path)]
        # stored under its base name, which is what an eval downloads by
        assert run.saved[0][1] == str(tmp_path)
        assert run.saved[0][2] == "now"


def test_a_checkpoint_goes_only_to_its_own_agents_run(tmp_path):
    log, backend = logger(num_agents=2)
    first = tmp_path / "ckpt_10.ckpt"
    first.write_bytes(b"weights")

    log.checkpoint(1, 10, first)
    assert backend.runs[0].saved == []
    assert [name for name, _, _ in backend.runs[1].saved] == [str(first)]


def test_uploading_a_file_that_is_not_there_raises(tmp_path):
    log, _ = logger()
    with pytest.raises(FileNotFoundError) as err:
        log.save_file(tmp_path / "missing.ckpt")
    assert "missing.ckpt" in str(err.value)


def test_every_flush_is_committed_at_once():
    """wandb holds a row logged at an explicit step until a later step arrives; committing
    each flush makes a point visible as soon as it is written, not one interval later."""
    log, fake = logger(num_agents=1)
    log(0, {"loss": torch.tensor(1.0)}, step=10)
    log.flush(10)
    assert fake.runs[0].commits == [True]

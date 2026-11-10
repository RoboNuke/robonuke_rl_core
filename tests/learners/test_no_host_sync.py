"""The per-step rule (CLAUDE.md): the learner's rollout path makes exactly one host synchronization per step (its
combined read), whatever is logged. Checked on a CUDA device with torch's sync-debug mode, with no Isaac Sim."""

import warnings

import pytest
import torch

from helpers import ACT_DIM, OBS_DIM, build_learner

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for the sync-debug mode")


class _Count:
    """Counts torch's 'called a synchronizing CUDA operation' warnings inside the block."""

    def __enter__(self):
        torch.cuda.synchronize()
        self._ctx = warnings.catch_warnings(record=True)
        self.seen = self._ctx.__enter__()
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(1)
        return self

    def __exit__(self, *exc):
        torch.cuda.set_sync_debug_mode(0)
        self._ctx.__exit__(*exc)
        self.syncs = [str(w.message) for w in self.seen if "called a synchronizing" in str(w.message)]
        return False


def _step(learner, n, labels, done=None):
    obs = torch.randn(n, OBS_DIM, device="cuda")
    done = torch.zeros(n, 1, dtype=torch.bool, device="cuda") if done is None else done
    infos = {
        "eval_state": {"phase/id": labels},
        "metrics_to_log": {"residual/position_mm": torch.rand(n, device="cuda")},
        "episode_metrics_to_log": {"episode/success": torch.ones(n, device="cuda")},
    }
    learner.record_transition(
        observations=obs, states=None, actions=torch.zeros(n, ACT_DIM, device="cuda"), rewards=torch.ones(n, 1, device="cuda"),
        next_observations=obs, next_states=None, terminated=done, truncated=torch.zeros_like(done), infos=infos,
        timestep=1, timesteps=10,
    )
    return obs


def test_a_labelled_logging_step_synchronizes_once():
    learner = build_learner(
        "sac", num_agents=2, envs_per_agent=3, device="cuda", std_by_label="eval_state/phase/id",
        std_label_names=["corner", "grasp"], metrics_by_label=["residual/position_mm"], store_labels=["grasp"],
        trainer_overrides={"success_window": 4},
    )
    learner.on_log.append(lambda agent, metrics, step: None)
    n = learner.num_envs
    labels = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.float32, device="cuda")
    _step(learner, n, labels)  # the first steps: buffers, the label index and the memory streams created
    obs = _step(learner, n, labels)
    done = torch.tensor([[True], [False], [False], [True], [False], [False]], device="cuda")
    with _Count() as c:
        _step(learner, n, labels, done)
        learner.act(obs, None, timestep=2, timesteps=10)
    assert len(c.syncs) == 1, c.syncs  # the step's combined host read (_read_host), nothing else


def test_the_memory_writes_kept_rows_without_a_host_read():
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory

    memory = MultiRandomMemory(capacity=10, num_envs=4, num_agents=2, device="cuda")
    memory.create_tensor(name="x", size=1, dtype=torch.float32)
    keep1 = torch.tensor([True, False, True, True], device="cuda")  # built outside: a list to the device is a copy
    keep2 = torch.tensor([True, True, False, False], device="cuda")
    with _Count() as c:
        memory.add_rows(keep1, x=torch.arange(4.0, device="cuda").unsqueeze(-1))
        memory.add_rows(keep2, x=torch.arange(10.0, 14.0, device="cuda").unsqueeze(-1))
    assert c.syncs == []
    assert memory.fills() == [3, 2] and memory.size == 2
    x = memory.get_tensor_by_name("x")
    assert x[0, :3, 0].tolist() == [0.0, 10.0, 11.0] and x[1, :2, 0].tolist() == [2.0, 3.0]

"""``sac.cuda_graph``: a captured-and-replayed gradient step does what the same step does
eagerly. Needs CUDA only (no Isaac Sim)."""

from __future__ import annotations

import copy

import gymnasium
import numpy as np
import pytest
import torch

from robonuke_rl_core.learners.cfg import SACCfg, TrainerCfg
from robonuke_rl_core.learners.cuda_graph import StepGraphs
from robonuke_rl_core.learners.sac import SAC
from robonuke_rl_core.memory.multi_random import MultiRandomMemory
from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaCriticCfg, SimbaModelCfg
from robonuke_rl_core.models.factory import build_models

pytestmark = pytest.mark.gpu


def _learner(device) -> SAC:
    box = lambda d: gymnasium.spaces.Box(-np.inf, np.inf, (d,), np.float32)
    obs, state, act = box(6), box(9), box(2)
    cfg = SACCfg(batch_size=16, gradient_steps=2, policy_delay=2, num_critics=4, critic_subset=2,
                 actor_q_reduction="mean", learning_starts=0)
    model = SimbaModelCfg(actor=SimbaActorCfg(actor_n=1, actor_latent=16), critic=SimbaCriticCfg(critic_n=1, critic_latent=16))
    torch.manual_seed(0)
    models = build_models("sac", model, obs, state, act, 2, device, num_critics=4)
    memory = MultiRandomMemory(capacity=64, num_envs=2, num_agents=2, device=device)
    learner = SAC(models=models, memory=memory, observation_space=obs, state_space=state, action_space=act,
                  device=device, cfg=cfg, trainer_cfg=TrainerCfg(learner="sac", total_timesteps=100),
                  num_agents=2, num_envs=2, model_cfg=model)
    learner.init()
    learner.enable_training_mode(True)
    for name, tensor in memory.tensors.items():
        tensor.copy_(torch.zeros_like(tensor) if tensor.dtype == torch.bool else torch.randn_like(tensor))
    memory.size = 64
    memory.sync_size()
    return learner


def test_a_replayed_graph_step_equals_the_same_step_run_eagerly():
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    device = torch.device("cuda")
    eager = _learner(device)
    graphed = copy.deepcopy(eager)
    graphed._graphs = StepGraphs(warmup=1)
    mode = (False, True, True)  # critic + actor
    step = lambda learner: (lambda: learner._gradient_step(mode, 16, 10, graph=True))

    # warm-up (eager on a side stream), capture + first replay, two more replays: each against
    # the same step run eagerly from the same RNG state
    for _ in range(4):
        state = torch.cuda.get_rng_state()
        graphed._graphs.run(mode, step(graphed))
        torch.cuda.set_rng_state(state)
        step(eager)()
    torch.cuda.synchronize()
    assert graphed._graphs.captured == 1
    for key in eager._checkpoint_model_keys():
        for agent in range(2):
            a, b = getattr(eager, key).agent_state_dict(agent), getattr(graphed, key).agent_state_dict(agent)
            for name in a:
                assert torch.allclose(a[name], b[name], atol=1e-5), (key, agent, name)
    assert torch.allclose(eager._entropy_coefficient, graphed._entropy_coefficient)
    assert torch.allclose(eager.observation_normalizer.running_mean, graphed.observation_normalizer.running_mean)

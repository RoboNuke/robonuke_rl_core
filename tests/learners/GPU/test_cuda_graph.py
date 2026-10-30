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


def _learner(device, n_step: int = 1) -> SAC:
    box = lambda d: gymnasium.spaces.Box(-np.inf, np.inf, (d,), np.float32)
    obs, state, act = box(6), box(9), box(2)
    cfg = SACCfg(batch_size=16, gradient_steps=2, policy_delay=2, num_critics=4, critic_subset=2,
                 actor_q_reduction="mean", learning_starts=0, n_step=n_step)
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


@pytest.mark.parametrize("n_step", [1, 3])  # n_step > 1: the target reads the rows' own discounts
def test_a_replayed_graph_step_equals_the_same_step_run_eagerly(n_step):
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    device = torch.device("cuda")
    eager = _learner(device, n_step)
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


def test_a_graph_safe_aux_loss_is_captured_and_its_metric_stays_fresh():
    """With every aux term graph-safe the step is captured with the loss inside it: the weights
    match the eager step's, and the loss's raw values, returned with the step's other outputs,
    are each replay's own (not the capture's)."""
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    from robonuke_rl_core.losses import LOSSES, AuxLoss, LossesCfg, LossTermCfg, build_aux_losses, register_loss

    class _SquaredActions(AuxLoss):
        name = "_test_graph_squared_actions"
        supported_targets = ("policy", "critic")
        graph_safe = True

        def compute(self, ctx):
            values = ctx.actions if ctx.target == "policy" else ctx.critic_1_values
            return values.pow(2).view(ctx.learner.num_agents, -1).mean(dim=1)

    known = dict(LOSSES)
    register_loss(_SquaredActions)
    try:
        device = torch.device("cuda")
        terms = [LossTermCfg(name=_SquaredActions.name, target=t, weight=0.5, graph_safe=True) for t in ("policy", "critic")]
        eager = _learner(device)
        eager.aux_loss.append(build_aux_losses(LossesCfg(terms=terms), 2))
        assert eager.aux_loss_graph_safe()
        graphed = copy.deepcopy(eager)
        graphed._graphs = StepGraphs(warmup=1)
        mode = (False, True, True)
        step = lambda learner: (lambda: learner._gradient_step(mode, 16, 10, graph=True))

        seen = []
        for _ in range(4):
            state = torch.cuda.get_rng_state()
            out_g = graphed._graphs.run(mode, step(graphed))
            got = {k: v.clone() for k, v in {**out_g["critic"]["aux_metrics"], **out_g["actor"]["actor_aux_metrics"]}.items()}
            torch.cuda.set_rng_state(state)
            out_e = step(eager)()
            want = {**out_e["critic"]["aux_metrics"], **out_e["actor"]["actor_aux_metrics"]}
            assert sorted(got) == sorted(want) == sorted(f"loss/{_SquaredActions.name}_{t}" for t in ("critic", "policy"))
            for name in want:
                assert torch.allclose(got[name], want[name], atol=1e-5), name
            seen.append(got[f"loss/{_SquaredActions.name}_policy"])
        torch.cuda.synchronize()
        assert graphed._graphs.captured == 1
        # replays after the capture produced new values, not the captured ones repeated
        assert not torch.equal(seen[-1], seen[-2])
        for key in eager._checkpoint_model_keys():
            for agent in range(2):
                a, b = getattr(eager, key).agent_state_dict(agent), getattr(graphed, key).agent_state_dict(agent)
                for name in a:
                    assert torch.allclose(a[name], b[name], atol=1e-5), (key, agent, name)
    finally:
        LOSSES.clear()
        LOSSES.update(known)


def test_the_built_in_selection_loss_captures_and_replays_to_the_eager_value():
    """``supervised_selection`` declares ``graph_safe``: capture its compute and replay it."""
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    from types import SimpleNamespace

    from robonuke_rl_core.losses import LOSSES, LossContext

    device = torch.device("cuda")
    loss = LOSSES["supervised_selection"](num_axes=3)
    prob = torch.rand(8, 3, device=device).clamp(0.05, 0.95)
    flags = (torch.rand(8, 3, device=device) > 0.5).float()
    ctx = LossContext(learner=SimpleNamespace(num_agents=2), target="policy", sampled={"in_contact": flags}, policy_outputs={"selection_prob": prob})
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        loss.compute(ctx)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = loss.compute(ctx)
    for _ in range(3):
        prob.copy_(torch.rand_like(prob).clamp(0.05, 0.95))
        flags.copy_((torch.rand_like(flags) > 0.5).float())
        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(out, loss.compute(ctx), atol=1e-6)


def test_a_routed_step_with_critic_only_demos_replays_to_the_eager_step(tmp_path):
    """A routed residual's step (per-agent fills from add_block / add_rows, demos flagged for the critics only, the
    actor's own batch drawn from the online rows [n_demo, fill), the PARTS terms) is captured and replays to what
    the same step does eagerly."""
    if not torch.cuda.is_available():
        pytest.fail("this GPU test needs CUDA")
    device = torch.device("cuda")
    box = lambda d: gymnasium.spaces.Box(-np.inf, np.inf, (d,), np.float32)  # noqa: E731
    obs, state, act = box(6), box(9), box(2)
    rows = 12
    demo = {
        "observations": torch.randn(rows, 6), "next_observations": torch.randn(rows, 6),
        "states": torch.randn(rows, 9), "next_states": torch.randn(rows, 9),
        "actions": torch.rand(rows, 2) * 2 - 1, "rewards": torch.zeros(rows, 1),
        "terminated": torch.tensor([[False]] * (rows - 1) + [[True]]), "truncated": torch.zeros(rows, 1, dtype=torch.bool),
    }
    demo["rewards"][-1] = 1.0
    path = tmp_path / "demo.pt"
    torch.save({"transitions": demo}, path)
    cfg = SACCfg(batch_size=16, gradient_steps=2, policy_delay=2, num_critics=4, critic_subset=2, actor_q_reduction="mean",
                 learning_starts=0, demo_transitions=str(path), demo_storage="memory_critic",
                 success_clone_weight=10.0, failure_anchor_weight=10.0, parts_term_reduction="mean", actor_q_scale=2.5)
    model = SimbaModelCfg(actor=SimbaActorCfg(actor_n=1, actor_latent=16), critic=SimbaCriticCfg(critic_n=1, critic_latent=16))
    torch.manual_seed(0)
    models = build_models("sac", model, obs, state, act, 1, device, num_critics=4)
    memory = MultiRandomMemory(capacity=128, num_envs=1, num_agents=1, device=device)
    learner = SAC(models=models, memory=memory, observation_space=obs, state_space=state, action_space=act,
                  device=device, cfg=cfg, trainer_cfg=TrainerCfg(learner="sac", total_timesteps=100),
                  num_agents=1, num_envs=1, model_cfg=model)
    learner.init()
    learner.enable_training_mode(True)
    assert memory._agent_size == [rows] and learner._demo_rows == [rows]
    online = {name: torch.randn(20, memory.get_tensor_by_name(name).shape[-1], device=device) for name in learner._tensors_names}
    online["terminated"] = torch.zeros(20, 1, dtype=torch.bool, device=device)
    online["truncated"] = torch.zeros(20, 1, dtype=torch.bool, device=device)
    online["is_demo"] = torch.zeros(20, 1, device=device)
    online["episode_ids"] = torch.zeros(20, 1, device=device)  # episode 0: the first env episode
    learner._outcomes[0] = 1.0  # a success: the clone term has rows
    memory.add_block(**online)
    memory.sync_size()
    assert learner._online_rows_available() and not learner._memory_wrapped()

    eager, graphed = learner, copy.deepcopy(learner)
    graphed._graphs = StepGraphs(warmup=1)
    mode = (False, True, True)
    step = lambda learner: (lambda: learner._gradient_step(mode, 16, 10, graph=True))  # noqa: E731
    for _ in range(4):
        state_rng = torch.cuda.get_rng_state()
        graphed._graphs.run(mode, step(graphed))
        torch.cuda.set_rng_state(state_rng)
        step(eager)()
    torch.cuda.synchronize()
    assert graphed._graphs.captured == 1
    for key in eager._checkpoint_model_keys():
        a, b = getattr(eager, key).agent_state_dict(0), getattr(graphed, key).agent_state_dict(0)
        for name in a:
            assert torch.allclose(a[name], b[name], atol=1e-5), (key, name)
    # the actor's batch is drawn from the online rows only
    batch = eager._sample_online(16, graph=True)
    assert float(batch["is_demo"].sum()) == 0.0

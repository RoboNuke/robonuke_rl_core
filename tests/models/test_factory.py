"""The models `build_models` returns: block-only parameters, independence, slicing.

Every test runs on the models themselves, without a learner or an env.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from robonuke_rl_core.optim import BlockAdamW
from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaCriticCfg, SimbaModelCfg
from robonuke_rl_core.models.factory import MODEL_BUILDERS, build_models

NUM_AGENTS = 3
OBS_DIM = 4
STATE_DIM = 6
ACT_DIM = 2
ROWS = 5
LEARNERS = sorted({learner for _, learner in MODEL_BUILDERS})


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def tiny_cfg() -> SimbaModelCfg:
    return SimbaModelCfg(
        actor=SimbaActorCfg(actor_n=1, actor_latent=8),
        critic=SimbaCriticCfg(critic_n=1, critic_latent=8),
    )


def make(learner: str, num_agents: int = NUM_AGENTS, asymmetric: bool = False) -> dict:
    torch.manual_seed(0)
    return build_models(
        learner,
        tiny_cfg(),
        box(OBS_DIM),
        box(STATE_DIM) if asymmetric else None,
        box(ACT_DIM),
        num_agents,
        "cpu",
    )


def forward(model, observations: torch.Tensor, actions: torch.Tensor, training: bool = False):
    """One forward pass, whichever kind of model this is."""
    inputs = {"observations": observations, "training": training}
    if model.__class__.__name__.endswith("QCritic"):
        inputs["taken_actions"] = actions
        value, _ = model.act(inputs, role="critic")
        return value
    if model.__class__.__name__.endswith("ValueCritic"):
        value, _ = model.act(inputs, role="value")
        return value
    _, outputs = model.act({**inputs, "taken_actions": actions}, role="policy")
    return outputs["mean_actions"]  # deterministic: no sampling noise


# ------------------------------------------------------------------ 1. stacked parameters only
@pytest.mark.parametrize("learner", LEARNERS)
def test_every_parameter_is_stacked_across_agents(learner):
    for name, model in make(learner).items():
        for param_name, param in model.named_parameters():
            assert param.dim() >= 1 and param.shape[0] == NUM_AGENTS, (
                f"{learner}/{name}.{param_name} has shape {tuple(param.shape)}: every parameter "
                f"needs a leading agent dimension of {NUM_AGENTS}, or it couples the agents"
            )
            assert not param.is_meta, f"{learner}/{name}.{param_name} is a meta tensor"


# ------------------------------------------------------------------ 2. independence
@pytest.mark.parametrize("learner", LEARNERS)
def test_one_agents_rows_cannot_change_another_agents_output(learner):
    """Agent 1's rows only; the other agents' outputs must be untouched."""
    torch.manual_seed(1)
    observations = torch.randn(NUM_AGENTS * ROWS, OBS_DIM)
    actions = torch.rand(NUM_AGENTS * ROWS, ACT_DIM) * 2.0 - 1.0

    extreme_obs = observations.clone()
    extreme_obs[ROWS : 2 * ROWS] *= 1.0e3  # agent 1's rows only
    extreme_actions = actions.clone()
    extreme_actions[ROWS : 2 * ROWS] = 1.0

    for name, model in make(learner).items():
        model.enable_training_mode(False)
        torch.manual_seed(2)
        plain = forward(model, observations, actions, training=True)
        torch.manual_seed(2)
        changed = forward(model, extreme_obs, extreme_actions, training=True)
        for agent in (0, 2):
            rows = slice(agent * ROWS, (agent + 1) * ROWS)
            assert torch.equal(plain[rows], changed[rows]), f"{learner}/{name}, agent {agent}"
        rows = slice(ROWS, 2 * ROWS)
        assert not torch.equal(plain[rows], changed[rows]), f"{learner}/{name}, agent 1"


# ------------------------------------------------------------------ 3. checkpoint slicing
@pytest.mark.parametrize("learner", LEARNERS)
def test_slicing_one_agent_into_a_single_agent_model(learner):
    source = make(learner)
    target = make(learner, num_agents=1)
    torch.manual_seed(3)
    observations = torch.randn(ROWS, OBS_DIM)
    actions = torch.rand(ROWS, ACT_DIM) * 2.0 - 1.0
    padding = torch.zeros(2 * ROWS, OBS_DIM), torch.zeros(2 * ROWS, ACT_DIM)

    for name, model in source.items():
        target[name].load_agent_state_dict(0, model.agent_state_dict(2))
        model.enable_training_mode(False)
        target[name].enable_training_mode(False)
        with torch.no_grad():
            # agent 2's rows of a full batch, against the 1-agent copy
            full = forward(
                model,
                torch.cat([padding[0], observations]),
                torch.cat([padding[1], actions]),
            )
            single = forward(target[name], observations, actions)
        assert torch.allclose(full[2 * ROWS :], single, atol=1e-6), f"{learner}/{name}"


@pytest.mark.parametrize("learner", LEARNERS)
def test_optimizer_state_slices_per_agent(learner):
    """One agent's optimizer slice loads into one slot of another optimizer."""
    for name, model in make(learner).items():
        source = BlockAdamW(model.parameters(), NUM_AGENTS, lr=0.1)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        source.step()

        target_model = make(learner, num_agents=1)[name]
        target = BlockAdamW(target_model.parameters(), 1, lr=0.1)
        target.load_agent_state_dict(0, source.agent_state_dict(2))
        for index in range(len(source.params)):
            assert torch.equal(target.exp_avg[index][0], source.exp_avg[index][2]), name
            assert torch.equal(target.exp_avg_sq[index][0], source.exp_avg_sq[index][2]), name
        assert int(target.t[0]) == int(source.t[2])


# ------------------------------------------------------------------ 4. unknown learner
def test_build_models_raises_on_an_unknown_learner():
    with pytest.raises(ValueError) as err:
        make("dqn")
    message = str(err.value)
    assert "dqn" in message
    for learner in LEARNERS:
        assert learner in message


@pytest.mark.parametrize("learner", LEARNERS)
def test_asymmetric_critics_take_the_state_space(learner):
    models = make(learner, asymmetric=True)
    assert models["policy"].num_observations == OBS_DIM
    critics = [name for name in models if name != "policy"]
    assert critics
    for name in critics:
        assert models[name].num_observations == STATE_DIM, name


def test_the_builders_cover_the_registered_learners():
    from robonuke_rl_core.learners.cfg import LEARNERS as CONFIGURED
    from robonuke_rl_core.models.cfg import MODEL_ARCHITECTURES

    # every (architecture, learner) pair has a builder
    expected = {(arch, learner) for arch in MODEL_ARCHITECTURES for learner in CONFIGURED}
    assert set(MODEL_BUILDERS) == expected


def test_an_unknown_architecture_learner_pair_raises():
    import dataclasses

    cfg = tiny_cfg()
    bad = dataclasses.replace(cfg, architecture="resnet")
    with pytest.raises(ValueError) as err:
        build_models("ppo", bad, box(OBS_DIM), None, box(ACT_DIM), 2, "cpu")
    assert "resnet" in str(err.value) and "simba" in str(err.value)

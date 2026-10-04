"""The models `build_models` returns: block-only parameters, independence, slicing.

Every test runs on the models themselves, without a learner or an env.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn

from robonuke_rl_core.models.block_utils import (
    assign_block_slice,
    merge_optimizer_states,
    slice_block_state_dict,
    slice_optimizer_state,
)
from robonuke_rl_core.models.cfg import ActorCfg, CriticCfg, ModelCfg
from robonuke_rl_core.models.factory import MODEL_BUILDERS, build_models

NUM_AGENTS = 3
OBS_DIM = 4
STATE_DIM = 6
ACT_DIM = 2
ROWS = 5
LEARNERS = sorted(MODEL_BUILDERS)


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def tiny_cfg() -> ModelCfg:
    return ModelCfg(
        actor=ActorCfg(actor_n=1, actor_latent=8),
        critic=CriticCfg(critic_n=1, critic_latent=8, n_atoms=5, v_min=-2.0, v_max=2.0),
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
    if hasattr(model, "forward_dist"):  # FlashSAC critic
        value, _ = model.forward_dist(observations, actions, training=training)
        return value
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


# ------------------------------------------------------------------ 1. block parameters only
@pytest.mark.parametrize("learner", LEARNERS)
def test_every_parameter_is_block_parallel(learner):
    for name, model in make(learner).items():
        lists = [
            prefix
            for prefix, module in model.named_modules()
            if isinstance(module, nn.ParameterList) and len(module) == NUM_AGENTS
        ]
        for param_name, param in model.named_parameters():
            is_block = param.dim() >= 1 and param.shape[0] == NUM_AGENTS
            in_list = any(param_name.startswith(prefix + ".") for prefix in lists)
            assert is_block or in_list, (
                f"{learner}/{name}.{param_name} has shape {tuple(param.shape)}: not a block "
                f"parameter (leading dim {NUM_AGENTS}) and not a per-agent ParameterList entry"
            )


# ------------------------------------------------------------------ 2. independence
@pytest.mark.parametrize("learner", LEARNERS)
def test_one_agents_rows_cannot_change_another_agents_output(learner):
    """BlockBatchNorm runs in training mode here, so its batch statistics are exercised."""
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
        assign_block_slice(
            target[name], 0, 1, slice_block_state_dict(model, 2, NUM_AGENTS)
        )
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
def test_optimizer_state_slices_and_merges(learner):
    models = make(learner)
    for name, model in models.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        per_agent = [
            slice_optimizer_state(optimizer.state_dict(), agent, NUM_AGENTS)
            for agent in range(NUM_AGENTS)
        ]
        merged = merge_optimizer_states(per_agent, NUM_AGENTS)
        original = optimizer.state_dict()["state"]
        for param_id, state in original.items():
            for key, value in state.items():
                if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == NUM_AGENTS:
                    assert torch.equal(merged["state"][param_id][key], value), f"{name}.{key}"
        # and the merged state loads back into an identically shaped optimizer
        optimizer.load_state_dict(merged)


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

    assert set(MODEL_BUILDERS) == set(CONFIGURED)

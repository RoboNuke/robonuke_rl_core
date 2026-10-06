"""Every actor option has to build on the GPU, not just on the CPU.

`VmapEnsemble` builds each plain module on the CPU and moves it to the device afterwards,
while the wrapper builds its init tensors (`log_std_init`, the per-row output scale) on the
target device. Any init path that multiplies those two together is therefore CPU-vs-CUDA —
and only when the option that uses it is actually set, which is how a `last_layer_scale` plus
`scale_down_action_dims` config reached a real training run before anything caught it.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from robonuke_rl_core.models.simba import EnsembleActor, EnsembleQCritic

pytestmark = pytest.mark.gpu

OBS_DIM, ACT_DIM, NUM_AGENTS = 12, 7, 3


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"last_layer_scale": 0.1},
        {"last_layer_scale": 0.1, "scale_down_action_dims": [0, 1, 2, 3, 4, 5]},
        {"use_state_dependent_std": True},
        {"use_state_dependent_std": True, "last_layer_scale": 0.01},
        {"act_init_std": 1.0, "second_act_init_std": 0.1, "second_act_init_std_dims": [6]},
        {"bernoulli_action_dims": [6], "last_layer_scale": 0.1},
        {"force_zero_action_dims": [5], "last_layer_scale": 0.1},
    ],
)
def test_the_actor_builds_on_cuda_with_every_option(options):
    torch.manual_seed(0)
    actor = EnsembleActor(
        box(OBS_DIM), box(ACT_DIM), "cuda", num_agents=NUM_AGENTS,
        actor_n=1, actor_latent=16, **options,
    )
    for name, tensor in actor.net.state_dict().items():
        assert tensor.device.type == "cuda", f"{name} landed on {tensor.device}"

    observations = torch.randn(NUM_AGENTS * 4, OBS_DIM, device="cuda")
    actions, outputs = actor.act({"observations": observations}, role="policy")
    assert actions.device.type == "cuda" and torch.isfinite(actions).all()
    assert torch.isfinite(outputs["log_prob"]).all()


def test_the_critic_builds_on_cuda():
    critic = EnsembleQCritic(
        box(OBS_DIM), box(ACT_DIM), "cuda", num_agents=NUM_AGENTS, critic_n=1, critic_latent=16
    )
    for name, tensor in critic.net.state_dict().items():
        assert tensor.device.type == "cuda", f"{name} landed on {tensor.device}"
    values, _ = critic.act(
        {
            "observations": torch.randn(NUM_AGENTS * 4, OBS_DIM, device="cuda"),
            "taken_actions": torch.rand(NUM_AGENTS * 4, ACT_DIM, device="cuda") * 2 - 1,
        },
        role="critic",
    )
    assert values.device.type == "cuda" and torch.isfinite(values).all()

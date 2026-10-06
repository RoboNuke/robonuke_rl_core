"""MATCH end to end on a learner: the contact flags reach the loss, and the logs show it.

No env and no Isaac Lab — the controller cfg is only read for its action layout, so the
pair indices, the action width and the selection axis names are the real ones while the
transitions are hand fed. What this covers that the unit tests cannot:

* a loss's ``required_memory_keys`` really becomes a memory tensor, gets filled from
  ``infos`` and comes back in the sampled batch;
* the update consumes it and moves the weights;
* ``selection/p_force_<axis>`` is emitted, named after the controller's force-eligible axes.
"""

from __future__ import annotations

import pytest
import torch

from robonuke_rl_core.envs.cfg import ControllerCfg
from robonuke_rl_core.envs.interface import ActionLayout
from robonuke_rl_core.losses import LossesCfg, LossTermCfg, build_aux_losses
from robonuke_rl_core.models.cfg import SimbaModelCfg

from helpers import build_learner, fill_memory

NUM_AGENTS = 3
ENVS_PER_AGENT = 2
AXES = 3


def controller_cfg() -> ControllerCfg:
    return ControllerCfg(
        enabled=True, use_pose=True, use_force=True, force_axes=[1, 1, 1, 0, 0, 0]
    )


def hybrid_learner(learner: str = "sac", *, style: str = "match", with_loss: bool = True):
    controller = controller_cfg()
    layout = ActionLayout(controller)
    aux = (
        [
            build_aux_losses(
                LossesCfg(
                    terms=[
                        LossTermCfg(
                            name="supervised_selection",
                            target="policy",
                            weight=1.0,
                            kwargs={"num_axes": AXES},
                        )
                    ]
                ),
                NUM_AGENTS,
            )
        ]
        if with_loss
        else None
    )
    return build_learner(
        learner,
        num_agents=NUM_AGENTS,
        envs_per_agent=ENVS_PER_AGENT,
        controller_cfg=controller,
        action_dim=layout.action_dim,
        model_overrides={
            "bernoulli_action_dims": layout.selection_indices,
            "selection_distribution": style,
        },
        aux_loss=aux,
    )


def contact_infos(seed: int = 0):
    """``infos`` with the per-axis contact flags the contact wrapper would publish."""
    generator = torch.Generator().manual_seed(seed)

    def build(step: int, num_envs: int) -> dict:
        flags = (torch.rand(num_envs, AXES, generator=generator) > 0.5).float()
        return {"in_contact": flags}

    return build


def collector(learner, prefix: str):
    """Collect every emitted metric whose name starts with ``prefix``."""
    seen: dict = {}

    def hook(agent_idx: int, metrics: dict, step: int) -> None:
        for name, value in metrics.items():
            if name.startswith(prefix):
                seen.setdefault(name, {})[agent_idx] = value
    learner.on_log.append(hook)
    return seen


# ------------------------------------------------------------------ the memory key
def test_the_loss_s_memory_key_becomes_a_memory_tensor():
    learner = hybrid_learner()
    assert learner.aux_memory_keys() == {"in_contact": AXES}
    assert "in_contact" in learner._tensors_names
    assert learner.memory.get_tensor_by_name("in_contact").shape[-1] == AXES

    fill_memory(learner, infos_fn=contact_infos())
    sampled = learner.memory.sample(names=learner._tensors_names, batch_size=2)
    batch = dict(zip(learner._tensors_names, sampled[0]))
    assert batch["in_contact"].shape == (2 * NUM_AGENTS, AXES)
    assert set(batch["in_contact"].unique().tolist()) <= {0.0, 1.0}


def test_no_loss_means_no_extra_tensor():
    learner = hybrid_learner(with_loss=False)
    assert learner.aux_memory_keys() == {}
    assert "in_contact" not in learner._tensors_names
    fill_memory(learner)  # and an env that publishes nothing is still fine


def test_a_missing_contact_key_says_which_wrapper_is_off():
    learner = hybrid_learner()
    with pytest.raises(RuntimeError, match="in_contact"):
        fill_memory(learner)


def test_a_wrong_width_raises_rather_than_broadcasting():
    learner = hybrid_learner()

    def wrong(step: int, num_envs: int) -> dict:
        return {"in_contact": torch.zeros(num_envs, AXES + 1)}

    with pytest.raises(ValueError, match="in_contact"):
        fill_memory(learner, infos_fn=wrong)


# ------------------------------------------------------------------ the update
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_the_update_consumes_the_flags_and_moves_the_weights(learner_name):
    learner = hybrid_learner(learner_name)
    fill_memory(learner, infos_fn=contact_infos())
    before = {name: tensor.detach().clone() for name, tensor in learner.policy.named_parameters()}
    seen = collector(learner, "loss/supervised_selection")
    learner.update(timestep=8, timesteps=100)

    assert "loss/supervised_selection_policy" in seen, "the loss never ran"
    values = seen["loss/supervised_selection_policy"]
    assert len(values) == NUM_AGENTS
    assert all(float(value) > 0.0 for value in values.values())  # BCE is positive
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in learner.policy.named_parameters()
    )


def test_the_loss_changes_the_policy_gradient_it_is_added_to():
    """Same data, same seed, with and without the term: the updates must differ."""
    with_loss = hybrid_learner("sac")
    without = hybrid_learner("sac", with_loss=False)
    fill_memory(with_loss, infos_fn=contact_infos(), seed=4)
    fill_memory(without, infos_fn=contact_infos(), seed=4)
    torch.manual_seed(7)
    with_loss.update(timestep=8, timesteps=100)
    torch.manual_seed(7)
    without.update(timestep=8, timesteps=100)

    changed = [
        name
        for (name, left), (_, right) in zip(
            with_loss.policy.named_parameters(), without.policy.named_parameters()
        )
        if not torch.equal(left, right)
    ]
    assert changed, "the supervised term had no effect on the policy"


# ------------------------------------------------------------------ the logging
@pytest.mark.parametrize("learner_name", ["sac", "ppo"])
def test_the_force_probability_is_logged_per_axis_by_name(learner_name):
    learner = hybrid_learner(learner_name, with_loss=False)
    seen = collector(learner, "selection/")
    fill_memory(learner, steps=2)

    assert sorted(seen) == ["selection/p_force_x", "selection/p_force_y", "selection/p_force_z"]
    for name, per_agent in seen.items():
        assert len(per_agent) == NUM_AGENTS, name
        for value in per_agent.values():
            assert value.shape == ()  # a learner metric is 0-d
            assert 0.0 <= float(value) <= 1.0


def test_a_policy_without_selection_dims_logs_nothing():
    learner = build_learner("sac", num_agents=NUM_AGENTS, envs_per_agent=ENVS_PER_AGENT)
    seen = collector(learner, "selection/")
    fill_memory(learner, steps=2)
    assert seen == {}


def test_the_axis_names_fall_back_to_the_bit_index_without_a_controller():
    """A model built with no controller still logs, named by the bit it came from."""
    learner = build_learner(
        "sac",
        num_agents=NUM_AGENTS,
        envs_per_agent=ENVS_PER_AGENT,
        action_dim=4,
        model_overrides={"bernoulli_action_dims": [2, 3]},
    )
    seen = collector(learner, "selection/")
    fill_memory(learner, steps=2)
    assert sorted(seen) == ["selection/p_force_0", "selection/p_force_1"]

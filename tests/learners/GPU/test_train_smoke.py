"""Every learner must run the real train loop on a real Isaac Lab task and checkpoint.

Run on the GPU machine: `pytest -m gpu`. A few dozen env steps per learner, 2 agents, tiny
networks: this proves the wiring (env -> memory -> update -> checkpoint -> load), not that
anything learns.

The config and the env come from the session fixtures in `tests/conftest.py`: Isaac Lab hangs
when a second env is created in the same process, so the whole GPU suite shares one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from robonuke_rl_core.learners.base import run_dirs

pytestmark = pytest.mark.gpu

HERE = Path(__file__).resolve().parent
LEARNERS = ["sac", "ppo"]


def build_learner(cfg, env, learner_name: str, dirs):
    from robonuke_rl_core.learners.ppo import PPO
    from robonuke_rl_core.learners.sac import SAC
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory
    from robonuke_rl_core.models.factory import build_models

    classes = {"sac": SAC, "ppo": PPO}
    learner_cfg = cfg[learner_name]
    num_agents = cfg.experiment.num_agents
    # asymmetric actor-critic when the env advertises a state space
    state_space = env.state_space

    models = build_models(
        learner_name,
        cfg.model,
        env.observation_space,
        state_space,
        env.action_space,
        num_agents,
        env.device,
    )
    capacity = (
        int(learner_cfg.rollouts) * (env.num_envs // num_agents)
        if learner_name == "ppo"
        else int(cfg.memory.memory_size)
    )
    memory = MultiRandomMemory(
        capacity=capacity, num_envs=env.num_envs, num_agents=num_agents, device=env.device
    )
    extra = {"model_cfg": cfg.model} if learner_name == "sac" else {}
    return classes[learner_name](
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        state_space=state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=learner_cfg,
        trainer_cfg=cfg.trainer,
        num_agents=num_agents,
        num_envs=env.num_envs,
        run_dirs=dirs,
        **extra,
    )


@pytest.mark.parametrize("learner_name", LEARNERS)
def test_train_loop_and_checkpoints(gpu_cfg, gpu_env, tmp_path, learner_name):
    from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

    cfg, env = gpu_cfg, gpu_env
    num_agents = cfg.experiment.num_agents
    dirs = [tmp_path / f"{learner_name}_a{agent}" for agent in range(num_agents)]
    learner = build_learner(cfg, env, learner_name, dirs)

    logged: list[tuple[int, dict, int]] = []
    learner.on_log.append(lambda agent, metrics, step: logged.append((agent, metrics, step)))

    SequentialTrainer(
        env=env,
        agents=learner,
        cfg=SequentialTrainerCfg(
            timesteps=cfg.trainer.total_timesteps, headless=True, disable_progressbar=True
        ),
    ).train()

    # metrics reached the hook, per agent
    assert {agent for agent, _, _ in logged} == set(range(num_agents))
    assert all(torch.is_tensor(value) for _, metrics, _ in logged for value in metrics.values())

    # one checkpoint per agent, and it loads back into its slot
    for agent in range(num_agents):
        files = sorted((dirs[agent] / "checkpoints").glob("ckpt_*.pt"))
        assert files, f"no checkpoint for agent {agent} in {dirs[agent]}"
        meta = learner.load_agent(files[-1], slot=agent)
        assert meta["agent_idx"] == agent
        assert meta["num_agents"] == num_agents

    # the env ran with the configured partition
    assert env.num_envs == cfg.task_cfg.scene.num_envs
    assert env.num_envs % num_agents == 0


def test_run_dirs_match_the_configured_layout(gpu_cfg):
    cfg = gpu_cfg
    dirs = run_dirs(cfg)
    assert len(dirs) == cfg.experiment.num_agents
    for agent, directory in enumerate(dirs):
        assert directory.name == cfg.derived["run_names"][agent]
        assert directory.parent.name == cfg.wandb.group
        assert directory.parent.parent.name == cfg.wandb.project

"""Train `trainer.learner` on `task.name`.

    python scripts/train.py --config examples/forge_exp.yaml --headless \\
        task.cfg.scene.num_envs=128 experiment.seed=3

Every agent trains in the same Isaac Sim instance on its own block of envs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robonuke_rl_core.config import add_config_args, dump, load_from_args  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    app_launcher = AppLauncher(args)

    # Isaac Sim is up: the task registry, the env and the learners can be imported now
    import gymnasium as gym
    import torch
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

    from robonuke_rl_core.learners.base import run_dirs
    from robonuke_rl_core.losses import build_aux_losses
    from robonuke_rl_core.learners.flash_sac import FlashSAC
    from robonuke_rl_core.learners.ppo import PPO
    from robonuke_rl_core.learners.sac import SAC
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory
    from robonuke_rl_core.models.factory import build_models

    learners = {"sac": SAC, "ppo": PPO, "flash_sac": FlashSAC}

    cfg = load_from_args(args, overrides)
    learner_name = cfg.trainer.learner
    learner_cfg = cfg[learner_name]
    num_agents = cfg.experiment.num_agents
    dirs = run_dirs(cfg)

    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    # one file for the whole run, written once the env exists so it records what the env
    # really runs with (every agent in this process shares it)
    group_dir = Path(cfg.trainer.output_dir) / cfg.wandb.project / cfg.wandb.group
    print(f"[train] config: {dump(cfg, group_dir, env.unwrapped.cfg)}")
    env = wrap_env(env, wrapper="isaaclab")

    torch.manual_seed(cfg.experiment.seed)

    total_envs = env.num_envs
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

    per_env_depth = (
        int(learner_cfg.rollouts)
        if learner_name == "ppo"
        else max(1, cfg.memory.memory_size // total_envs)
    )
    memory = MultiRandomMemory(
        memory_size=per_env_depth, num_envs=total_envs, num_agents=num_agents, device=env.device
    )

    extra = {"model_cfg": cfg.model} if learner_name in ("sac", "flash_sac") else {}
    learner = learners[learner_name](
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        state_space=state_space,
        action_space=env.action_space,
        device=env.device,
        cfg=learner_cfg,
        trainer_cfg=cfg.trainer,
        num_agents=num_agents,
        num_envs=total_envs,
        run_dirs=dirs,
        **extra,
    )

    aux = build_aux_losses(cfg.losses, num_agents)
    if aux is not None:
        learner.aux_loss.append(aux)
        print(f"[train] aux losses: {[t.name for t in cfg.losses.terms]}")

    print(f"[train] {learner_name}: {num_agents} agents x {total_envs // num_agents} envs")
    print(f"[train] runs: {', '.join(str(d) for d in dirs)}")

    trainer = SequentialTrainer(
        env=env,
        agents=learner,
        cfg=SequentialTrainerCfg(timesteps=cfg.trainer.total_timesteps, headless=True),
    )
    trainer.train()

    env.close()
    app_launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

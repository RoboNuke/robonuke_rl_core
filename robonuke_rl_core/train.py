"""Train `trainer.learner` on `task.name`.

    python scripts/train.py --config examples/forge_exp.yaml --headless \\
        task.cfg.scene.num_envs=128 experiment.seed=3

Every agent trains in the same Isaac Sim instance on its own block of envs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robonuke_rl_core.config import add_config_args, dump, load_from_args  # noqa: E402


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    app_launcher = AppLauncher(args)
    if setup is not None:
        # Isaac Sim is up: a project imports its tasks (gym.register) and registers its
        # config sections, losses, overlays and model architectures here, before the
        # config loads.
        setup()


    # Isaac Sim is up: the task registry, the env and the learners can be imported now
    import gymnasium as gym
    import torch
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

    from omegaconf import OmegaConf

    from robonuke_rl_core.envs.build import build_env, describe, prepare_task
    from robonuke_rl_core.learners.base import run_dirs
    from robonuke_rl_core.logging import WandbLogger
    from robonuke_rl_core.losses import build_aux_losses
    from robonuke_rl_core.learners.ppo import PPO
    from robonuke_rl_core.learners.sac import SAC
    from robonuke_rl_core.memory.multi_random import MultiRandomMemory
    from robonuke_rl_core.models.factory import build_models

    learners = {"sac": SAC, "ppo": PPO}

    cfg = load_from_args(args, overrides)
    learner_name = cfg.trainer.learner
    learner_cfg = cfg[learner_name]
    num_agents = cfg.experiment.num_agents
    dirs = run_dirs(cfg)

    prepare_task(cfg, cfg.task_name, cfg.task_cfg)  # before gym.make: sensors and obs layout
    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    # one file for the whole run, written once the env exists so it records what the env
    # really runs with (every agent in this process shares it)
    group_dir = Path(cfg.trainer.output_dir) / cfg.wandb.project / cfg.wandb.group
    config_path = dump(cfg, group_dir, env.unwrapped.cfg)
    print(f"[train] config: {config_path}", flush=True)
    # our wrappers go on before skrl's, so the models are built from the wrapped spaces
    env = build_env(cfg, env, cfg.task_name)
    if describe(cfg):
        print(f"[train] env wrappers: {' -> '.join(describe(cfg))}", flush=True)
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

    # transitions per agent: PPO holds one rollout, SAC holds what the config asks for
    capacity = (
        int(learner_cfg.rollouts) * (total_envs // num_agents)
        if learner_name == "ppo"
        else int(cfg.memory.memory_size)
    )
    memory = MultiRandomMemory(
        capacity=capacity, num_envs=total_envs, num_agents=num_agents, device=env.device
    )

    extra = {"model_cfg": cfg.model} if learner_name == "sac" else {}
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
        print(f"[train] aux losses: {[t.name for t in cfg.losses.terms]}", flush=True)

    # one wandb run per agent, carrying exactly what resolved_config.yaml holds
    logger = WandbLogger.from_config(
        cfg,
        OmegaConf.to_container(OmegaConf.load(config_path), resolve=True),
        config_path=config_path,  # uploaded once, as a plain run file
        device=env.device,
    )
    learner.on_log.append(logger)
    learner.on_flush.append(logger.flush)
    learner.on_checkpoint.append(logger.checkpoint)  # every checkpoint mirrors to its run

    print(f"[train] {learner_name}: {num_agents} agents x {total_envs // num_agents} envs", flush=True)
    print(f"[train] runs: {', '.join(str(d) for d in dirs)}")
    print(f"[train] wandb: mode={cfg.wandb.mode} group={cfg.wandb.group}", flush=True)

    trainer = SequentialTrainer(
        env=env,
        agents=learner,
        cfg=SequentialTrainerCfg(timesteps=cfg.trainer.total_timesteps, headless=True),
    )
    try:
        trainer.train()
    finally:
        logger.close()  # publish what is pending and finish every run

    env.close()
    app_launcher.app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

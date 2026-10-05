"""Where one PPO / SAC update spends its time, and which ops sync the GPU.

Stage 2 diagnostics for the one-minute updates. Standalone: fake rollout data at real sizes,
no Isaac Lab and no env.

    python scripts/bench_update_profile.py                    # both learners
    python scripts/bench_update_profile.py --learner ppo --agents 4 --num-envs 256
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import gymnasium
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robonuke_rl_core.learners.cfg import PPOCfg, SACCfg, TrainerCfg  # noqa: E402
from robonuke_rl_core.learners.ppo import PPO  # noqa: E402
from robonuke_rl_core.learners.sac import SAC  # noqa: E402
from robonuke_rl_core.memory.multi_random import MultiRandomMemory  # noqa: E402
from robonuke_rl_core.models.cfg import SimbaActorCfg, SimbaCriticCfg, SimbaModelCfg  # noqa: E402
from robonuke_rl_core.models.factory import build_models  # noqa: E402

OBS_DIM, STATE_DIM, ACT_DIM = 44, 56, 12
CLASSES = {"sac": SAC, "ppo": PPO}


def box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def build(learner_name: str, agents: int, num_envs: int, device: str, rollouts: int, batch: int):
    model_cfg = SimbaModelCfg(actor=SimbaActorCfg(), critic=SimbaCriticCfg())  # real widths: 512 x 2
    if learner_name == "ppo":
        cfg = PPOCfg(rollouts=rollouts, learning_epochs=4, mini_batches=4, learning_starts=0)
        depth = rollouts
    else:
        cfg = SACCfg(batch_size=batch, gradient_steps=1, learning_starts=0)
        depth = 64
    observation_space, state_space, action_space = box(OBS_DIM), box(STATE_DIM), box(ACT_DIM)
    models = build_models(
        learner_name, model_cfg, observation_space, state_space, action_space, agents, device
    )
    memory = MultiRandomMemory(
        memory_size=depth, num_envs=num_envs, num_agents=agents, device=device
    )
    extra = {"model_cfg": model_cfg} if learner_name == "sac" else {}
    learner = CLASSES[learner_name](
        models=models,
        memory=memory,
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        cfg=cfg,
        trainer_cfg=TrainerCfg(learner=learner_name, total_timesteps=10_000),
        num_agents=agents,
        num_envs=num_envs,
        **extra,
    )
    learner.init()
    learner.enable_training_mode(True)

    # fill the buffer with plausible data, through the learner's own record path
    torch.manual_seed(0)
    for step in range(depth):
        observations = torch.randn(num_envs, OBS_DIM, device=device)
        states = torch.randn(num_envs, STATE_DIM, device=device)
        actions, _ = learner.act(observations, states, timestep=step, timesteps=10_000)
        learner.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=torch.randn(num_envs, 1, device=device),
            next_observations=torch.randn(num_envs, OBS_DIM, device=device),
            next_states=torch.randn(num_envs, STATE_DIM, device=device),
            terminated=torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            truncated=torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            infos={},
            timestep=step,
            timesteps=10_000,
        )
    learner._next_observations = torch.randn(num_envs, OBS_DIM, device=device)
    learner._next_states = torch.randn(num_envs, STATE_DIM, device=device)
    return learner


def report(learner_name: str, learner, device: str, rows: int) -> None:
    update = lambda: learner.update(timestep=1000, timesteps=10_000)  # noqa: E731

    update()  # warm up (allocator, autotune)
    if device == "cuda":
        torch.cuda.synchronize()

    # wall time
    start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    if start is not None:
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        update()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        import time

        begin = time.perf_counter()
        update()
        elapsed_ms = (time.perf_counter() - begin) * 1e3
    print(f"\n=== {learner_name}: one update = {elapsed_ms:.1f} ms")

    # implicit GPU->CPU syncs inside the update
    if device == "cuda":
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            update()
            torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("default")
        syncs = [str(w.message) for w in caught if "sync" in str(w.message).lower()]
        print(f"{learner_name}: {len(syncs)} implicit GPU->CPU sync(s) inside the update")
        for message in syncs[:5]:
            print(f"  - {message.splitlines()[0]}")

    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        update()
        if device == "cuda":
            torch.cuda.synchronize()
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", nargs="*", default=["ppo", "sac"], choices=list(CLASSES))
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(
        f"device={args.device} agents={args.agents} num_envs={args.num_envs} "
        f"rollouts={args.rollouts} batch={args.batch}"
    )
    for learner_name in args.learner:
        learner = build(
            learner_name, args.agents, args.num_envs, args.device, args.rollouts, args.batch
        )
        report(learner_name, learner, args.device, args.num_envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

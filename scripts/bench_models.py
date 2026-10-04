"""Forward+backward wall time: vmap over N plain models vs a sequential loop over them.

This was the decision gate for replacing the hand-written block (einsum) layers with vmap;
on a 5090 vmap came out at 0.75-0.97x of the einsum path at every real size, so the einsum
layers are gone and the comparison here is now vmap vs the obvious alternative, one model at
a time. Runs standalone — no Isaac Lab, no env.

    python scripts/bench_models.py                 # the default size sweep
    python scripts/bench_models.py --agents 4 --rows 256 --repeats 100
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEFAULT_AGENTS = (1, 4, 8)
#: (name, obs_dim, hidden, out_dim, blocks, rows) at the sizes the real configs use
DEFAULT_CASES = (
    ("actor  512x2", 44, 512, 12, 2, 256),
    ("critic 512x2", 56, 512, 1, 2, 256),
    ("critic 512x2 (ppo minibatch)", 56, 512, 1, 2, 1024),
    ("actor  1024x3", 44, 1024, 12, 3, 256),
)


class PlainSimBa(nn.Module):
    """The SimBa trunk for one agent (the shape robonuke_rl_core/models/simba.py builds)."""

    def __init__(self, obs_dim: int, hidden: int, out_dim: int, blocks: int):
        super().__init__()
        self.fc_in = nn.Linear(obs_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln": nn.LayerNorm(hidden),
                        "fc1": nn.Linear(hidden, 4 * hidden),
                        "fc2": nn.Linear(4 * hidden, hidden),
                    }
                )
                for _ in range(blocks)
            ]
        )
        self.ln_out = nn.LayerNorm(hidden)
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        for block in self.blocks:
            x = x + block["fc2"](F.relu(block["fc1"](block["ln"](x))))
        return self.fc_out(self.ln_out(x))


def timed(step, repeats: int, warmup: int, device: str) -> float:
    """Mean microseconds per forward+backward."""
    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / repeats
    begin = time.perf_counter()
    for _ in range(repeats):
        step()
    return (time.perf_counter() - begin) * 1e6 / repeats


def bench_loop(agents, obs_dim, hidden, out_dim, blocks, rows, device, repeats, warmup):
    """One model at a time: the baseline vmap has to beat."""
    models = [PlainSimBa(obs_dim, hidden, out_dim, blocks).to(device) for _ in range(agents)]
    x = torch.randn(agents, rows, obs_dim, device=device)

    def step():
        for model in models:
            for param in model.parameters():
                param.grad = None
        out = torch.stack([model(x[i]) for i, model in enumerate(models)])
        out.square().mean().backward()

    return timed(step, repeats, warmup, device)


def bench_vmap(agents, obs_dim, hidden, out_dim, blocks, rows, device, repeats, warmup):
    models = [PlainSimBa(obs_dim, hidden, out_dim, blocks).to(device) for _ in range(agents)]
    params, buffers = stack_module_state(models)
    params = {k: nn.Parameter(v.detach().clone()) for k, v in params.items()}
    meta = copy.deepcopy(models[0]).to("meta")

    def fmodel(p, b, inputs):
        return functional_call(meta, (p, b), (inputs,))

    batched = torch.vmap(fmodel)
    x = torch.randn(agents, rows, obs_dim, device=device)

    def step():
        for param in params.values():
            param.grad = None
        out = batched(params, buffers, x)
        out.square().mean().backward()

    return timed(step, repeats, warmup, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, nargs="*", default=list(DEFAULT_AGENTS))
    parser.add_argument("--rows", type=int, default=None, help="override every case's row count")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"device={args.device} torch={torch.__version__} repeats={args.repeats}")
    if args.device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    header = f"{'case':<32} {'N':>3} {'rows':>5} {'loop us':>10} {'vmap us':>10} {'vmap/loop':>12}"
    print(header)
    print("-" * len(header))

    verdicts = []
    for name, obs_dim, hidden, out_dim, blocks, rows in DEFAULT_CASES:
        rows = args.rows or rows
        for agents in args.agents:
            sizes = (agents, obs_dim, hidden, out_dim, blocks, rows, args.device)
            loop_us = bench_loop(*sizes, args.repeats, args.warmup)
            vmap_us = bench_vmap(*sizes, args.repeats, args.warmup)
            ratio = vmap_us / loop_us
            verdicts.append(ratio)
            print(
                f"{name:<32} {agents:>3} {rows:>5} {loop_us:>10.1f} {vmap_us:>10.1f} "
                f"{ratio:>11.2f}x"
            )

    worst = max(verdicts)
    print(f"\nworst vmap/loop ratio: {worst:.2f}x (lower is better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

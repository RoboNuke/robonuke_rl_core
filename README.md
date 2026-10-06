# robonuke_rl_core

End-to-end components for RL research on Isaac Lab: model definitions, skrl integration, a
training pipeline, an evaluation pipeline, configuration and experiment tracking, and modular
env definitions.

Several agents train in parallel in one Isaac Sim instance on one GPU. Agent `i` owns a
contiguous block of envs; the models are ordinary single-agent `nn.Module`s stacked with
`torch.vmap` so every agent runs in one pass, one `BlockAdamW` is N independent AdamW
instances over the stacked parameters, and nothing computed from agent `j`'s data touches
agent `i`'s update.

Built area by area. Today: the config manager (`robonuke_rl_core/config.py`), the learners
(`learners/`), the models (`models/`), the optimizer (`optim.py`), the memory (`memory/`) and
the auxiliary losses (`losses/`).

## Install

The package assumes an **Isaac Lab environment is already installed** and documents what it
was built against — it never installs Isaac Lab, torch or gymnasium itself:

* Python 3.11, Isaac Lab **0.47.1**, Isaac Sim **5.1.0**, torch 2.7.0+cu128, gymnasium 1.2.1

Install it **once, editable, from one clone** into that environment; every project then
imports the same copy, and a `git pull` in the clone updates all of them (each run records
`meta.pkg_commit`, so results stay attributable to a package state):

```bash
conda activate general
pip install -e ~/robonuke_rl_core
```

The remaining dependencies (omegaconf, skrl, wandb, pandas, pyarrow, imageio) install
automatically with the package. A project repo consuming the package starts from
https://github.com/RoboNuke/robonuke_project_template ("Use this template"): its scripts call
`robonuke_rl_core.train.main(setup=...)` and register the project's tasks and config sections
in the `setup` hook.

## Running

Run from the repo root, with the `general` conda env's interpreter (`conda activate general`,
or call `/home/hunter/miniconda3/envs/general/bin/python` directly — the base env has no Isaac
Lab):

```bash
# train: every agent in one Isaac Sim instance, one wandb run each
python scripts/train.py --config examples/forge_exp.yaml --headless \
    task.cfg.scene.num_envs=128 experiment.seed=3

# evaluate a trained policy under the conditions an eval config names
python scripts/eval.py --run entity/project/run_id \
    --eval_config examples/eval/quick.yaml --headless [--checkpoint best] [--record]
python scripts/eval.py --local runs/proj/group/group_a0 \
    --eval_config examples/eval/quick.yaml --headless

# watch one env: the policy acting, or the initial conditions it spawns
python scripts/debug.py --run entity/project/run_id
python scripts/debug.py --local runs/proj/group/group_a0 --resets --hold_seconds 2
python scripts/debug.py --run entity/project/run_id --resets --headless \
    --num_resets 10 --out resets.mp4
```

`--run` takes `entity/project/<run id or exact run name>` and pulls the config and checkpoint
from that run's files; `--local` takes a run directory instead. `--checkpoint` accepts `best`
(the default), a step number, or a file name. An eval writes `summary.yaml`, a per-step
`.parquet` trace, its own resolved config and (with `--record`) one mp4 per episode under
`<run dir>/eval/<eval-config-stem>_<timestamp>/`, and with `--run` mirrors them back to the
training run. See `CLAUDE.md` for the episode accounting rule and the debug keys.

## Tests

```bash
/home/hunter/miniconda3/envs/general/bin/python -m pytest       # CPU tests
/home/hunter/miniconda3/envs/general/bin/python -m pytest -m gpu  # needs Isaac Sim + a GPU
```

See `CLAUDE.md` for the development rules, the config pipeline, and the checklists for adding
a config class, a learner, a model or a loss.

---

# Configuration reference

Every configurable parameter of every registered section. One YAML file sets `base` and
overrides only what differs from these defaults; the CLI overrides anything with a dotted
path. `required` means no default: some layer must set it.

The task's own config is not listed here: **any field of the Isaac Lab env cfg can be
overridden under `task.cfg.*`** (for example `task.cfg.scene.num_envs`). A few fields cannot
be changed that way — see "Task cfg fields an override cannot change" in `CLAUDE.md`.

```yaml
base: ../base/forge.yaml        # optional; relative to this file
task:
  name: Isaac-Forge-PegInsert-Direct-v0
  cfg:
    scene:
      num_envs: 256
experiment:
  num_agents: 4
  seed: 10
```

## experiment

| field | type | default | what it does |
| --- | --- | --- | --- |
| `num_agents` | int | `1` | independent agents trained in parallel; `task.cfg.scene.num_envs` must divide by it |
| `seed` | int | required | the one seed for the run; also written to `task.cfg.seed` |

## wandb

Run names derive from `group`: `{group}_a{i}` for agent `i` (`derived.run_names`). There is no
run-name field.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `entity` | str | required | wandb entity |
| `project` | str | required | wandb project; also the first level of the run directory |
| `group` | str | required | wandb group; run names and the run directory derive from it. No whitespace or `/` |
| `tags` | list[str] | `[]` | wandb tags |
| `mode` | str | `"online"` | `online`, `offline` or `disabled`; passed to `wandb.init`. One run per agent either way |

Each run's Files tab gets `resolved_config.yaml` and every checkpoint (`ckpt_{step}.ckpt`,
`ckpt_best.ckpt`) as ordinary run files — the Artifacts API is never used. That is what
`scripts/eval.py --run entity/project/run` downloads, and an eval pushes its own results back
to the same run under `eval/<eval-config-stem>_<timestamp>/`.

## trainer

| field | type | default | what it does |
| --- | --- | --- | --- |
| `learner` | str | required | which learner runs: `sac` or `ppo`. Only that section is used; both are dumped |
| `total_timesteps` | int | required | env steps to train for |
| `write_interval` | int | `1000` | env steps between episode-stat flushes through the `on_log` hooks (0 disables) |
| `checkpoint_interval` | int | `10000` | env steps between per-agent checkpoints (0 disables) |
| `output_dir` | str | `"runs"` | run directories are `{output_dir}/{wandb.project}/{wandb.group}/{run_name}` |

## sac

Soft Actor-Critic. `batch_size` and the memory are per agent.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `discount_factor` | float | `0.99` | reward discount |
| `random_timesteps` | int | `0` | env steps of uniform random actions before the policy is used |
| `learning_starts` | int | `0` | env steps before the first update |
| `weight_decay` | float | `0.0` | AdamW decoupled weight decay (not applied to the temperature) |
| `grad_norm_clip` | float | `0.0` | per-agent gradient-norm clip; each agent is clipped by its own norm (0 disables) |
| `normalize_observations` | bool | `True` | per-agent running normalization of observations (and states, when asymmetric) |
| `normalizer_clip` | float | `5.0` | normalized values are clipped to +-this |
| `rewards_shaper` | str or null | `null` | `"module:name"` of `f(rewards, timestep, timesteps) -> rewards`; applied to stored rewards, not to episode stats |
| `gradient_steps` | int | `1` | update steps per env step, once `learning_starts` is passed |
| `batch_size` | int | `64` | transitions sampled **per agent** per gradient step |
| `polyak` | float | `0.005` | target-critic update rate, in (0, 1] |
| `actor_lr` | float | `1.0e-3` | policy learning rate |
| `critic_lr` | float | `1.0e-3` | critic learning rate |
| `entropy_lr` | float | `1.0e-3` | temperature learning rate |
| `lr_schedule` | str | `"constant"` | `constant` or `cosine` (cosine decays actor and critic to `lr_end`). No data-driven schedule: one optimizer over block parameters holds one LR, which would couple the agents |
| `lr_end` | float | `1.5e-4` | floor of the cosine schedule |
| `learn_entropy` | bool | `True` | learn the temperature per agent |
| `initial_entropy_value` | float | `0.2` | starting temperature |
| `target_entropy` | float or null | `null` | target entropy; `null` means `-|A|` |
| `entropy_loss_form` | str | `"log_alpha"` | `log_alpha` (skrl's form) or `alpha` (the gentler multiplicative form) |
| `periodic_reset_enabled` | bool | `False` | SimBa periodic reset: rebuild networks and optimizers, keep the replay buffer and the normalizer stats |
| `periodic_reset_frequency` | int | `0` | env steps between resets; must be > 0 when resets are enabled |
| `periodic_reset_max` | int | `0` | maximum number of resets (0 = unlimited) |

## ppo

Proximal Policy Optimization. The rollout buffer is `rollouts` steps per env;
`rollouts * envs_per_agent` must divide by `mini_batches`.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `discount_factor` | float | `0.99` | reward discount |
| `random_timesteps` | int | `0` | env steps of uniform random actions before the policy is used |
| `learning_starts` | int | `0` | env steps before the first update |
| `weight_decay` | float | `0.0` | AdamW decoupled weight decay |
| `grad_norm_clip` | float | `0.0` | per-agent gradient-norm clip for the policy and the value net (0 disables) |
| `normalize_observations` | bool | `True` | per-agent running normalization of observations (and states, when asymmetric) |
| `normalizer_clip` | float | `5.0` | normalized values are clipped to +-this |
| `rewards_shaper` | str or null | `null` | `"module:name"` of `f(rewards, timestep, timesteps) -> rewards` |
| `rollouts` | int | `16` | rollout length per env; an update runs every `rollouts` steps |
| `learning_epochs` | int | `8` | passes over each rollout |
| `mini_batches` | int | `2` | minibatches per epoch; each holds the same number of rows per agent |
| `gae_lambda` | float | `0.95` | GAE lambda. Advantages are normalized **per agent** |
| `policy_lr` | float | `3.0e-4` | policy learning rate |
| `value_lr` | float | `3.0e-4` | value learning rate |
| `lr_schedule` | str | `"constant"` | `constant` or `cosine` |
| `lr_end` | float | `1.5e-4` | floor of the cosine schedule |
| `ratio_clip` | float | `0.2` | PPO surrogate clip |
| `value_clip` | float | `0.2` | value clip around the stored value (0 disables) |
| `entropy_loss_scale` | float | `0.0` | entropy bonus weight (0 disables the term) |
| `value_loss_scale` | float | `1.0` | value-loss weight |
| `kl_threshold` | float | `0.0` | per-agent KL early stop: an agent above this is frozen — policy **and** critic, including the `value_update_ratio` passes (weights and Adam moments unchanged) — for the rest of the epoch while the others continue (0 disables) |
| `time_limit_bootstrap` | bool | `False` | add `discount_factor * V(next)` to the reward on truncation. **Set this True on Factory/Forge-style tasks**, which raise `terminated` and `truncated` together at the time limit; left False, every timeout is treated as a terminal state. SAC needs no flag: it bootstraps unless `terminated & ~truncated` |
| `normalize_values` | bool | `True` | per-agent running normalization of values and returns |
| `value_update_ratio` | int | `1` | extra value-only updates per minibatch after the combined update (1 = none) |

## model

The networks: plain single-agent modules, stacked across agents with `torch.vmap`. The keys
are always `actor.*` and `critic.*`; `architecture` picks which architecture's fields sit
behind them (the way `task.name` picks the env cfg), and another architecture's fields are
rejected loudly. The only architecture today is **SimBa**
([Lee et al., 2025](https://arxiv.org/abs/2410.09754)) — residual MLP blocks with LayerNorm.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `architecture` | str | `"simba"` | which registered architecture's dataclasses back `actor` / `critic`; the last layer that sets it wins |

Fields for `architecture: simba`:

| field | type | default | what it does |
| --- | --- | --- | --- |
| `actor.actor_n` | int | `2` | residual blocks in the actor backbone |
| `actor.actor_latent` | int | `512` | actor hidden width |
| `actor.act_init_std` | float | `0.60653066` | initial action standard deviation |
| `actor.second_act_init_std` | float or null | `null` | a second initial std, for the dims in `second_act_init_std_dims` |
| `actor.second_act_init_std_dims` | list[int] or null | `null` | action dims that start at `second_act_init_std` |
| `actor.last_layer_scale` | float | `1.0` | scale applied to the action-mean output weights at init |
| `actor.clip_log_std` | bool | `True` | clamp log_std to `[min_log_std, max_log_std]` |
| `actor.min_log_std` | float | `-20.0` | lower log_std bound; must be below `max_log_std` |
| `actor.max_log_std` | float | `2.0` | upper log_std bound |
| `actor.reduction` | str | `"sum"` | how skrl reduces the log-probability over action dims: `sum`, `mean`, `prod` or `none` |
| `actor.use_state_dependent_std` | bool | `False` | predict log_std from the observation instead of a learned per-agent parameter |
| `actor.bernoulli_action_dims` | list[int] or null | `null` | action dims drawn from a Bernoulli and mapped to {-1, +1} (e.g. a gripper) |
| `actor.force_zero_action_dims` | list[int] or null | `null` | action dims the policy never produces; emitted as 0, with no parameters |
| `actor.scale_down_action_dims` | list[int] or null | `null` | action dims that `last_layer_scale` applies to (default: all) |
| `actor.selection_distribution` | str | `"product"` | how the selection dims and the continuous dims form one joint distribution: `product` (independent) or `match` (each selection dim picks which of its (pose, force) pair is live). `match` needs `bernoulli_action_dims` and a controller with a selection block; the pair indices are derived from the action layout, never configured |
| `actor.selection_init_bias` | float | `0.0` | added to the selection logits' output bias at init, exactly as written. A bit of 1 is `S = 1`, i.e. force control, so `-2.2` (`sigmoid(-2.2) ≈ 0.1`) starts the policy ~90% position-dominant. Needs `bernoulli_action_dims` |
| `critic.critic_n` | int | `2` | residual blocks in a critic backbone |
| `critic.critic_latent` | int | `512` | critic hidden width |
| `critic.critic_output_init_mean` | float | `0.0` | initial critic output bias |
| `critic.clip_actions` | bool | `False` | clip actions to the action space inside the critic |

## memory

| field | type | default | what it does |
| --- | --- | --- | --- |
| `memory_size` | int | `1000000` | SAC replay capacity in transitions **per agent**, used exactly as given (any positive int). PPO's buffer is sized by `ppo.rollouts` instead: `rollouts * envs_per_agent` transitions per agent |

## losses

Auxiliary loss terms added to a learner's policy or critic loss. A list, so a new loss needs no
new config fields. An empty list means no extra loss and no hook.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `terms` | list | `[]` | the terms below, in order |
| `terms[i].name` | str | required | a registered loss name |
| `terms[i].target` | str | required | which loss it feeds: `policy` or `critic`; the loss must support it |
| `terms[i].weight` | float | required | multiplies the term's per-agent mean |
| `terms[i].kwargs` | dict | `{}` | constructor arguments for the loss class |

```yaml
losses:
  terms:
    - name: my_project_loss
      target: policy
      weight: 0.1
```

The package ships **one** built-in loss; every other `name` comes from a project's own
`@register_loss` (registered before the config is loaded). The bar for shipping one is that
it cannot be written as a reward: a penalty on action magnitude belongs in the env's reward,
not in a policy-side term. See the "Add a loss" checklist in `CLAUDE.md`.

| name | target | kwargs | what it does |
| --- | --- | --- | --- |
| `supervised_selection` | `policy` | `num_axes` (int, required): the number of selection dims, i.e. `sum(controller.force_axes)` | binary cross entropy between the actor's per-axis probability of **force** control and whether that axis is in contact. Needs `wrappers.contact.enabled` (it publishes the per-transition flags) and a policy with selection dims |

```yaml
losses:
  terms:
    - name: supervised_selection
      target: policy
      weight: 1.0
      kwargs:
        num_axes: 3
```

## eval

Read by `scripts/eval.py` and `scripts/debug.py`; training ignores it. An eval layers its own
config file on top of the run being evaluated (`--run entity/project/run --eval_config <file>`,
or `--local <run_dir>`), and that file is where the test conditions (`task.cfg.*`), the env
count and these fields are set. Anything it leaves out keeps the trained run's value.

`num_rollouts` counts **valid episodes, not steps**: the runner takes
`ceil(num_rollouts / task.cfg.scene.num_envs)` rounds, and each env contributes exactly one
episode per round — its first, which either hit a terminal condition or ran out of the
`max_episode_length` budget. Everything an env does after its first done is masked out of the
data, the counts and the video. The last round uses only as many envs as are still needed.

An episode counts as terminal only when `terminated` is set **without** `truncated`: Isaac
Lab's Factory and Forge tasks raise both at the time limit (`_get_dones` returns the same
`time_out` tensor twice), so anything else would report every timeout as a terminal outcome.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `num_rollouts` | int | `64` | valid episodes to collect |
| `deterministic` | bool | `True` | act on the distribution's mean (Bernoulli dims thresholded); `false` samples |
| `save_state` | bool | `True` | capture the full per-step state (observations, states, actions, rewards, dones, both metric channels, `infos["eval_state"]`) into the eval's `<stem>.parquet` trace, one row per (round, env, step) |
| `record` | bool | `False` | write one mp4 per (round, env); `--record` forces it on. Use few envs by setting `task.cfg.scene.num_envs` in the eval config |
| `overlays` | list[str] | `["hud"]` | registered overlay names, drawn in this order |
| `video_fps` | int | `30` | playback rate of the written mp4s |
| `video_height` | int | `180` | per-env camera height in pixels |
| `video_width` | int | `240` | per-env camera width in pixels |

```yaml
# quick.yaml -- used as: --run entity/project/run_id --eval_config quick.yaml
eval:
  num_rollouts: 256
task:
  cfg:
    scene:
      num_envs: 64
```

## controller

The unified operational-space controller, attached as an env wrapper when `enabled`. One
torque path serves every configuration:

```
tau = J^T [ (I - S) (K e_pose - D v) + S K_f (f_d - f) ] + nullspace
```

**`S` selects force: `S = 1` on an axis means that axis is force-controlled, `S = 0` means
position-controlled.** One convention, everywhere — the policy's selection bit, the action
vector, the `selection/*` metrics and this matrix all read "1 is force".
**The action layout is inferred from the capabilities, never configured** — there is no mode
field. The selection block exists exactly when both branches are on, because that is the only
case where the policy has a choice to make:

| `use_pose` | `use_force` | action layout | fixed |
| --- | --- | --- | --- |
| ✓ | — | `[pose \| gains?]` | `S = 0`, `f_d = 0` |
| — | ✓ | `[force \| gains?]` | `S = I` |
| ✓ | ✓ | `[pose \| selection \| force \| gains?]` | — |

`gains?` is empty under `gain_mapping: constant`; under `variable_diagonal` it is one action
per axis of each live branch (`[K \| K_f]`). How wide the force side is comes from
`force_axes`, which defaults to `[1,1,1,0,0,0]` — the 3-D hybrid, force on the translation
axes with orientation always position-controlled (13 action dims with constant gains on
Forge). `[1]*6` is 6-D (19) and `[0,0,1,0,0,0]` is force on z alone (9); force-only control
(`use_pose: false`) must set the mask to all ones, since every axis then needs a controller.
Selection actions are ±1 by construction (**+1 is force**), so
`model.actor.bernoulli_action_dims` must be exactly the selection block's indices — the config
refuses to load otherwise.

Everything Forge's own control config already defines — `ema_factor`, the dead zone,
`pos_action_bounds`, `default_task_prop_gains`, `kp_null`/`kd_null` — is **not** repeated
here. The wrapper reads it from the live env at runtime, so those defaults are Forge's and an
experiment tunes them under `task.cfg.ctrl.*`.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `enabled` | bool | `False` | attach the controller wrapper at all; off leaves the env's own controller in charge |
| `use_pose` | bool | `True` | pose branch on, and pose-target actions in the layout |
| `use_force` | bool | `False` | force branch on, and force-target actions in the layout |
| `gain_mapping` | str | `"constant"` | `constant` (gains from config, 0 action dims) or `variable_diagonal` (one gain action per axis of each live branch, geometric per-axis scaling) |
| `force_axes` | list[int] | `[1, 1, 1, 0, 0, 0]` | length-6 binary mask `[x, y, z, Rx, Ry, Rz]` of the axes the force branch may take — this is what makes hybrid control 3-D (the default: force on translation, orientation always position-controlled), 6-D (`[1]*6`), or z-only (`[0,0,1,0,0,0]`). The selection, force-target and K_f blocks are each as wide as its sum; an axis outside it is always position-controlled. Force-only control (`use_pose: false`) requires all ones |
| `native_action_dim` | int | `7` | the env's **own** action width, handed through untouched as the first block: Forge is 7 (3 position + 3 rotation + the success prediction its reward reads), Factory is 6. Re-checked against the live env, which names the right value |
| `gain_min` | list[float] | `[100, 100, 100, 5, 5, 5]` | per-axis lower bound of the pose stiffness K, `[x, y, z, Rx, Ry, Rz]` |
| `gain_max` | list[float] | `[2000, 2000, 2000, 100, 100, 100]` | per-axis upper bound of K; also the constant K when the env supplies none |
| `damping_ratio` | float | `1.0` | `D = 2 * damping_ratio * sqrt(K)`; 1.0 is critical damping. D is always derived, never commanded |
| `force_gain_min` | list[float] | `[1, 1, 1, 1, 1, 1]` | per-axis lower bound of the force stiffness K_f |
| `force_gain_max` | list[float] | `[100, 100, 100, 10, 10, 10]` | per-axis upper bound of K_f |
| `default_force_gains` | list[float] | `[0.1, 0.1, 0.1, 0.01, 0.01, 0.01]` | K_f under `gain_mapping: constant` (Forge's ctrl cfg has no force gains) |
| `force_target_bounds` | list[float] | `[50, 50, 50, 5, 5, 5]` | the force action in [-1, 1] scales to ± these, in N and Nm |

Setting any force field while `use_force: false` raises, rather than being silently ignored.

## wrappers

Env wrappers, each off by default. They are applied in a fixed order (controller → efficient
reset → fragile → contact); see `CLAUDE.md` for why.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `fragile.enabled` | bool | `False` | terminate an env when the held object's contact load breaks it |
| `fragile.break_force` | list[float] | `[100.0]` | `[magnitude]`, or `[shear, normal]` with `direction_break_force` |
| `fragile.direction_break_force` | bool | `False` | split the load into shear and axial components on the live peg axis instead of one magnitude |
| `fragile.require_contact` | bool | `False` | also fail an episode that loses contact after making it; needs `contact.enabled` |
| `fragile.require_contact_grace_steps` | int | `5` | steps at the start of an episode where loss of contact cannot fail it |
| `fragile.require_contact_debounce_steps` | int | `3` | consecutive out-of-contact steps before loss of contact counts as a break (1 = any single step) |
| `efficient_reset.enabled` | bool | `False` | reset a finished env by teleporting it onto a donor's cached fresh state, instead of running the task's own all-envs reset. **Required when `fragile.enabled`** (the config refuses the pair otherwise): a peg breaks one env at a time, and Factory/Forge's reset path is written assuming every env resets together. Eval uses it too — the teleported episodes are never counted, because an env stops being tracked when its first episode of the round closes |
| `contact.enabled` | bool | `False` | mount a contact sensor on the held asset and publish per-axis in-contact flags. With a controller that has a selection block it also publishes `infos["in_contact"]`, the flags in selection order, which is what the `supervised_selection` loss trains against — and it then refuses a force-eligible rotation axis, since a contact force says nothing about a torque. Needs `task.cfg.scene.clone_in_fabric: false` (a contact reporter needs real per-env prims) |
| `contact.force_threshold` | float | `1.0` | \|f\| above this on an end-effector axis counts as contact, in N |
| `contact.append_to_policy_obs` | bool | `False` | append the 3 flags to the policy observation (grows the observation space) |
| `contact.append_to_critic_state` | bool | `False` | append the 3 flags to the critic state (asymmetric tasks only) |
| `contact.held_prim_expr` | str | `"/World/envs/env_.*/HeldAsset"` | prim path of the sensor's asset root |
| `contact.fixed_prim_expr` | str | `"/World/envs/env_.*/FixedAsset"` | prim path the contact is filtered against |
| `task_metrics.enabled` | bool | `True` | publish the task's own outcomes **per agent** (success, termination cause, reward terms, `Success_Prediction/*`). The env logs the same things already, but averaged over every env, which mixes the agents. Skipped on a non-Forge task |
| `orientation.mode` | str | `"quat"` | `quat` (the env's own `(w, x, y, z)`) or `6d_rot_mat` (first two columns of R, Zhou et al. 2019 — continuous, no double cover) |

## hpc

SLURM resources and container details for the launchers (`launch_train`, `launch_sweep`,
`launch_eval`). **Resources are config, not a shell file**: cluster-wide values live in a base
YAML in the project, an experiment overrides what differs, and a one-off goes on the CLI
(`hpc.time=2-00:00:00`). The section rides the normal chain, so `resolved_config.yaml` records
the resources each run actually had.

**Every field defaults**, empty where there is no sane default — a required field here would
break every local training run that never touches SLURM. `validate` checks *formats only*, and
only on values that are set; the four fields a job cannot be built without (`account`,
`partitions`, `sif_image`, `cache_home`) are enforced by the submitter, just before it queues
anything, with an error naming each one.

**`WANDB_API_KEY` is an environment variable, never a config field.** The launcher passes
`sbatch --export=ALL`, which carries it from your login shell into the job. With no key (or a
key shorter than 40 characters, i.e. a bad paste) the job logs offline rather than dying —
an unattended run must not be killed by its logging.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `account` | str | `""` | `-A`, the allocation to charge. **Required at submit** |
| `partitions` | str | `""` | `-p`, comma-separated and tried in order. **Required at submit** |
| `time` | str | `"0-09:00:00"` | `--time`, as SLURM's `[D-]HH:MM:SS` |
| `gpus` | int | `1` | `--gres=gpu:{gpus}`. The package trains every agent in one process on one GPU, so everything is written and tested for 1 |
| `mem` | str | `"32G"` | `--mem` |
| `cpus` | int | `12` | `-c`, CPUs per task |
| `signal` | str | `"TERM@300"` | `--signal`; SLURM warns the job this far ahead of the walltime kill. The job `exec`s python, so the signal reaches the training process and not bash |
| `exp_log_dir` | str | `"exp_logs"` | where `.out` / `.err` land, relative to the project root; logs go to `{exp_log_dir}/{project}/{name}_%j.out` |
| `sif_image` | str | `""` | absolute path to the `.sif` on the cluster. **Required at submit** |
| `apptainer_bin` | str | `"apptainer"` | or `"singularity"` |
| `container_python` | str | `"python"` | the python inside the image |
| `cache_home` | str | `""` | bound as the container `HOME`. Kit and shader caches land here and run to GBs, so it must be scratch, **never an NFS home with a quota**. **Required at submit** |
| `binds` | list[str] | `[]` | extra `host:container` mounts |

## derived and meta

Written by the pipeline into `resolved_config.yaml`; never set them in a config.

| key | what it is |
| --- | --- |
| `derived.run_names` | `[f"{wandb.group}_a{i}" for i in range(experiment.num_agents)]` |
| `meta.pkg_commit` | git commit of the package that ran |
| `meta.project_commit` | git commit of the working directory's repo, or null |
| `meta.created` | UTC timestamp of the run |

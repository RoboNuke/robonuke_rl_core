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

**The package ships no built-in loss**, so every `name` comes from a project's own
`@register_loss` (registered before the config is loaded). A penalty on action magnitude
belongs in the env's reward, not in a policy-side term. See the "Add a loss" checklist in
`CLAUDE.md`.

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
tau = J^T [ S (K e_pose - D v) + (I - S) K_f (f_d - f) ] + nullspace
```

`S` multiplies the pose branch, so `S = 1` on an axis means that axis is position-controlled.
**The action layout is inferred from the capabilities, never configured** — there is no mode
field. The selection block exists exactly when both branches are on, because that is the only
case where the policy has a choice to make:

| `use_pose` | `use_force` | action layout | fixed |
| --- | --- | --- | --- |
| ✓ | — | `[pose \| gains?]` | `S = I`, `f_d = 0` |
| — | ✓ | `[force \| gains?]` | `S = 0` |
| ✓ | ✓ | `[pose \| selection \| force \| gains?]` | — |

`gains?` is empty under `gain_mapping: constant`; under `variable_diagonal` it is one action
per axis of each live branch (`[K \| K_f]`). How wide the force side is comes from
`force_axes`: `[1,1,1,0,0,0]` is the usual 3-D hybrid (12 action dims with constant gains),
`[1]*6` is 6-D (18), `[0,0,1,0,0,0]` is force on z alone (8). Selection actions are 0/1 by
construction, so `model.actor.bernoulli_action_dims` must be exactly the selection block's
indices — the config refuses to load otherwise.

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
| `force_axes` | list[int] | `[1, 1, 1, 1, 1, 1]` | length-6 binary mask of the axes the force branch may take — this is what makes hybrid control 3-D (`[1,1,1,0,0,0]`), 6-D, or z-only. The selection, force-target and K_f blocks are each as wide as its sum; an axis outside it is always position-controlled |
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
| `efficient_reset.enabled` | bool | `False` | reset a finished env by teleporting it onto another env's fresh state. **Training only** — `scripts/eval.py` refuses it, because the accounting needs independently sampled conditions |
| `contact.enabled` | bool | `False` | mount a contact sensor on the held asset and publish per-axis in-contact flags. Needs `task.cfg.scene.clone_in_fabric: false` (a contact reporter needs real per-env prims) |
| `contact.force_threshold` | float | `1.0` | \|f\| above this on an end-effector axis counts as contact, in N |
| `contact.append_to_policy_obs` | bool | `False` | append the 3 flags to the policy observation (grows the observation space) |
| `contact.append_to_critic_state` | bool | `False` | append the 3 flags to the critic state (asymmetric tasks only) |
| `contact.held_prim_expr` | str | `"/World/envs/env_.*/HeldAsset"` | prim path of the sensor's asset root |
| `contact.fixed_prim_expr` | str | `"/World/envs/env_.*/FixedAsset"` | prim path the contact is filtered against |
| `orientation.mode` | str | `"quat"` | `quat` (the env's own `(w, x, y, z)`) or `6d_rot_mat` (first two columns of R, Zhou et al. 2019 — continuous, no double cover) |

## derived and meta

Written by the pipeline into `resolved_config.yaml`; never set them in a config.

| key | what it is |
| --- | --- |
| `derived.run_names` | `[f"{wandb.group}_a{i}" for i in range(experiment.num_agents)]` |
| `meta.pkg_commit` | git commit of the package that ran |
| `meta.project_commit` | git commit of the working directory's repo, or null |
| `meta.created` | UTC timestamp of the run |

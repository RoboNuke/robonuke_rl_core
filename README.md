# robonuke_rl_core

End-to-end components for RL research on Isaac Lab: model definitions, skrl integration, a
training pipeline, an evaluation pipeline, configuration and experiment tracking, and modular
env definitions.

Several agents train in parallel in one Isaac Sim instance on one GPU. Agent `i` owns a
contiguous block of envs, every network holds all agents' parameters with a leading
`num_agents` dimension, and nothing computed from agent `j`'s data touches agent `i`'s update.

Built area by area. Today: the config manager (`robonuke_rl_core/config.py`), the learners
(`learners/`), the models (`models/`), the memory (`memory/`) and the auxiliary losses
(`losses/`).

## Running

```bash
python scripts/train.py --config examples/forge_exp.yaml --headless \
    task.cfg.scene.num_envs=128 experiment.seed=3

python scripts/train.py --from_run runs/forge_pih/fgain_k100 --headless   # rerun a past run
```

Exactly one of `--config` / `--from_run` is required. Every leftover argument is a dotted-path
override; values parse as YAML (`x=null`, `x=[1,2]`, `x={a: 1}`), and floats need a decimal
point (`1.0e-4`, not `1e-4`).

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

## trainer

| field | type | default | what it does |
| --- | --- | --- | --- |
| `learner` | str | required | which learner runs: `sac`, `ppo` or `flash_sac`. Only that section is used; all are dumped |
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
| `entropy_loss_form` | str | `"log_alpha"` | `log_alpha` (skrl's form) or `alpha` (FlashSAC's gentler form) |
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
| `kl_threshold` | float | `0.0` | per-agent KL early stop: an agent above this stops updating its policy for the rest of the epoch while the others continue (0 disables) |
| `time_limit_bootstrap` | bool | `False` | add `discount_factor * V(next)` to the reward on truncation |
| `normalize_values` | bool | `True` | per-agent running normalization of values and returns |
| `value_update_ratio` | int | `1` | extra value-only updates per minibatch after the combined update (1 = none) |

## flash_sac

FlashSAC: every `sac` field above, plus a distributional critic (`model.critic.n_atoms`,
`v_min`, `v_max`), weight normalization, and noise repetition.

| field | type | default | what it does |
| --- | --- | --- | --- |
| *(every field of `sac`)* | | | same meaning as above |
| `actor_update_period` | int | `1` | gradient steps between actor updates; the critic updates every step |
| `weight_norm_enabled` | bool | `True` | project each weight row onto the unit sphere after every step |
| `noise_repeat_enabled` | bool | `True` | hold a rollout noise draw for a Zeta-sampled number of steps |
| `noise_repeat_zeta_mu` | float | `2.0` | Zeta exponent for the hold length (larger = shorter holds) |
| `noise_repeat_max` | int | `16` | longest hold, in steps |
| `reward_scaling_enabled` | bool | `False` | adaptive reward scaling from the per-agent running discounted-return variance |
| `reward_scaling_g_max` | float | `5.0` | scaling cap; must equal `model.critic.v_max` so scaled returns land inside the categorical support |
| `reward_scaling_eps` | float | `1.0e-8` | epsilon in the scaling denominator |
| `target_entropy_mode` | str | `"neg_action_dim"` | `neg_action_dim` (SAC's `-|A|`) or `unified` (FlashSAC's `0.5*|A|*log(2*pi*e*sigma^2)`); ignored when `target_entropy` is set |
| `entropy_sigma_target` | float | `0.15` | sigma of the `unified` target entropy |

## model

The networks. `actor_*` sizes feed the policy, `critic_*` the critics; `n_atoms` / `v_min` /
`v_max` are FlashSAC's categorical support and are ignored by the SimBa critics.

| field | type | default | what it does |
| --- | --- | --- | --- |
| `actor.actor_n` | int | `2` | residual blocks in the actor backbone |
| `actor.actor_latent` | int | `512` | actor hidden width |
| `actor.act_init_std` | float | `0.60653066` | initial action standard deviation |
| `actor.second_act_init_std` | float or null | `null` | a second initial std, for the dims in `second_act_init_std_dims` |
| `actor.second_act_init_std_dims` | list[int] or null | `null` | action dims that start at `second_act_init_std` |
| `actor.last_layer_scale` | float | `1.0` | scale on the action-mean output weights at init (FlashSAC applies it at forward time) |
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
| `critic.critic_output_init_mean` | float | `0.0` | initial critic output bias (SimBa critics) |
| `critic.clip_actions` | bool | `False` | clip actions to the action space inside the critic |
| `critic.n_atoms` | int | `101` | FlashSAC: atoms in the categorical value distribution (>= 2) |
| `critic.v_min` | float | `-5.0` | FlashSAC: lowest atom; must be below `v_max` |
| `critic.v_max` | float | `5.0` | FlashSAC: highest atom |

## memory

| field | type | default | what it does |
| --- | --- | --- | --- |
| `memory_size` | int | `1000000` | SAC replay capacity in transitions **per agent**. PPO's buffer is sized by `ppo.rollouts` instead |

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
    - name: action_l2
      target: policy
      weight: 0.1
```

### Built-in losses

| name | targets | kwargs | what it does |
| --- | --- | --- | --- |
| `action_l2` | `policy` | *(none)* | mean squared action magnitude per agent, a regularizer toward smaller actions |

Projects add their own with `@register_loss` before loading the config; see the "Add a loss"
checklist in `CLAUDE.md`.

## derived and meta

Written by the pipeline into `resolved_config.yaml`; never set them in a config.

| key | what it is |
| --- | --- |
| `derived.run_names` | `[f"{wandb.group}_a{i}" for i in range(experiment.num_agents)]` |
| `meta.pkg_commit` | git commit of the package that ran |
| `meta.project_commit` | git commit of the working directory's repo, or null |
| `meta.created` | UTC timestamp of the run |

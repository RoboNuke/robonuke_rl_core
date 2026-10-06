# robonuke_rl_core — Claude context

Shared RL package for Isaac Lab research: skrl agents, several independent agents trained in
parallel in one Isaac Sim instance on one GPU. Built area by area. **Today the config manager
(`robonuke_rl_core/config.py`), the learners (`learners/`), the models (`models/`), the memory
(`memory/`), the logging layer (`logging.py`), eval + recording (`evaluation.py`,
`recording.py`) and the envs (`envs/`: the unified controller and the Forge wrappers)
exist.** `scripts/`: `train.py`, `eval.py`, `debug.py`.

Test env: the `general` conda env (`/home/hunter/miniconda3/envs/general/bin/python`). Never
create a new conda env for this project.

## Rules

* Fail fast and loud. No silent fallbacks. Errors say what is wrong and which file or CLI
  argument caused it.
* Use OmegaConf (installed with Isaac Lab). Do not use Hydra.
* No OmegaConf interpolation (`${...}`) in configs (a layer that uses it raises).
* One seed: `experiment.seed` is also written to `task.cfg.seed`; no layer may set
  `task.cfg.seed` itself.
* `config.py` must import without Isaac Lab installed. Isaac Lab is imported only inside
  `load_task_cfg`, `task_cfg_to_dict` and `apply_task_cfg`.
* Do not commit or push without approval. Report tests as passed / failed / total.
* **Every config field is documented.** Adding, renaming or removing a field in any section
  means updating its row in the README's configuration reference in the same change. No test
  checks this. The README never documents `task.cfg` fields — it says once that any env cfg
  field can be overridden under `task.cfg.*` and points here for the ones that cannot.

## How config works

Load order: class defaults and task defaults → most-base file → … → passed file → CLI.

`load_config(path, overrides)`:

1. Follow `base` from the passed file (each `base` is relative to the file that names it);
   raise on a missing file or a cycle, printing the chain; drop each file's `base` key.
2. Check every leftover CLI arg is `a.b.c=value` and build that layer with `from_dotlist`.
3. Per layer: reject `${...}` interpolation and `task.cfg.seed`, and record every
   `task.cfg` path the layer sets (`Config.task_overrides`). The last layer that sets
   `task.name` wins (CLI included); raise if none does. `model.architecture` is picked the
   same way (default `simba`); it selects which registered dataclass builds the `model`
   node in step 4, so the YAML keys stay `model.actor.*` / `model.critic.*` while the
   schema behind them is the chosen architecture's.
4. Build the root: one `OmegaConf.structured` node per registered section plus
   `task: {name, cfg}` from the task's env cfg defaults, then `set_struct(True)` so unknown
   keys raise at every level.
5. Merge each layer in order; an OmegaConf error is re-raised with the layer name added.
6. `OmegaConf.missing_keys` — raise listing every required field no layer set. Then write
   `experiment.seed` into `task.cfg.seed` (raise if the env cfg has no `seed` field).
7. Walk the task dict against the defaults' types (OmegaConf does not type it): int is fine
   for float **and is converted to one** (Isaac Lab's `from_dict` needs
   `isinstance(value, type(current))`), bool is not an int or float, a `None` default accepts
   anything. The checked copy is what the env gets, and it goes back into `root` so the dump
   matches.
8. Build the objects: `to_object` per section, `apply_task_cfg` for the env cfg.
9. Derive `run_names = [f"{group}_a{i}"]`.
10. Call `validate(cfg)` on each section object that defines it.
11. Return a `Config` with `task_name`, `task_cfg`, the sections by name, `derived`, `meta`,
    the merged `root` (kept for `dump`), and `task_overrides`.

Nothing may change the config after `load_config` returns.

`dump(cfg, path, env_cfg)` runs **after the env exists**, with `env_cfg = env.unwrapped.cfg`.
It first calls `check_env_kept_overrides`, which raises if the env changed any `task.cfg`
value a file or the CLI set (some envs rewrite their cfg in `__init__`; see below). Then it
writes `resolved_config.yaml`: `meta` (`pkg_commit`, `project_commit` or null, timestamp),
`derived`, then every value, with `task.cfg` taken from the live env cfg so the file holds
what actually ran. `load_from_run(run_dir, overrides)` uses that file minus `meta`,
`derived` and `task.cfg.seed` as the only file layer, then steps 2–11.

## Task cfg fields an override cannot change

* **Fields the env rewrites in `__init__`.** Factory and Forge (`FactoryEnv.__init__`)
  recompute `observation_space` and `state_space` from `obs_order`, `state_order` and
  `action_space`. Override the source instead: `obs_order`, `state_order`. `dump` raises if an
  override was discarded.
* **Spaces must be written in Isaac Lab's serialized form.** `observation_space`,
  `state_space` and `action_space` leave the env cfg object as JSON strings
  (`'{"type": "python", "space": "Box", "value": 21}'`), so a config layer must set that
  string, not an int — an int raises in step 7. This is acceptable because the env computes
  these fields anyway (Factory and Forge recompute `observation_space` and `state_space` in
  `__init__`), so overriding them never takes effect: set `obs_order` / `state_order` instead.
* **New keys in dict-valued fields** (e.g. `init_state.joint_pos`): struct mode allows
  overriding existing keys only.
* **Module constants and code** (e.g. Factory's `OBS_DIM_CFG`, `STATE_DIM_CFG`).
* **Values a task cfg computes in `__post_init__`** from other fields are not recomputed when
  you override those other fields. Forge and Factory have no `__post_init__`; check any new
  task before relying on such an override.
* **Setting a `None`-default field to a nested cfg** (e.g. a noise model) is untested; add a
  GPU test before relying on it.

## Learners

Words: a **learner** is the RL algorithm (`sac`, `ppo`); an **agent** is one of
`experiment.num_agents` independent policies trained in parallel; a **block network** holds
every agent's parameters with a leading `num_agents` dimension so one forward pass serves all
of them. Agent `i` owns envs `[i*envs_per_agent, (i+1)*envs_per_agent)`.

### The independence rule

**Nothing computed from agent `j`'s data may change agent `i`'s update.** That covers
normalizers, gradient clipping, KL early stop, advantage normalization, LR rules and memory
sampling: each is per agent. Two consequences that are easy to undo by accident:

* **Reductions keep a constant denominator.** A masked loss is
  `(keep * per_agent_mean).sum() / num_agents`, never a mean over the kept agents' rows: with
  a mean over kept rows, dropping one agent rescales every other agent's gradient by
  `num_agents / kept`. Adam does not wash that out — `eps`, stale moments, decoupled weight
  decay and the clipping threshold all see the change. With every agent kept, the fixed
  denominator is exactly the plain mean, so nothing changes in the common case.
* **No data-driven learning rate.** `BlockAdamW` could hold one LR per agent, but the
  schedule is deliberately a function of the global update count only (`lr_at`): a KL-adaptive
  LR would couple the agents. Only `constant` and `cosine` exist.
* **A dropped agent is frozen, policy and critic.** PPO's KL early stop masks the agent's
  policy loss *and* passes the keep mask to both optimizers: `BlockAdamW.step(keep)` computes
  the update and writes it only where `keep` is true, so a frozen agent comes out
  bit-identical — weights, both Adam moments, and its own step count. When **every** agent is
  dropped the epoch `break`s out; with the whole agent frozen that skips only masked work —
  almost: a break in epoch 0 also skips the remaining minibatches' observation-normalizer
  stat updates (stats train in epoch 0 only) and their logging, for every agent alike. That
  residue is collective, rare and accepted. (SpinningUp-style "the critic keeps training" is
  a one-argument change: pass `keep=None` to the value optimizer.)
* **A time limit is not a terminal state.** The value bootstrap drops `gamma*V(next)` only
  where `terminated & ~truncated`; a timeout keeps it. This is not pedantry: Isaac Lab's
  Factory and Forge `_get_dones` returns one `time_out` tensor for *both* flags, so reading
  `terminated` alone would cut the bootstrap on every episode, since those tasks end only on
  the clock. SAC stores `truncated` beside `terminated` for exactly this. PPO instead adds
  `gamma*V(next)` to the reward when `truncated` is set (`ppo.time_limit_bootstrap`) and
  treats any done as the end of the GAE chain, which comes to the same thing — but with
  `time_limit_bootstrap: False` on these tasks, PPO does cut the bootstrap at every timeout,
  so set it True there. Eval labels outcomes by the same rule (`terminal` vs `timeout`).
* **Loading one agent touches one slot.** Every per-agent tensor (including SAC's target
  critics) is saved and loaded per slot; nothing on load may write a whole block.

Test it the way `tests/learners/test_independence.py` does, which is the template for any new
learner: fill the memory, `copy.deepcopy` the learner, make agent 1's data extreme in the copy
(rewards x1e6, observations x1e3), run a few updates on both, and require agents 0 and 2 to be
bit-identical (`torch.equal`) in weights, optimizer moments, normalizer stats and learner
extras. Run more than one update: Adam's first step is sign-only, so a single step cannot show
that a gradient's magnitude changed. Make sure each case really runs the path it names: the
test checks the *weights* changed (normalizer stats alone do not count), and the KL cases
assert from `ppo/kept` that agent 1 was dropped in the copy but not in the control.

### The optimizer

`BlockAdamW` (`robonuke_rl_core/optim.py`) is the only optimizer. AdamW's update is
elementwise, so N independent `torch.optim.AdamW` instances are the same thing as one update
rule over parameters with a leading agent dimension, plus:

* a **per-agent step count** `t`, so the bias correction is right for an agent that was frozen
  for part of training;
* a **per-agent learning rate** (`set_lr` takes a scalar or an `(N,)` tensor);
* a **keep mask**: `step(keep)` writes the update only where `keep` is true, leaving a frozen
  agent bit-identical.

Every parameter must have a leading dimension of `num_agents` or the constructor raises, so a
shared parameter cannot silently couple the agents. There are no param groups, no amsgrad and
no foreach/fused plumbing — add a feature with a test when something needs it.

`tests/optim/test_block_adamw.py` is the guard: it runs N plain models with N
`torch.optim.AdamW` instances beside the stacked version for 50 steps and requires a match at
`rtol=1e-6` (float32) and `1e-12` (float64), with and without an agent frozen for part of the
run. If it fails, the optimizer is wrong — fix it rather than loosening the tolerance.

The LR schedule is `lr_at(update, total_updates, lr, lr_end, schedule)`: `constant` or
`cosine`, from the update count alone.

### Add a learner

1. Subclass `LearnerBase` in `robonuke_rl_core/learners/<name>.py`. Implement
   `_create_memory_tensors`, `act`, `record_transition`, `_update_if_ready`, `update`, and the
   checkpoint hooks (`_checkpoint_model_keys`, `_checkpoint_optimizer_keys`,
   `_checkpoint_normalizers`, `_checkpoint_extras` / `_load_extras`).
2. Add its config dataclass to `learners/cfg.py` and register the section at the bottom of
   `config.py`. Add the name to `trainer.learner`'s allowed values (`learners/cfg.py:LEARNERS`).
3. Add a builder to `MODEL_BUILDERS` in `models/factory.py`.
4. Copy the independence test for it (one line in the parametrize list) and add it to
   `tests/learners/GPU/test_train_smoke.py`'s `LEARNERS`.

### Add a model

1. Write an ordinary single-agent `nn.Module` in `robonuke_rl_core/models/`. Normal PyTorch:
   `nn.Linear`, `nn.LayerNorm`, whatever you like. **No block layers, no einsum, no per-agent
   `ParameterList`** — those are gone.
2. Two rules, because the module runs under `torch.vmap`: **no sampling and no in-place buffer
   mutation inside `forward`**. Return distribution parameters and build the distribution
   outside (that is what the wrappers in `models/simba.py` do). A vmap "falling back to a
   for-loop" warning is a failure, not a nuisance — `VmapEnsemble.forward` raises on it.
3. Wrap it: `VmapEnsemble(build_fn, num_agents, device=...)` builds N copies under sequential
   RNG (so the inits differ), stacks them with `stack_module_state`, and registers the stacked
   tensors as parameters. Everything else then works unchanged — `BlockAdamW`,
   `clip_grad_norm_per_agent`, and the per-agent checkpoint slices — because every stacked
   parameter has a leading agent dimension.
4. Add a builder to `MODEL_BUILDERS` in `models/factory.py`, keyed by
   `(architecture, learner)`. Critics take `state_space` when it is not None (asymmetric
   actor-critic).
5. Give the architecture **its own section class**: `<Arch>ModelCfg` in `models/cfg.py` —
   an `architecture` field whose default is the architecture's name, plus its own actor /
   critic dataclasses, with the paper linked in the docstring — registered with
   `register_architecture("<arch>", <Arch>ModelCfg)`. Experiments select it with
   `model.architecture: <arch>`; the YAML keys stay `model.actor.*` / `model.critic.*`,
   and struct mode rejects another architecture's fields. SimBa is the default
   (`SimbaModelCfg`, https://arxiv.org/abs/2410.09754). Add the README rows under a
   "Fields for `architecture: <arch>`" table, and a config test like
   `tests/config/test_model_architecture.py`'s swap test.
6. Copy the tests: the slot-vs-plain-module equivalence in `tests/models/test_ensemble.py`
   (same outputs **and** same gradients), plus tests 1-3 of `tests/models/test_factory.py`
   (stacked-only parameters, independence on agent 1's rows, checkpoint slicing).

### The hybrid selection: two distribution styles

A policy with a controller selection block emits one Bernoulli bit per force-eligible axis
alongside the continuous dims. `model.actor.selection_distribution` says how those form **one**
distribution, and both styles live in the same `EnsembleActor`:

* **`product`** (the default, and what every run before MATCH used) — independent:
  `log p = Σ_i log p_cont(a_i) + Σ_k log p_bern(s_k)`, entropy per dim. Its expression is kept
  exactly as it was, because re-ordering the float summation moves the result by ~2e-6 and
  `tests/models/test_match.py`'s first test asserts it with `torch.equal`. Do not refactor the
  two styles into one path.
* **`match`** — conditional (MATCH): each bit `s_k` names a `(pose_k, force_k)` pair of
  continuous dims and only the **selected** member is a random variable, because the other is a
  controller input the env ignores on that axis:

  `log p = Σ_{i free} log p_i + Σ_k [ s_k log p_{force_k} + (1 - s_k) log p_{pose_k} ] + Σ_k log p_bern(s_k)`

  and, since a mixture is not a per-dim quantity, `get_entropy` returns one column:

  `H = Σ_{i free} H_i + Σ_k [ (1 - p_k) H_{pose_k} + p_k H_{force_k} ] + Σ_k H_bern(s_k)`

**A bit of 1 is `S = 1`, which is force control**, so the force member is the live one there —
the same convention as the controller, which is why `p_k` is directly "how likely is this axis
to be force-controlled" and `outputs["selection_prob"]` is what both the supervised loss and
the logs read, with no complement anywhere
(`models/simba.py`, `SELECTION_BIT_IS_FORCE`).
`tests/models/test_match.py::test_the_gate_agrees_with_the_controller_s_selection_matrix` pins
the actor against `ActionInterface.split` itself rather than a hand-written expectation, so an
inverted sign anywhere in the chain fails there.

`selection_init_bias` lands on the selection logit **as written**: `-2.2` is ~10% force, i.e.
~90% position-dominant at init, which is what an experiment wants.

**The pair indices are derived, never configured.** `ActionLayout.pos_component_indices` is
`[pose_slice.start + axis for axis in force_axes]` (the pose block is the env's own action
vector, so an axis index *is* its offset) and `force_component_indices` is the force block,
which is packed in `force_axes` order. `models/factory.actor_kwargs` passes both, plus
`selection_axis_names` for the logs, whenever the controller has a selection block; without one,
`match` raises for want of pairs. An experiment that hand-wrote them could shift one index and
gate the wrong axis in silence.

### Add a loss

1. Subclass `AuxLoss` in the project, set `name` and `supported_targets`, implement
   `compute(ctx)`, and decorate it with `@register_loss`, at import time and before the config
   is loaded. The package ships **one** loss, `supervised_selection`, and the bar it had to
   clear is the rule: a term that could be written as a reward (a penalty on action magnitude,
   say) belongs in the env's reward, not in a policy-side term. That one cannot — it supervises
   a head of the network against a per-transition fact.
2. `compute` returns **one raw value per agent**, shape `(num_agents,)`. Reshape a flat
   `(num_agents * rows, ...)` tensor with `.view(ctx.learner.num_agents, -1)` first.
   **Never average across agents inside `compute`** — `build_aux_losses` applies the fixed
   `1/num_agents` weight, and an average inside the loss would make one agent's data scale
   another's gradient.
3. **Needs something stored per transition?** Override `required_memory_keys()` to return
   `{name: width}`. `build_aux_losses` merges them onto the hook as `memory_keys`; the learner
   creates each as a float tensor in `_create_memory_tensors`, fills it from the top-level
   `infos[name]` in `record_transition` (`aux_memory_values` checks the shape and raises), and
   it comes back in `ctx.sampled[name]`. The read is at `init()` time, so the hook must be in
   `learner.aux_loss` **before** the trainer calls `init()` — `robonuke_rl_core/train.py`
   appends it there; a hook added later gets no tensor and the loss raises by name through
   `AuxLoss.sampled`.
4. Add a term to the experiment YAML (`losses.terms`), with `kwargs` for the constructor. No
   new config fields are needed, so nothing to add to the README except a row in its built-in
   loss table for a package loss.
5. Copy tests 7-8 of `tests/losses/test_losses.py`: the weighted total and the per-agent
   values, and independence under a change to agent 1's data. A loss with memory keys also
   needs the end-to-end pass of `tests/learners/test_match_integration.py`: the tensor is
   created, filled, sampled, and the update moves the weights.

### Memory

One self-contained class (`MultiRandomMemory`) — no skrl base. Every registered name is one
tensor of shape `(num_agents, capacity, dim)`, so an agent's transitions are a contiguous slab
and nothing it samples can come from another agent's rows.

* **Capacity** is transitions **per agent**, used exactly as given, for any positive integer
  (no multiple-of rule). SAC passes `capacity=memory.memory_size`; PPO passes
  `capacity=ppo.rollouts * envs_per_agent` and fills it exactly each rollout.
* **Writing**: `add_samples(**tensors)` takes one env-step as `(num_envs, dim)` tensors in env
  order and advances one shared pointer by `envs_per_agent` rows (every agent writes the same
  number of rows per step, so one pointer is enough). When full it wraps, overwriting the
  oldest rows uniformly per agent, mid-step if the capacity is not divisible.
* **Row order**: row `step * envs_per_agent + env` within an agent's slab. `time_view(name)`
  is the `(num_agents, steps, envs_per_agent, dim)` view this implies and is how PPO gets the
  time axis GAE needs; it raises unless the capacity divides by the envs per agent.
* **Sampling** is batched, never a Python loop over envs: `sample(batch_size=B)` draws
  `torch.randint((num_agents, B))` and gathers, returning `B * num_agents` rows as
  `[agent 0 | agent 1 | ...]` for the vmap reshape; `sample_all` permutes with one
  `argsort(rand(num_agents, rows))` and splits into equal mini-batches, so every row is used
  exactly once per pass and `mini_batches` must divide the stored rows. Draw shapes never
  depend on the data, so the shared RNG stream cannot couple the agents.
* Float tensors start as NaN: a row sampled before it is written shows up as a NaN loss instead
  of a plausible zero.

### Emit a metric

Two env channels, both `(num_envs,)` tensors, both forwarded unchanged — no aggregation, no
renaming. A wrong shape or a non-dict raises, naming the key; a missing key forwards nothing.

* **Every step:** `infos["metrics_to_log"]`. The learner slices it to each agent's envs and
  emits the `(envs_per_agent,)` slice.
* **End of episode:** `infos["episode_metrics_to_log"]`. The learner masks the slice with that
  step's `terminated | truncated` and emits the compacted `(k,)` tensor of the envs that
  finished; an agent with none emits nothing, so an interval with no episodes publishes no
  point (a gap, not a zero). Slots of envs that did not finish are never read, so an env may
  leave anything there — there is no NaN convention.
* **Learner side:** build a `(num_agents,)` tensor and call `emit_per_agent({name: value},
  step)`. The name is free. `episode/return` and `episode/length` ride the episode channel as
  the episodes end; `episode/count` comes once per write interval.

The hybrid selection is logged from both ends, on purpose: `LearnerBase.emit_selection` (called
from each learner's `act`) publishes `selection/p_force_<axis>`, the policy's own probability
that the axis is force-controlled, averaged over the agent's envs; the controller wrapper
publishes `selection/force_<axis>`, the fraction of envs whose axis was force-controlled that step.
Both are "how much force control", because the selection bit is 1 for force everywhere; the
probability is the smoother of the two and is what the supervised selection loss moves, so it
is the one to watch when asking whether a policy is learning *when* to push. The axis names
come from the model, which the factory names from the controller's force-eligible axes.

The episode channel is the only host sync in the rollout path (compacting needs the count on
the host), it happens once per step, and only while an `on_log` hook is attached.

### What gets logged

Every metric below is **per agent** and reaches wandb through the two env channels or
`emit_per_agent`. Names in `DISTRIBUTIONS` (`robonuke_rl_core/logging.py`) also publish
`<name>/min` and `<name>/max`, because the spread across envs is often the finding.

**A NaN means "not applicable", never zero.** An episode that did not break has no break
step; one that never succeeded has no time-to-success. The accumulator skips NaN entries in
the sum, the count and the extremes, so a metric only some episodes have still averages over
the episodes that have it.

| metric | emitted by | notes |
| --- | --- | --- |
| `episode/return`, `episode/length` | `LearnerBase` | + `/min`, `/max`; one value per finished episode |
| `episode/count` | `LearnerBase` | episodes finished in the interval |
| `episode/best_success_rate` | `LearnerBase` | **training only**: the success rate measured over the interval that produced the current `ckpt_best`, republished every interval so the line is flat between improvements — "how good is the policy I would actually ship". It **can go down**: best is chosen by mean episode return, so a new best-by-return may have had a worse success rate, and forcing it upward would report a number no checkpoint on disk ever achieved. Nothing is published until an interval with episodes has been seen |
| `reward/step` | `LearnerBase` | the instantaneous reward, per env, every step; + `/min`, `/max` |
| `loss/policy`, `loss/critic`, `loss/value`, `loss/entropy` | the learner | SAC's `loss/entropy` is the temperature's loss, PPO's is the entropy bonus's contribution |
| `policy/std`, `policy/entropy`, `policy/log_prob` | the learner | `policy/std` is the mean commanded sigma |
| `entropy/coefficient` | SAC | alpha |
| `q/q1_mean`, `q/q2_mean`, `q/target_mean` | SAC | + `/min`, `/max` |
| `ppo/kl`, `ppo/clip_fraction`, `ppo/kept` | PPO | `ppo/kept` is the per-agent KL early stop |
| `grad_norm/*`, `lr/*` | the learner | per optimizer |
| `stats/update_time_ms` | the learner | wall time of one update, the number to compare against another implementation. Emitted from inside the timed block, so it **lags one update** and the first point is 0 |
| `stats/ram_mb` | the learner | this process's resident set size, from `/proc/self/statm` — it counts what Kit and PhysX hold too, so it is the number that predicts an OOM kill. + `/min`, `/max` |
| `stats/gpu_used_mb` | the learner | whole-device memory in use (`mem_get_info`: total - free), so it includes Isaac Sim. CUDA only. + `/min`, `/max` |
| `stats/gpu_torch_reserved_mb` | the learner | our allocator's share of the above; a growing gap between the two says the leak is not in the learner. CUDA only |
| `loss/<name>_<target>` | `losses.build_aux_losses` | one per configured aux loss |
| `episode/success`, `episode/success_step`, `episode/engaged` | `ForgeTaskMetricsWrapper` | `success_step` is NaN unless the episode succeeded. `engaged` is **latched** over the episode, like the env's own `ep_succeeded`: it is the only quantity here read from live scene state, and `_reset_idx` has already restored the initial condition by the time the wrapper's `step` runs, so a live reading was 0 for every episode that ended. Latched, the engagement rate is always >= the success rate, as it must be |
| `termination/success`, `termination/timeout` | `ForgeTaskMetricsWrapper` | one 0/1 per cause, so each averages into a rate. `timeout` is the **clock** (`truncated & ~success`), not "any done that was not a success" — a broken peg ends an episode through `terminated`, and counting it as a timeout made the two namespaces disagree. With `success` + `timeout` + `fragile/broke` every episode is accounted for |
| `reward/<term>` | `ForgeTaskMetricsWrapper` | the per-term reward decomposition, per step |
| `Success_Prediction/error`, `/<thr>_precision`, `/<thr>_recall`, `/<thr>_delay_all`, `/<thr>_delay_correct` | `ForgeTaskMetricsWrapper` | how good Forge's 7th action is |
| `fragile/force` | `ForgeFragileObjectWrapper` | per step |
| `fragile/broke` | `ForgeFragileObjectWrapper` | the break rate over the episodes that ended |
| `fragile/rate_cause_{force,normal,shear,contact_loss}` | `ForgeFragileObjectWrapper` | the fraction of ended episodes whose break had this cause, so a value under 1 reads as the rate it is. **The set depends on the mode**: magnitude mode (`break_force: [n]`) publishes `rate_cause_force` only; `direction_break_force` publishes `rate_cause_normal` and `rate_cause_shear` instead, never `force`. `rate_cause_contact_loss` appears only with `require_contact` |
| `fragile/break_step` | `ForgeFragileObjectWrapper` | the episode step the break happened on, NaN where nothing broke. Sampled in `_get_dones`, because `_reset_idx` zeroes `episode_length_buf` for exactly the envs that broke before the wrapper's `step` sees them |
| `contact/in_contact_{x,y,z,any}` | `ForgeContactSensorWrapper` | per step |
| `selection/force_<axis>` | `ForgeControllerWrapper` | per step; the fraction of envs whose axis was force-controlled (1 = force) |
| `selection/p_force_<axis>` | `LearnerBase.emit_selection` | the policy's probability of force control on that axis |

**Each metric is emitted where it is decided.** A break rate comes from the wrapper that
decides breaks; a KL early stop from the learner that applies it; the task's outcomes from the
wrapper that taps the env's own hooks. Nothing re-derives another component's number.

### Logging

`logging.py` holds the whole logging layer; the learners only route.

* `MetricAccumulator` keeps per `(agent, name)` sums on the device and counts on the host
  between flushes, so no metric costs a GPU->CPU sync when it is emitted. `flush()` converts
  the whole interval in one `tolist()` and returns the mean per agent, skipping any name whose
  count is zero.
* `WandbLogger` owns one wandb run per agent (`derived.run_names`, grouped by `wandb.group`,
  with the resolved config attached) and is itself the `on_log` hook; `flush` is the `on_flush`
  hook, `checkpoint` is the `on_checkpoint` hook and `close()` finishes the runs.
* **Files go up as plain run files, never as Artifacts.** `resolved_config.yaml` once and each
  checkpoint as it is written, stored under their base names (`run.save(..., policy="now")`),
  because eval downloads them back by name. An eval launched with `--run` pushes its own
  outputs back to the *same* training run under `eval/<eval-config-stem>_<timestamp>/`, and
  logs no metrics: the parquet trace is the data. `wandb` is imported lazily, so the package imports
  without it. Runs are created with `reinit="create_new"` and logged through their own
  `run.log(data, step=...)` — never the global `wandb.log`, which with several live runs would
  publish to whichever started last. A wandb older than `MIN_WANDB_VERSION` cannot keep N runs
  alive, so the logger raises instead of merging the agents into one run.
* The x-axis of every point is the global env timestep, and the publish cadence is the
  learner's `trainer.write_interval` (one flush point, in `post_interaction`).
* `wandb.mode` is `online`, `offline` or `disabled`; tests and debug runs use `disabled`.

### Hooks

Both lists live on the learner and are **empty by default** (nothing is logged, no extra loss):

* `on_log`: `fn(agent_idx: int, metrics: dict[str, Tensor], step: int)`. Per-step env metrics
  arrive as `(envs_per_agent,)` tensors, episode metrics as compacted `(k,)` tensors, learner
  metrics as 0-d tensors.
* `on_flush`: `fn(step: int)`, called once per `trainer.write_interval` right after the episode
  flush. Where a logger publishes.
* `on_checkpoint`: `fn(agent_idx: int, step: int, path: Path)`, called for every checkpoint
  file written (periodic and best). `WandbLogger.checkpoint` is the hook that mirrors it to
  that agent's run.
* `aux_loss`: `fn(ctx: AuxLossContext) -> Tensor | None`, called for `ctx.target == "policy"`
  and `"critic"`. A returned tensor is added to that loss.

### Checkpoints

Per agent, at `{trainer.output_dir}/{wandb.project}/{wandb.group}/{run_name}/checkpoints/`:
`ckpt_{step}.ckpt` every `trainer.checkpoint_interval` steps, plus `ckpt_best.ckpt` whenever an
agent's mean episode return over a write interval improves. A file holds that agent's model
weights, optimizer state, normalizer stats, learner extras, the step and the mean return.
`load_agent(path, slot)` loads one file into any slot; optimizer state only with
`with_optimizer=True`, which needs `num_agents == 1`.

## Envs: the controller and the Forge wrappers

### One controller, many faces

`robonuke_rl_core/envs/forge/control.py` has a single torque path:

```
tau = J^T [ (I - S) (K e_pose - D v) + S K_f (f_d - f) ] + nullspace
```

**`S` selects force: `S = 1` on an axis means that axis is force-controlled, `S = 0` means
position-controlled.** That one convention holds from the policy's Bernoulli bit through the
action vector and the `selection/*` metrics to this matrix — nothing anywhere takes a
complement, and `tests/envs/test_interface.py` asserts the values rather than describing
them. The expression is linear in `S`, so the degenerate cases are exact, not
approximations: `S = 0` with `f_d = 0` *is* Factory's own controller. That is a tested claim,
not a hopeful one — `tests/envs/GPU/test_controller.py` runs our math beside Isaac Lab's
`compute_dof_torque` on live env state for 50 steps and requires agreement (measured: a
worst-case difference of **0.0**, bit for bit). Which is why pose control needs no separate
path.

**There is no mode field.** The action layout is inferred from the capabilities:

| `use_pose` | `use_force` | layout | fixed |
| --- | --- | --- | --- |
| ✓ | — | `[pose \| gains?]` | `S = 0`, `f_d = 0` |
| — | ✓ | `[force \| gains?]` | `S = I` |
| ✓ | ✓ | `[pose \| selection \| force \| gains?]` | — |

The selection block exists exactly when both branches do, because that is the only case where
the policy has a choice to make. `controller.force_axes` (a length-6 binary mask) sets how
wide the force side is, and the selection, force-target and K_f blocks are all that wide.
**It defaults to `[1,1,1,0,0,0]`, the 3-D hybrid**: force on the translation axes, orientation
always position-controlled. That is the configuration this project runs — a wrist wrench's
torque channels are its noisy ones and a regulated torque is rarely what a contact task
wants — so `[1]*6` is opt-in, and force-only control (`use_pose: false`) has to ask for it
explicitly because every axis then needs a controller. `gain_mapping` is orthogonal:
`constant` costs no action dims, `variable_diagonal` adds one per axis of each live branch and
maps it geometrically onto `[gain_min, gain_max]`. Adding a mapping is a new entry in
`ActionLayout.gain_dims`, not a rewrite.

Two invariants worth keeping:

* **The pose block is the env's own action vector**, `controller.native_action_dim` wide — 7
  on Forge (6 pose dims plus the success prediction its reward reads), 6 on Factory. It is
  handed to the env untouched, so the env's EMA, position bounds, upright constraint and
  `prev_actions` observation channel all behave exactly as they did. The wrapper replaces only
  the step that turns a target pose into joint torque.
* **Selection actions are Bernoulli by construction**, so `controller.validate` requires
  `model.actor.bernoulli_action_dims` to be exactly the selection block's indices. A shifted
  index would make a continuous action a 0/1 switch with no crash and no symptom, so it fails
  when the config loads. The actor emits them as ±1: **+1 is force**, -1 is position.

**Forge's own control defaults are inherited at runtime, not by subclassing.** `ema_factor`,
the dead zone, `pos_action_bounds`, `default_task_prop_gains`, `kp_null`/`kd_null` and
`default_dof_pos_tensor` are read from `env.unwrapped.cfg.ctrl` each step, so the defaults are
Forge's and an experiment tunes them in one place: `task.cfg.ctrl.*`. The `controller` section
holds only what Forge's ctrl cfg does not.

### The Forge rule

Forge-family envs expose internals other Isaac Lab envs do not: a smoothed wrist wrench, an
operational-space `cfg.ctrl`, fingertip state on the env object, `_reset_idx` semantics a
wrapper can lean on. So **anything that reads those lives in `robonuke_rl_core/envs/forge/`,
carries a `Forge` class-name prefix, and calls `require_forge_env` before it touches
anything** — and the rule that function enforces is the **task name**: a Forge env is one
whose name contains `"forge"`, case-insensitively. Not duck typing, not a capability probe.
A wrapper applied to another family then fails up front, naming the task, the rule and the
attributes it would have needed, instead of dying on a missing attribute mid-rollout. A
wrapper for a different env family starts a **sibling directory**, never a special case
inside `forge/`.

Env-agnostic pieces — the action-interface math (`envs/interface.py`), the 6-D rotation
(`envs/orientation.py`), the composer (`envs/build.py`) — stay directly under `envs/`.

### The wrapper order

`build_env` applies exactly what the config enables, innermost first, and the order is fixed
(`WRAPPER_ORDER`): **controller → efficient reset → fragile → contact**. The controller is
innermost because it owns the action space; efficient reset wraps `_reset_idx` and must see
the env's own reset chain; fragile wraps `_get_dones` and triggers those resets; contact is
last because it appends to the observation, so nothing that edits obs may wrap after it.

Two things run **before** `gym.make` (`prepare_task`, beside the recorder camera): the contact
sensor and the orientation rewrite, because both change what the env's spaces are sized from.
Both also need `task.cfg.scene.clone_in_fabric: false` — a contact reporter and a camera need
real per-env prims — and both say so rather than failing cryptically.

### The wrappers

* **`ForgeFragileObjectWrapper`** — breaks the held object on force magnitude, or on the
  axial/shear split measured against the peg's live axis (a thin peg shears long before it
  crushes), and optionally on loss of contact with a grace period and a debounce. Terminations
  go through `_get_dones`; `fragile/broke` rides the episode channel and `fragile/force` the
  step channel.
* **`ForgeEfficientResetWrapper`** — a full reset runs the env's chain and caches the scene;
  a partial reset runs only `DirectRLEnv._reset_idx` and teleports the finished env onto a
  random donor's cached state, then copies the bookkeeping the lightweight path skips,
  re-samples Forge's per-episode randomization and clears the force smoothing.
  **`wrappers.fragile.enabled` requires it** (`WrappersCfg.validate`), and that is not an
  optimization: a peg breaks one env at a time, so the env resets a *subset* mid-episode, and
  Factory/Forge's `randomize_initial_state` is written assuming every env resets together —
  it samples `len(env_ids)` rows, assigns them into the full buffer (which raises) and builds
  the rest at `num_envs`. Stock Factory/Forge never notices because those tasks end only on
  the clock, so every env times out on the same step. Eval uses the wrapper too: a teleported
  episode is not independently sampled, but it is never counted either (see below).
* **`ForgeContactSensorWrapper`** — per-axis in-contact flags from a real contact sensor,
  published on the step channel, on `env.in_contact` for other wrappers, and optionally
  appended to the observation. The sensor is built *after* `clone_environments`, because the
  PhysX contact API sits on an asset-specific child prim that does not exist until the stage
  is cloned.

### Orientation

`wrappers.orientation.mode: 6d_rot_mat` swaps every `*_quat` observation channel for the 6-D
rotation representation (Zhou et al. 2019 — the first two columns of R). `q` and `-q` give the
same answer, which is the point: the quaternion double cover makes the obvious representation
discontinuous. The rewrite registers the new dims, swaps `obs_order`/`state_order`, and
augments the observation dict inside `factory_utils.collapse_obs_dict` — the one function both
Factory and Forge flatten through, so neither env's observation method has to be copied into
this package.

## Eval, recording and debug

`scripts/eval.py` evaluates one trained policy under conditions an eval config names, and
`scripts/debug.py` lets you watch one. `robonuke_rl_core/evaluation.py` holds the parts worth
testing (accounting, state capture, run loading) and `robonuke_rl_core/recording.py` the
camera and video path. Both scripts take the same policy sources and config layering.

### The accounting rule

**One round = one global reset, up to `max_episode_length` steps, and each env contributes
exactly its FIRST episode of the round.** Isaac Lab keeps auto-resetting envs mid-round;
every step after an env's first done is masked out of the counts, the returns, the metrics,
the trace and the video — one mask (`EvalAccounting.step` returns it) drives all of them, so
they cannot disagree.

That mask is also why eval runs the **efficient reset** like training does. Those mid-round
resets have to happen (the env auto-resets whatever we do, and with a fragile peg they are
partial), the wrapper is what makes them safe, and the episodes they start are already
excluded: `valid = active & ~finished`. Independent initial conditions come from the global
`force_env_reset` at the top of each round, which is the only reset whose episodes count. `ceil(num_rollouts / num_envs)` rounds run, and the last one uses only
the envs still needed.

Two details that are easy to get wrong:

* **A round ends when nothing is valid, not after a fixed count.** Factory and Forge time out
  at `max_episode_length - 1`, so stepping the full budget would run one step of a fresh,
  discarded episode. `max_episode_length` is the upper bound and stepping past it raises.
* **`terminal` means `terminated & ~truncated`.** Those tasks raise both flags at the limit,
  so anything else would label every timeout a terminal outcome. Episodes keep both raw flags.
* **Rounds really reset.** skrl's `IsaacLabWrapper.reset()` is a no-op after the first call,
  so `force_env_reset` clears the `_reset_once` guard down the wrapper stack and then checks
  `episode_length_buf == 0`, failing loudly rather than collecting mid-rollout fragments.

### Where the policy comes from

`--run entity/project/<run id or name>` (the usual way) downloads `resolved_config.yaml` and
the checkpoint from that run's **plain files** into `~/.cache/robonuke_rl_core/...`, shaped
like a local run dir; `--local <run dir>` skips wandb. Either way the config for the eval is
the run's resolved config, then the `--eval_config` file (with its own `base` chain), then
CLI overrides — so the eval config sets the test conditions and the env count, and anything
it omits keeps the trained run's value.

**Both entry points inject `experiment.num_agents=1`** (`evaluation.SINGLE_AGENT_OVERRIDE`,
appended last so it wins over a CLI value too). Eval and debug load one checkpoint slot into
a one-agent shell and give it every env, so the trained run's agent count is not just unused:
left in place, `trainer.validate`'s divisibility rule rejects any eval env count that does not
divide by it (64 envs over a 3-agent run), and the eval's own `resolved_config.yaml` would
claim an agent count that never ran. `--checkpoint` takes `best` (default), a step number
or an exact file name. The policy is a 1-agent shell whatever the run trained, loaded with
`load_agent(..., with_optimizer=False)`, normalizers frozen, `mean_actions` unless
`eval.deterministic: false`.

### What an eval writes

Under `<run dir>/eval/<eval-config-stem>_<timestamp>/`: `summary.yaml` (aggregates plus one
row per episode; each metric gets a `/mean`, and a `/std` **only where it carries
information** — the std of a 0/1 column is `sqrt(p(1-p))`, so beside the mean it can only
restate it, while inviting the reading that a 0.89 success rate with a 0.31 "std" says
something about spread between episodes), `<stem>.parquet` (the per-step trace: one row per valid (round, env, step),
vector signals expanded to `name_0..k`, NaN back-fill so it stays rectangular),
`resolved_config.yaml` (the eval's own, so it reproduces like a run), and `videos/` when
recording. With `--run`, those same files go back to the training run under
`eval/<stem>_<timestamp>/`; nothing is ever logged as a wandb metric.

### Capture an env signal in eval

* Per step: `infos["metrics_to_log"]`, as in training.
* At episode end: `infos["episode_metrics_to_log"]`, masked to the envs that finished.
* Whole-state: `infos["eval_state"]`, a dict of `(num_envs, ...)` tensors — trailing dims are
  allowed here, and every element becomes its own parquet column. Shape-policed like the
  metric channels, and only read on valid steps.

### Add an overlay

Subclass `Overlay` in the project, set `name`, implement `apply(frame, step) -> frame` where
`frame` is `(H, W, 3)` uint8 and `step` is a `StepData` (env, step, reward, return so far,
the done flags, that env's metric values), decorate with `@register_overlay` at import time,
and list the name in `eval.overlays`. The package ships `hud`; an unknown name raises when
the config loads, not when the camera starts rolling.

### Debug the env or the policy

`scripts/debug.py` runs **one** env (it forces `task.cfg.scene.num_envs=1`) and collects
nothing. Same `--run` / `--local`, `--checkpoint` and optional `--eval_config` as eval.

* **Live viewer** (default, needs a window): the policy acts while you watch. `j` pauses,
  `k` resets, `l` quits.
* **Reset viewer** (`--resets`): a reset every `--hold_seconds` seconds of wall clock, `k`
  resets now, `l` quits. Between resets the sim is **rendered but never stepped**, so what you see is exactly
  the initial condition the env sampled, not a pose physics has already pulled on.
* **Headless reset video** (`--resets --headless --num_resets N --out resets.mp4`): the same
  loop through `recording.render_resets`, holding each condition for `hold_seconds` of video
  (`round(hold_seconds * video_fps)` frames). It injects
  `task.cfg.scene.clone_in_fabric=false` as a CLI-layer override, because it needs a camera
  and has no config of its own to say so; eval instead *requires* the eval config to state it,
  since recording there is a choice that config makes.

**Pausing and holding still means `sim.render()`, never `app.update()`.** `app.update()`
steps physics — Isaac Lab says so itself, in the FIXME inside `SimulationContext.step()` — so
a "paused" loop built on it keeps the robot moving. `SimulationContext.render()` is
`app.update()` wrapped in `/app/player/playSimulations = False`: the viewport stays live and
keeps taking input, and physics does not advance. That is what both debug viewers and
`render_resets` use between steps.

Keys go through the carb input interface the way Isaac Lab's own devices do
(`isaaclab/devices/keyboard/se2_keyboard.py`): `carb.input.acquire_input_interface()`, the app
window's keyboard, `subscribe_to_keyboard_events`, and `event.input.name` on `KEY_PRESS`.

**Do not bind a letter the viewport already owns.** Kit claims `p` (Parent Prim), `w/e/r`
(gizmos), `f` (frame selected) and others, consumes the event before this script's
subscription sees it, and answers with a toast — so the viewer looks like it ignored the key.
`j/k/l` are the defaults for that reason, `--key_pause` / `--key_reset` / `--key_quit`
override them, the mapping is printed when a viewer starts, and the first key press of any
kind prints a line, so "Kit ate it" and "wrong binding" are distinguishable. Note the
interface spells the teardown `unsubscribe_to_keyboard_events`; `unsubscribe_from_...` does
not exist (Isaac Lab's own device calls the missing name inside `__del__`, where Python
swallows the error).

**The keyboard paths cannot be tested automatically** — the headless reset video has a GPU
test, the rest is this checklist, worth one pass whenever `debug.py` changes:

1. `--local <run> ` opens a window, the robot moves, no files are written anywhere.
2. `r` snaps the scene back to a fresh initial condition.
3. `p` freezes the robot; the viewport still orbits and zooms; `p` again resumes.
4. `s` exits cleanly (no hang, no traceback).
5. `--resets` cycles initial conditions on the timer, each held perfectly still.
6. `r` in `--resets` jumps to the next condition immediately; `q` exits cleanly.

Recording needs the camera in the scene **before** `gym.make`
(`install_recorder_camera(task_name, task_cfg, ...)`, which also forces
`scene.clone_in_fabric=False` — a `TiledCamera` needs real per-env prims) and the app
launched with `enable_cameras=True`. One mp4 per (round, env) is written as the round runs,
so memory is one frame per env. The GPU test suite therefore carries the camera on its shared
env, which is why `pytest -m gpu` takes about three minutes rather than ninety seconds.

## HPC launch

`robonuke_rl_core/hpc/` submits package runs to SLURM + Apptainer. Three verbs, thin callers
in the project's `launchers/`:

```bash
python launchers/launch_train.py <folder | config.yaml ...> --project P --group_prefix G
python launchers/launch_sweep.py <configs> --project P --group_prefix G \
    --sweep_param sac.actor_lr --label lr --value 1.0e-4 --value 3.0e-4
python launchers/launch_eval.py --eval_config F --project P [--group G ...]
```

### Naming

wandb is the interface, so every name derives from how runs appear there. `--project` and
`--group_prefix` are **required**; `--tag` is repeatable.

| thing | value |
| --- | --- |
| wandb project | `--project` |
| wandb group | `{group_prefix}_{config_stem}`, sweeps append `_{LABEL}-{value}` |
| run names | `{group}_a{i}` (the existing `derived.run_names` rule) |
| SLURM job name | the group, exactly |
| SLURM logs | `{hpc.exp_log_dir}/{project}/{group}_%j.out` / `.err` |
| wandb tags | chain's `wandb.tags` + every `--tag` + (sweeps) `{LABEL}-{value}` |

A hyphen joins a `{LABEL}-{value}` pair because the two belong together; joins between parts
of a name stay underscores. The launcher passes the three wandb keys in as ordinary CLI
overrides, **last**, so they are the final layer — and therefore it **rejects** a user
override of `wandb.project`, `wandb.group` or `wandb.tags`, naming the flag instead. Two
sources of truth for a run's name is how runs get lost. Every computed group is checked
against the same rule `WandbCfg.validate` applies, before anything is queued.

### Three rules that shape the code

* **Import discipline.** Nothing in `robonuke_rl_core.hpc` may import torch, wandb or Isaac
  Lab at module level: the submitters run on a login node with none of it. wandb is imported
  inside `launch_eval.find_runs` only. This is why `configfile.py` exists — the `base`-chain
  reader had to come out of `config.py`, which pulls torch through the `eval` section, so both
  `load_config` and the submitter call the *same* chain functions.
  `tests/hpc/test_import_discipline.py` enforces it in a subprocess.
* **Fail before queue.** Every config in a batch is read, resolved and named before the first
  job is submitted; a bad config aborts the whole submit naming the file, the field and the
  value. After that gate a *submission* failure is reported and the batch continues.
* **Bake the stack, bind the code.** The image bakes Isaac Sim, Isaac Lab and the package's
  *dependencies* — the package itself is installed **editable** from a build-time clone at
  `/opt/robonuke_rl_core`, and the job binds the cluster's live clone over that path. So
  updating the package is `git pull`, and the image is rebuilt **only when `pyproject.toml`
  dependencies change**, the same rule as a local editable install. The project repo is bound
  and used as cwd; it is never installed.

### The job

`hpc/hpc_job.bash` ships as package data and is submitted **by path**. sbatch spools it, so
it has no siblings at runtime and can source nothing: every input arrives as an exported
`RNK_*` variable or as argv. It validates its inputs, picks a wandb mode (no key, or a key
under 40 characters, means offline — logging must never kill an unattended job), builds the
binds, and `exec`s apptainer so `--signal=TERM@300` reaches python rather than bash.

`--eval_config` on `launch_train` makes it a train-then-eval job, which cannot be one exec'd
python: the job runs `hpc_job_chain.bash` in-container instead, which runs training and then
one eval per agent, **only if training exited 0**, each **non-fatal**. Training that finished
is the expensive thing; a wandb hiccup in eval must not turn a completed run into a failed
job. The wrapper forwards TERM to the live child.

### Tests

`tests/hpc/` is CPU and runs by default: the section's rules, the chain reader, every name,
the three launchers end to end through `--dry_run` (which prints the exact `sbatch` line and
submits nothing), and `hpc_job_chain.bash` driven against a stub interpreter — so "does it
start the right processes" for a train-then-eval job is answered on CPU, with no cluster and
no container.

`tests/hpc/HPC/` carries `@pytest.mark.hpc` and needs a real cluster — run it on a login node
with `RNK_TEST_CONFIG=<a config with an hpc section> pytest -m hpc tests/hpc/HPC`. Like the
GPU marker, **nothing there skips silently**: a cluster test that cannot find its cluster
fails. It is about the **SLURM path**, not the container: whether the image can import the
stack is the build script's own `verify` step, run once at build time, so apptainer appears
there only where it does in real use — on a compute node, inside a job SLURM started. It
checks the tools and the paths, that `sbatch --test-only` accepts a submission without
queueing it, that the resource flags are the config's, and then submits **one trivial job**
and reads its log to prove our command ran, with the project root as cwd and the package
resolving to the bound clone. That last one is the only check that the bake-the-stack /
bind-the-code design actually holds; it costs a few seconds of one GPU. Nothing there records
video — the recorder camera is a separate path with its own problems, and a launcher test has
no business depending on it.

Not built, deliberately: **resume** (the package has no true mid-training resume — optimizer
state loads only at `num_agents == 1` — so a resume launcher would be a lie) and a **local
sequential runner** (`--dry_run` prints runnable commands, which covers it).

## Add a config class (package or project)

1. Write a `@dataclass` next to the code it configures. Type every field with an
   OmegaConf-supported type (primitives, `Optional`, `List`, `Dict[str, Any]`, enums, nested
   dataclasses). Use `MISSING` when every experiment must choose; otherwise give a default.
2. Callables or classes: a `str` field holding `"module:name"`, resolved with
   `resolve_callable` where the code uses it. Never a callable field.
3. New top-level section: `register_section("<name>", Cls)` — package sections at the bottom
   of `config.py`, project sections in the project's entry script, before loading. A sub-group
   of an existing area is a nested dataclass field, not a new section.
4. Cross-field rules: add `validate(self, cfg)` that raises with the dotted path and the bad
   values.
5. Add a test: defaults load, each `validate` rule fails on a bad value, the round trip passes.
6. Run all tests; report passed / failed / total.

## Experiment files

Set `base`, override only what differs. Never copy a class default into YAML. All YAML is
block style, one key per line. No `${...}` interpolation. Write floats with a decimal point
(`1.0e-4`) so a value stays a float wherever it is read.

```yaml
base: ../base/forge.yaml
experiment:
  seed: 10
wandb:
  group: fgain_k100
task:
  cfg:
    scene:
      num_envs: 256
```

CLI overrides take dotted paths, including task cfg values:

```bash
python scripts/train.py --config examples/forge_exp.yaml --headless \
    task.cfg.scene.num_envs=128 experiment.seed=3 "wandb.tags=[fgain,debug]"
```

Rerun or evaluate a past run: `--from_run <run_dir>` plus overrides. Exactly one of
`--config` / `--from_run` is required.

Entry points: the real `main(argv=None, setup=None)` of train, eval and debug lives IN the
package (`robonuke_rl_core/train.py`, `eval.py`, `debug.py`); this repo's `scripts/` are
thin callers, and a project repo writes its own thin caller the same way. The `setup` hook
runs **after** `AppLauncher` (so Isaac imports work) and **before** the config loads — it is
where a project imports its tasks (`gym.register`) and registers its sections, losses,
overlays and architectures. Everything a project registers goes inside `setup`, never at
the script's module top, because task modules import Isaac Lab:

```python
from robonuke_rl_core.train import main

def setup():
    import my_project.tasks                     # gym.register, after the app is up
    from robonuke_rl_core.config import register_section
    from my_project.cfg import RewardCfg
    register_section("reward", RewardCfg)       # project sections, before loading

if __name__ == "__main__":
    raise SystemExit(main(setup=setup))
```

Inside `main`, the order is unchanged: parse args -> `AppLauncher` -> `setup()` ->
`load_from_args` -> `prepare_task` -> `gym.make` -> `dump` (once, after the env exists) ->
`build_env` -> models/learner/logger -> train. The package install is one editable clone
shared by every project (`pip install -e`, see the README); it expects an Isaac Lab env
(0.47.1 / Isaac Sim 5.1.0, Python 3.11) and brings its other dependencies itself.

## Test layout

`tests/<module>/` holds the tests for `robonuke_rl_core/<module>.py` (or `<module>/`). CPU
tests sit directly in it; tests that need Isaac Sim or a GPU go in `tests/<module>/GPU/` and
carry `@pytest.mark.gpu` (the marker is registered in `tests/conftest.py`). Shared builders go
in a plain module (`tests/learners/helpers.py`), not in a `conftest.py`: two `conftest`
modules on the path shadow each other.

The config tests run with only the sections `config.py` itself owns
(`tests/config/conftest.py`), so a new section from another area does not make every config
fixture incomplete.

```bash
pytest          # CPU tests only (addopts = -m 'not gpu')
pytest -m gpu   # the Isaac Sim tests, on the GPU machine
```

Every GPU test shares **one** env (`gpu_env`), because Isaac Lab hangs when a second one is
created in a process. That env always carries **both** the recorder camera and the contact
sensor: each installs before `gym.make`, so the decision is made once per process, and having
both on means a single `pytest -m gpu` covers the recording, reset-video and contact paths
too. It costs roughly 10 s of render time and is why the config sets
`task.cfg.scene.clone_in_fabric=false` (a camera and a contact reporter both need real per-env
prims).

No test is skipped silently: a GPU test that cannot start Isaac Sim fails. Report each run as
passed / failed / total.

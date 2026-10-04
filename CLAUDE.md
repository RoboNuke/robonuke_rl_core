# robonuke_rl_core — Claude context

Shared RL package for Isaac Lab research: skrl agents, several independent agents trained in
parallel in one Isaac Sim instance on one GPU. Built area by area. **Today the config manager
(`robonuke_rl_core/config.py`), the learners (`learners/`), the models (`models/`) and the
memory (`memory/`) exist.** The models and the memory are provisional ports; they get their
own design pass later.

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
   `task.name` wins (CLI included); raise if none does.
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

Words: a **learner** is the RL algorithm (`sac`, `ppo`, `flash_sac`); an **agent** is one of
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
* **No data-driven learning rate.** One optimizer over block parameters holds a single LR, so
  a KL-adaptive LR would couple the agents. Only `constant` and `cosine` exist.
* **A dropped agent is frozen, and nothing else depends on who dropped.** PPO's KL early stop
  masks the agent's policy loss *and* steps the policy optimizer through
  `step_with_frozen_agents`, which puts the dropped agent's weights and Adam moments back
  after the step (otherwise momentum and weight decay keep moving it). The step is still taken
  on every minibatch, also when every agent is dropped, so the shared Adam step counter never
  depends on another agent's KL. There is no early `break`, and the critic trains on every
  minibatch for every agent.
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

1. Write the class in `robonuke_rl_core/models/`. **Every parameter must be a block parameter**
   (leading dim `num_agents`) or an entry of a per-agent `nn.ParameterList`;
   `clip_grad_norm_per_agent` raises on anything else, because a shared parameter would couple
   the agents. Prefer the block parameter: an optimizer-state tensor of shape `(1, k)` cannot
   be attributed to its agent, so a `ParameterList` entry's Adam moments cannot be sliced per
   agent (that is why the SimBa actor's `log_std` is a block parameter).
2. Add a builder to `MODEL_BUILDERS` in `models/factory.py`, keyed by learner name. Critics
   take `state_space` when it is not None (asymmetric actor-critic).
3. Add its fields to `ModelCfg` in `models/cfg.py` and their rows to the README.
4. Copy tests 1-3 of `tests/models/test_factory.py`: block-only parameters, independence
   (agent 1's rows only, with any BatchNorm in training mode), and checkpoint slicing. If a
   learner uses the model, also copy the checkpoint round trip in
   `tests/learners/test_base.py` (save agent 2 of 3, load into slot 0 of a 1-agent learner,
   compare deterministic actions).

### Add a loss

1. Subclass `AuxLoss` in the project (or in `robonuke_rl_core/losses/losses.py` for a package
   loss), set `name` and `supported_targets`, implement `compute(ctx)`, and decorate it with
   `@register_loss`. Projects register at import time, before the config is loaded.
2. `compute` returns **one raw value per agent**, shape `(num_agents,)`. Reshape a flat
   `(num_agents * rows, ...)` tensor with `.view(ctx.learner.num_agents, -1)` first.
   **Never average across agents inside `compute`** — `build_aux_losses` applies the fixed
   `1/num_agents` weight, and an average inside the loss would make one agent's data scale
   another's gradient.
3. Add a term to the experiment YAML (`losses.terms`), with `kwargs` for the constructor. No
   new config fields are needed, so nothing to add to the README except a row in its built-in
   loss table for a package loss.
4. Copy tests 7-8 of `tests/losses/test_losses.py`: the weighted total and the per-agent
   values, and independence under a change to agent 1's data.

### Memory

One memory class (`MultiRandomMemory`). Batch sizes are **per agent**: `sample(batch_size=B)`
returns `B * num_agents` rows as `[agent 0 | agent 1 | ...]`, drawn from each agent's own envs,
and `sample_all` keeps that order so a block-parallel reshape routes each agent's rows to its
own parameters. `memory.memory_size` is SAC's capacity in transitions **per agent**:
`replay_depth(memory_size, envs_per_agent)` gives the per-env depth and raises unless
`memory_size` is a multiple of the envs per agent. PPO's buffer is `ppo.rollouts` steps per env.

### Emit a metric

* **Env side:** put a `(num_envs,)` tensor in `infos["metrics_to_log"]`. The learner slices it
  to each agent's envs and forwards it unchanged — no aggregation, no renaming. A wrong shape
  raises; a missing key forwards nothing.
* **Learner side:** build a `(num_agents,)` tensor and call `emit_per_agent({name: value},
  step)`. The name is free.

### Hooks

Both lists live on the learner and are **empty by default** (nothing is logged, no extra loss):

* `on_log`: `fn(agent_idx: int, metrics: dict[str, Tensor], step: int)`. Env metrics arrive as
  `(envs_per_agent,)` tensors, learner metrics as 0-d tensors.
* `aux_loss`: `fn(ctx: AuxLossContext) -> Tensor | None`, called for `ctx.target == "policy"`
  and `"critic"`. A returned tensor is added to that loss.

### Checkpoints

Per agent, at `{trainer.output_dir}/{wandb.project}/{wandb.group}/{run_name}/checkpoints/`:
`ckpt_{step}.pt` every `trainer.checkpoint_interval` steps, plus `ckpt_best.pt` whenever an
agent's mean episode return over a write interval improves. A file holds that agent's model
weights, optimizer state, normalizer stats, learner extras, the step and the mean return.
`load_agent(path, slot)` loads one file into any slot; optimizer state only with
`with_optimizer=True`, which needs `num_agents == 1`.

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

Entry script shape:

```python
parser = argparse.ArgumentParser()
add_config_args(parser)                       # --config / --from_run
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
app_launcher = AppLauncher(args)              # start the sim app before loading the task

register_section("controller", ControllerCfg)  # project sections, before loading
cfg = load_from_args(args, overrides)
env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
dump(cfg, run_dir, env.unwrapped.cfg)          # once, after the env exists
```

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

No test is skipped silently: a GPU test that cannot start Isaac Sim fails. Report each run as
passed / failed / total.

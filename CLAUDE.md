# robonuke_rl_core — Claude context

Shared RL package for Isaac Lab research: skrl agents, several independent agents trained in
parallel in one Isaac Sim instance on one GPU. Built area by area. **Today only the config
manager exists** (`robonuke_rl_core/config.py`).

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

`tests/<module>/` holds the tests for `robonuke_rl_core/<module>.py`. CPU tests sit directly
in it; tests that need Isaac Sim or a GPU go in `tests/<module>/GPU/` and carry
`@pytest.mark.gpu` (the marker is registered in `tests/conftest.py`).

```bash
pytest          # CPU tests only (addopts = -m 'not gpu')
pytest -m gpu   # the Isaac Sim tests, on the GPU machine
```

No test is skipped silently: a GPU test that cannot start Isaac Sim fails. Report each run as
passed / failed / total.

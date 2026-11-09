"""The three launchers, end to end, through `--dry_run`.

`--dry_run` prints the exact `sbatch` command it would run and submits nothing, which makes
it the test harness as well as a user-facing feature: everything about a job except SLURM
actually accepting it is asserted here, on CPU, with no cluster.

What matters most is **fail before queue**: a bad config in a batch must abort the whole
submit before the first job goes out, or half a sweep ends up running.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core.hpc import launch_eval, launch_sweep, launch_train, launch_watch
from robonuke_rl_core.hpc import submit as S

from hpc_helpers import flag, sbatch_lines, tree


def train(paths, *args, capsys=None, **kwargs):
    argv = [str(paths["root"]), "--project", "P", "--group_prefix", "G", "--dry_run", *args]
    code = launch_train.main(argv)
    out = capsys.readouterr().out if capsys else ""
    return code, sbatch_lines(out), out


# ------------------------------------------------------------------ launch_train
def test_a_folder_submits_one_job_per_config(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha", "beta"))
    code, lines, out = train(paths, capsys=capsys)
    assert code == 0 and len(lines) == 2
    assert "G_alpha" in lines[0] and "G_beta" in lines[1]
    # the underscore-prefixed base is an overlay, not a runnable experiment
    assert "_base" not in out.replace(str(paths["base"]), "")


def test_the_resource_flags_come_from_that_config_s_hpc_section(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  time: 2-00:00:00\n  mem: 64G\n  cpus: 4\n  gpus: 1\n")
    _, lines, _ = train(paths, capsys=capsys)
    line = lines[0]
    assert flag(line, "-A") == "virl-grp"
    assert flag(line, "-p") == "dgxh,tiamat"
    assert flag(line, "--time") == "2-00:00:00"
    assert flag(line, "--mem") == "64G"
    assert flag(line, "-c") == "4"
    assert "--gres=gpu:1" in line
    assert flag(line, "--signal") == "TERM@300"


def test_the_job_name_is_the_group_and_the_logs_carry_it(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    line = lines[0]
    assert flag(line, "-J") == "G_alpha"
    assert flag(line, "-o").endswith("/P/G_alpha_%j.out")
    assert flag(line, "-e").endswith("/P/G_alpha_%j.err")


def test_the_launcher_wandb_keys_come_last(tmp_path, capsys):
    """They must be the final CLI layer, so nothing a user passed can shadow them."""
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, "--tag", "cli_tag", "sac.actor_lr=1.0e-4", capsys=capsys)
    parts = lines[0].split()
    assert parts[-3:] == [
        "wandb.project=P",
        "wandb.group=G_alpha",
        "'wandb.tags=[base_tag,cli_tag]'",
    ]
    # the user's override is forwarded, before them
    assert parts.index("sac.actor_lr=1.0e-4") < parts.index("wandb.project=P")


def test_the_env_exports_carry_all_plus_the_rnk_variables(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    export = next(part for part in lines[0].split() if part.startswith("--export="))
    # ALL first, so WANDB_API_KEY reaches the job from the login shell
    assert export.startswith("--export=ALL,")
    for name in ("RNK_PKG_ROOT", "RNK_PROJECT_ROOT", "RNK_SIF", "RNK_CACHE_HOME", "RNK_PYTHON"):
        assert f"{name}=" in export


def test_the_second_image_and_the_extra_repos_are_exported_to_the_job(tmp_path, capsys):
    """The job script decides on the node (which GPU it got), so both ride along as RNK_*
    variables -- with ~ expanded here, where a shell would, not inside the container."""
    paths = tree(
        tmp_path,
        names=("alpha",),
        extra="hpc:\n  sif_image_sm70: ~/images/general-cu126.sif\n  extra_repos:\n    - ~/furniture_lab\n    - /nfs/share/other_repo\n",
    )
    _, lines, _ = train(paths, capsys=capsys)
    export = next(part for part in lines[0].split() if part.startswith("--export="))
    home = str(S.Path("~").expanduser())
    assert f"RNK_SIF_SM70={home}/images/general-cu126.sif" in export
    assert f"RNK_EXTRA_REPOS={home}/furniture_lab,/nfs/share/other_repo" in export


def test_the_second_image_and_the_extra_repos_are_exported_empty_when_unset(tmp_path, capsys):
    """Exported always, so the job script reads one spelling and never a missing variable."""
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    export = next(part for part in lines[0].split() if part.startswith("--export="))
    assert "RNK_SIF_SM70=," in export
    assert "RNK_EXTRA_REPOS=," in export


def test_a_constraint_becomes_the_sbatch_flag_and_an_empty_one_emits_nothing(tmp_path, capsys):
    """`--constraint` only when set: an empty `--constraint=` is an sbatch error, not "none"."""
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    assert "--constraint" not in lines[0]
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  constraint: vram80g|vram140g|v100\n")
    _, lines, _ = train(paths, capsys=capsys)
    # shlex-quoted on the dry-run line because of the |, so compare the parsed argv
    import shlex

    assert "--constraint=vram80g|vram140g|v100" in shlex.split(lines[0])


def test_a_constraint_with_whitespace_aborts_before_anything_is_queued(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  constraint: vram80g | v100\n")
    code, lines, out = train(paths, capsys=capsys)
    assert code == 2 and lines == []
    assert "hpc.constraint" in out


def test_an_extra_repo_with_whitespace_aborts_before_anything_is_queued(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  extra_repos:\n    - /home/me/furniture lab\n")
    code, lines, out = train(paths, capsys=capsys)
    assert code == 2 and lines == []
    assert "hpc.extra_repos[0]" in out


def test_a_real_submit_checks_the_second_image_is_a_file(tmp_path, monkeypatch):
    """Same gate as `sif_image`: a missing image must fail here, not hours later on a V100."""
    paths = tree(tmp_path, names=("alpha",))
    submit = S.read_submit_config(paths["configs"]["alpha"])
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/sbatch")  # pretend: a login node
    S.preflight([submit], dry_run=False)  # sif_image exists, sm70 unset: fine
    submit.hpc.sif_image_sm70 = str(tmp_path / "missing-cu126.sif")
    with pytest.raises(S.SubmitError) as err:
        S.preflight([submit], dry_run=False)
    assert "hpc.sif_image_sm70" in str(err.value) and "missing-cu126.sif" in str(err.value)
    (tmp_path / "missing-cu126.sif").write_text("")
    S.preflight([submit], dry_run=False)
    # a dry run prints commands and checks no file, like sif_image
    submit.hpc.sif_image_sm70 = str(tmp_path / "never-built.sif")
    S.preflight([submit], dry_run=True)


def test_the_job_runs_the_train_script_in_the_container_python(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    parts = lines[0].split()
    script = parts[parts.index(str(S.job_script()))]
    assert script == str(S.job_script())
    assert parts[parts.index(script) + 1 : parts.index(script) + 4] == [
        "python", "scripts/train.py", "--config",
    ]
    assert "--headless" in parts


def test_a_bad_config_aborts_the_whole_batch_before_anything_is_queued(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha", "beta"))
    paths["configs"]["beta"].write_text(
        paths["configs"]["beta"].read_text() + "hpc:\n  time: tomorrow\n"
    )
    code, lines, out = train(paths, capsys=capsys)
    assert code == 2
    assert lines == []  # not even alpha, which was fine
    assert "hpc.time" in out and "beta.yaml" in out


def test_a_missing_required_field_aborts_the_batch(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    paths["base"].write_text(paths["base"].read_text().replace("account: virl-grp", 'account: ""'))
    code, lines, out = train(paths, capsys=capsys)
    assert code == 2 and lines == []
    assert "hpc.account" in out


def test_an_override_of_a_launcher_owned_key_is_refused(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code, lines, out = train(paths, "wandb.group=mine", capsys=capsys)
    assert code == 2 and lines == []
    assert "--group_prefix" in out


def test_explicit_files_and_folders_can_be_mixed_and_deduplicated(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha", "beta"))
    code = launch_train.main(
        [
            str(paths["root"]),
            str(paths["configs"]["alpha"]),  # named twice: must run once
            "--project", "P", "--group_prefix", "G", "--dry_run",
        ]
    )
    lines = sbatch_lines(capsys.readouterr().out)
    assert code == 0 and len(lines) == 2


def test_an_empty_folder_is_an_error_not_a_quiet_success(tmp_path, capsys):
    empty = tmp_path / "nothing"
    empty.mkdir()
    code = launch_train.main([str(empty), "--project", "P", "--group_prefix", "G", "--dry_run"])
    assert code == 2
    assert "no runnable" in capsys.readouterr().out


# ------------------------------------------------------------------ train-then-eval
def test_eval_after_train_uses_the_chain_wrapper_and_names_every_agent(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    eval_config = tmp_path / "quick.yaml"
    eval_config.write_text("eval:\n  num_rollouts: 4\n")
    _, lines, _ = train(paths, "--eval_config", str(eval_config), capsys=capsys)
    export = next(part for part in lines[0].split() if part.startswith("--export="))
    assert "RNK_CHAIN_SCRIPT=" in export
    assert f"RNK_EVAL_CONFIG={eval_config}" in export
    # num_agents is 2 in the temp base, so both runs are named
    assert "RNK_EVAL_RUNS=hur/P/G_alpha_a0,hur/P/G_alpha_a1" in export
    assert "RNK_EVAL_CHECKPOINT=best" in export


def test_a_plain_train_job_uses_no_chain_wrapper(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    _, lines, _ = train(paths, capsys=capsys)
    assert "RNK_CHAIN_SCRIPT" not in lines[0]


def test_a_missing_eval_config_aborts(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code, lines, out = train(paths, "--eval_config", str(tmp_path / "nope.yaml"), capsys=capsys)
    assert code == 2 and lines == []
    assert "is not a file" in out


# ------------------------------------------------------------------ skip_existing
def test_skip_existing_skips_a_populated_run_dir_and_says_why(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha", "beta"))
    populated = paths["runs"] / "P" / "G_alpha"
    populated.mkdir(parents=True)
    (populated / "resolved_config.yaml").write_text("x")

    code, lines, out = train(paths, "--skip_existing", capsys=capsys)
    assert code == 0
    assert len(lines) == 1 and "G_beta" in lines[0]
    assert "exists and is non-empty" in out
    assert "skipped" in out


def test_an_empty_run_dir_is_not_skipped_and_the_reason_is_printed(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    (paths["runs"] / "P" / "G_alpha").mkdir(parents=True)
    code, lines, out = train(paths, "--skip_existing", capsys=capsys)
    assert code == 0 and len(lines) == 1
    assert "is empty" in out  # a silent skip hides a bug, so the reason is always printed


def test_a_missing_run_dir_is_not_skipped_and_the_reason_is_printed(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code, lines, out = train(paths, "--skip_existing", capsys=capsys)
    assert code == 0 and len(lines) == 1
    assert "no run dir at" in out


# ------------------------------------------------------------------ launch_sweep
def test_a_sweep_is_one_job_per_config_times_value(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha", "beta"))
    code = launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "sac.actor_lr", "--label", "lr",
            "--value", "1.0e-4", "--value", "3.0e-4", "--dry_run",
        ]
    )
    lines = sbatch_lines(capsys.readouterr().out)
    assert code == 0 and len(lines) == 4
    groups = [flag(line, "-J") for line in lines]
    assert groups == [
        "G_alpha_lr-1.0e-4", "G_alpha_lr-3.0e-4", "G_beta_lr-1.0e-4", "G_beta_lr-3.0e-4",
    ]


def test_the_sweep_override_lands_before_the_wandb_keys(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "sac.actor_lr", "--label", "lr", "--value", "1.0e-4",
            "--dry_run", "sac.batch_size=64",
        ]
    )
    parts = sbatch_lines(capsys.readouterr().out)[0].split()
    # user override, then the sweep's, then the launcher's names
    assert (
        parts.index("sac.batch_size=64")
        < parts.index("sac.actor_lr=1.0e-4")
        < parts.index("wandb.project=P")
    )
    assert parts[-1] == "'wandb.tags=[base_tag,lr-1.0e-4]'"


def test_a_sweep_tag_is_appended_to_the_merged_tags(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "x.y", "--label", "kp", "--value", "100",
            "--tag", "cli", "--dry_run",
        ]
    )
    assert "'wandb.tags=[base_tag,cli,kp-100]'" in sbatch_lines(capsys.readouterr().out)[0]


def test_a_non_scalar_sweep_value_aborts_before_queueing(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code = launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "wrappers.fragile.break_force", "--label", "bf",
            "--value", "[10.0,40.0]", "--dry_run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2 and sbatch_lines(out) == []
    assert "NAME=value" in out


def test_a_named_non_scalar_sweep_value_works(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code = launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "wrappers.fragile.break_force", "--label", "bf",
            "--value", "low=[10.0,40.0]", "--dry_run",
        ]
    )
    line = sbatch_lines(capsys.readouterr().out)[0]
    assert code == 0
    assert flag(line, "-J") == "G_alpha_bf-low"
    assert "wrappers.fragile.break_force=[10.0,40.0]" in line


def test_sweeping_a_launcher_owned_key_is_refused(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",))
    code = launch_sweep.main(
        [
            str(paths["root"]), "--project", "P", "--group_prefix", "G",
            "--sweep_param", "wandb.group", "--label", "g", "--value", "x", "--dry_run",
        ]
    )
    assert code == 2 and "--group_prefix" in capsys.readouterr().out


# ------------------------------------------------------------------ launch_eval
def test_eval_is_one_job_per_run_times_eval_config(tmp_path, capsys):
    paths = tree(tmp_path, names=("quick", "heavy"))
    code = launch_eval.main(
        [
            "--eval_config", str(paths["configs"]["quick"]),
            "--eval_config", str(paths["configs"]["heavy"]),
            "--run", "hur/P/g_a0", "--run", "hur/P/g_a1",
            "--dry_run",
        ]
    )
    lines = sbatch_lines(capsys.readouterr().out)
    assert code == 0 and len(lines) == 4
    assert [flag(line, "-J") for line in lines] == [
        "eval_g_a0_quick", "eval_g_a1_quick", "eval_g_a0_heavy", "eval_g_a1_heavy",
    ]


def test_eval_resources_come_from_the_eval_config_s_own_chain(tmp_path, capsys):
    paths = tree(tmp_path, names=("heavy",), extra="hpc:\n  time: 1-00:00:00\n  mem: 96G\n")
    launch_eval.main(
        ["--eval_config", str(paths["configs"]["heavy"]), "--run", "hur/P/g_a0", "--dry_run"]
    )
    line = sbatch_lines(capsys.readouterr().out)[0]
    assert flag(line, "--time") == "1-00:00:00" and flag(line, "--mem") == "96G"


def test_the_eval_job_calls_eval_py_with_the_run_and_the_checkpoint(tmp_path, capsys):
    paths = tree(tmp_path, names=("quick",))
    launch_eval.main(
        [
            "--eval_config", str(paths["configs"]["quick"]),
            "--run", "hur/P/g_a0", "--checkpoint", "2500", "--dry_run",
        ]
    )
    line = sbatch_lines(capsys.readouterr().out)[0]
    assert "scripts/eval.py --run hur/P/g_a0" in line
    assert "--checkpoint 2500" in line
    assert "--headless" in line


def test_the_eval_logs_go_under_the_run_s_project(tmp_path, capsys):
    paths = tree(tmp_path, names=("quick",))
    launch_eval.main(
        ["--eval_config", str(paths["configs"]["quick"]), "--run", "hur/forge_pih/g_a0", "--dry_run"]
    )
    line = sbatch_lines(capsys.readouterr().out)[0]
    assert flag(line, "-o").endswith("/forge_pih/eval_g_a0_quick_%j.out")


@pytest.mark.parametrize("path", ["g_a0", "hur/g_a0", "a/b/c/d"])
def test_a_malformed_run_path_aborts_before_queueing(tmp_path, capsys, path):
    paths = tree(tmp_path, names=("quick",))
    code = launch_eval.main(
        ["--eval_config", str(paths["configs"]["quick"]), "--run", path, "--dry_run"]
    )
    out = capsys.readouterr().out
    assert code == 2 and sbatch_lines(out) == []
    assert "entity/project/<run id or name>" in out


def test_eval_needs_runs_or_a_project_to_query(tmp_path, capsys):
    paths = tree(tmp_path, names=("quick",))
    code = launch_eval.main(["--eval_config", str(paths["configs"]["quick"]), "--dry_run"])
    assert code == 2 and "--project" in capsys.readouterr().out


def test_a_missing_eval_config_aborts_before_any_wandb_query(tmp_path, capsys):
    code = launch_eval.main(
        ["--eval_config", str(tmp_path / "nope.yaml"), "--project", "P", "--dry_run"]
    )
    assert code == 2 and "is not a file" in capsys.readouterr().out


# ------------------------------------------------------------------ what sbatch can carry
def test_an_argument_with_whitespace_is_refused(tmp_path):
    """sbatch word-splits the arguments it passes to a job script.

    A multi-word argument therefore reaches the job as several, and a quoted python snippet
    arrives as `-c import`, dying with a SyntaxError inside a container for no visible reason.
    Caught at composition time instead, where the message can say what to do.
    """
    with pytest.raises(S.SubmitError) as err:
        S.sbatch_command(
            hpc=S.HpcCfg(),
            job_name="j",
            out_path=tmp_path / "o",
            err_path=tmp_path / "e",
            env={},
            argv=["python", "-c", "import os\nprint(1)"],
        )
    message = str(err.value)
    assert "whitespace" in message and "sbatch does not preserve" in message
    assert "put the code in a file" in message


def test_a_tag_list_with_a_space_is_refused_too(tmp_path):
    """The realistic version: `wandb.tags=[a, b]` would silently become two arguments."""
    with pytest.raises(S.SubmitError) as err:
        S.sbatch_command(
            hpc=S.HpcCfg(),
            job_name="j",
            out_path=tmp_path / "o",
            err_path=tmp_path / "e",
            env={},
            argv=["python", "scripts/train.py", "wandb.tags=[a, b]"],
        )
    assert "wandb.tags=[a,b]" in str(err.value)  # it shows the right shape


def test_the_launchers_never_produce_a_whitespace_argument(tmp_path, capsys):
    """Every real command the launchers build must already satisfy the rule."""
    paths = tree(tmp_path, names=("alpha",))
    code, lines, _ = train(paths, "--tag", "one", "--tag", "two", capsys=capsys)
    assert code == 0 and lines


# ------------------------------------------------------------------ hpc.train_script
def test_a_config_s_train_script_is_the_entry_the_job_runs(tmp_path, capsys):
    """A project learner (e.g. BC) has its own entry script; the config names it."""
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  train_script: scripts/train_bc.py\n")
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts" / "train_bc.py").write_text("")
    code, lines, _ = train(paths, "--project_root", str(project), capsys=capsys)
    assert code == 0
    parts = lines[0].split()
    script = str(S.job_script())
    assert parts[parts.index(script) + 1 : parts.index(script) + 3] == ["python", "scripts/train_bc.py"]
    # and a sweep follows it too
    argv = [str(paths["root"]), "--project", "P", "--group_prefix", "G", "--dry_run", "--project_root", str(project),
            "--sweep_param", "sac.actor_lr", "--label", "lr", "--value", "1.0e-4"]
    assert launch_sweep.main(argv) == 0
    sweep = sbatch_lines(capsys.readouterr().out)[0].split()
    assert sweep[sweep.index(script) + 2] == "scripts/train_bc.py"


def test_a_missing_train_script_aborts_before_anything_is_queued(tmp_path, capsys):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  train_script: scripts/nope.py\n")
    code, lines, out = train(paths, "--project_root", str(tmp_path), capsys=capsys)
    assert code == 2 and not lines and "hpc.train_script" in out


@pytest.mark.parametrize("bad", ["/abs/train.py", "~/train.py", "scripts/train", "scripts/my train.py", ""])
def test_a_bad_train_script_is_rejected(bad):
    from robonuke_rl_core.hpc.cfg import HpcCfg

    with pytest.raises(ValueError, match="train_script"):
        HpcCfg(train_script=bad).validate(None)


def test_a_bind_expands_tilde_on_both_sides(monkeypatch):
    """`~/x:~/x` must become two absolute paths: a literal `~` on the container side would be
    created as a directory named `~`, and the project's runs/ symlink into it would dangle."""
    from robonuke_rl_core.hpc.submit import expand_bind

    monkeypatch.setenv("HOME", "/home/someone")
    assert expand_bind("~/share/robonuke:~/share/robonuke") == "/home/someone/share/robonuke:/home/someone/share/robonuke"
    assert expand_bind("~/share/x:/opt/x:ro") == "/home/someone/share/x:/opt/x:ro"
    assert expand_bind("/abs/a:/abs/b") == "/abs/a:/abs/b"


# ------------------------------------------------------------------ launch_watch
def _watch_tree(tmp_path, *, groups=("G_alpha", "G_beta"), extra="hpc:\n  watch_script: scripts/watch.py\n"):
    """A training tree plus an eval config (``quick``) carrying the watch script, the script itself under the
    project root (``tmp_path``) and the run directories the trainings would have written."""
    paths = tree(tmp_path, names=("alpha", "beta", "quick"), extra=extra)
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "watch.py").write_text("")
    for group in groups:
        (paths["runs"] / "P" / group).mkdir(parents=True, exist_ok=True)
    return paths


def watch(paths, tmp_path, *args, capsys=None):
    argv = [
        str(paths["configs"]["alpha"]), str(paths["configs"]["beta"]),
        "--project", "P", "--group_prefix", "G",
        "--eval_config", str(paths["configs"]["quick"]),
        "--project_root", str(tmp_path), "--dry_run", *args,
    ]
    code = launch_watch.main(argv)
    out = capsys.readouterr().out if capsys else ""
    return code, sbatch_lines(out), out


def test_watch_is_one_job_per_training_config_over_its_run_dir(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    code, lines, _ = watch(paths, tmp_path, capsys=capsys)
    assert code == 0 and len(lines) == 2
    assert [flag(line, "-J") for line in lines] == ["watch_G_alpha", "watch_G_beta"]
    # the same group rule as launch_train, so the run dir is the training's
    assert f"scripts/watch.py --run {paths['runs']}/P/G_alpha --eval_config {paths['configs']['quick']}" in lines[0]
    assert "--headless" in lines[0] and "--once" in lines[0]
    assert flag(lines[1], "-o").endswith("/P/watch_G_beta_%j.out")


def test_watch_follow_drops_once(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    _, lines, _ = watch(paths, tmp_path, "--follow", capsys=capsys)
    assert "--once" not in lines[0] and "--headless" in lines[0]


def test_watch_resources_and_script_come_from_the_eval_config_s_chain(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    # the eval config asks for its own resources; the training files keep the defaults
    paths["configs"]["quick"].write_text(
        paths["configs"]["quick"].read_text() + "  time: 0-16:00:00\n  mem: 64G\n  partitions: dgx2\n"
    )
    _, lines, _ = watch(paths, tmp_path, capsys=capsys)
    assert flag(lines[0], "--time") == "0-16:00:00" and flag(lines[0], "--mem") == "64G"
    assert flag(lines[0], "-p") == "dgx2"


def test_watch_forwards_overrides_after_its_own_flags(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    _, lines, _ = watch(paths, tmp_path, "eval.num_rollouts=50", capsys=capsys)
    assert lines[0].rstrip().endswith("--once eval.num_rollouts=50")


def test_watch_without_a_watch_script_aborts_naming_the_field(tmp_path, capsys):
    paths = _watch_tree(tmp_path, extra="")
    code, lines, out = watch(paths, tmp_path, capsys=capsys)
    assert code == 2 and lines == [] and "hpc.watch_script" in out


def test_watch_with_a_missing_watch_script_file_aborts(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    (tmp_path / "scripts" / "watch.py").unlink()
    code, lines, out = watch(paths, tmp_path, capsys=capsys)
    assert code == 2 and lines == [] and "scripts/watch.py" in out and "not a file" in out


def test_watch_dry_run_names_a_missing_run_dir_but_still_prints(tmp_path, capsys):
    paths = _watch_tree(tmp_path, groups=("G_alpha",))
    code, lines, out = watch(paths, tmp_path, capsys=capsys)
    assert code == 0 and len(lines) == 2
    assert "missing" in out and "G_beta" in out


def test_a_real_watch_submit_aborts_on_a_missing_run_dir_before_queueing(tmp_path, capsys, monkeypatch):
    paths = _watch_tree(tmp_path, groups=("G_alpha",))
    calls = []
    monkeypatch.setattr(S, "run_sbatch", lambda command, *, dry_run: calls.append(command) or (True, "0"))
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/sbatch")
    argv = [
        str(paths["configs"]["alpha"]), str(paths["configs"]["beta"]),
        "--project", "P", "--group_prefix", "G",
        "--eval_config", str(paths["configs"]["quick"]), "--project_root", str(tmp_path),
    ]
    code = launch_watch.main(argv)
    out = capsys.readouterr().out
    assert code == 2 and calls == [] and "G_beta" in out and "Nothing was submitted" in out


def test_watch_rejects_a_launcher_owned_override(tmp_path, capsys):
    paths = _watch_tree(tmp_path)
    code, lines, out = watch(paths, tmp_path, "wandb.group=x", capsys=capsys)
    assert code == 2 and lines == [] and "wandb.group" in out

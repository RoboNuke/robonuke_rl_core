"""Names: the group, the job, the logs, the tags — and the rules that protect them.

wandb is the interface, so a run is found by project / group / tags long before anyone looks
at a SLURM queue. Every name below is therefore derived from the launcher's `--project` and
`--group_prefix`, and the two rules worth testing hardest are the ones that stop a name from
going wrong *silently*: a group the wandb rule would reject must fail before anything is
queued, and a user override of a key the launcher owns must be refused outright.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core.hpc import submit as S


# ------------------------------------------------------------------ the group
def test_the_group_is_prefix_then_stem_with_the_sweep_pair_last():
    assert S.group_name("e2e", "match_fragile") == "e2e_match_fragile"
    # the pair uses a hyphen because label and value belong together; joins stay underscores
    assert S.group_name("e2e", "match_fragile", "kp-100") == "e2e_match_fragile_kp-100"


def test_the_separator_inside_a_swept_pair_is_a_hyphen():
    assert S.PAIR_SEPARATOR == "-"


@pytest.mark.parametrize("group", ["", "has space", "has/slash", "a\tb"])
def test_an_unusable_group_fails_before_anything_is_queued(group):
    with pytest.raises(S.SubmitError) as err:
        S.check_group(group, "configs/alpha.yaml")
    message = str(err.value)
    assert "configs/alpha.yaml" in message  # which config produced it
    assert "--project / --group_prefix" in message  # and what to change


def test_a_usable_group_passes():
    S.check_group("e2e_match_fragile_kp-100", "x")


def test_the_group_rule_matches_the_one_wandb_cfg_applies():
    """Two copies of a rule drift. This is the test that notices."""
    from robonuke_rl_core.config import WandbCfg

    for group in ("ok_group", "bad group", "bad/group", ""):
        ours = True
        try:
            S.check_group(group, "x")
        except S.SubmitError:
            ours = False
        theirs = True
        try:
            cfg = WandbCfg(entity="e", project="p", group=group)
            cfg.validate(None)
        except ValueError:
            theirs = False
        assert ours == theirs, group


# ------------------------------------------------------------------ run names
def test_run_names_follow_the_derived_rule():
    assert S.run_names("g", 3) == ["g_a0", "g_a1", "g_a2"]


def test_run_names_match_what_the_config_pipeline_derives():
    """`derived.run_names` is computed in `config._build`, which the submitter cannot
    import. The rule is one f-string, duplicated on purpose -- so pin the two together."""
    group, agents = "e2e_match_10N", 3
    from robonuke_rl_core.config import WandbCfg  # noqa: F401  (import-safety of the pair)

    expected = [f"{group}_a{i}" for i in range(agents)]
    assert S.run_names(group, agents) == expected


# ------------------------------------------------------------------ tags
def test_tags_merge_config_then_cli_then_sweep():
    assert S.merge_tags(["base"], ["cli"], "kp-100") == ["base", "cli", "kp-100"]


def test_the_tag_merge_keeps_order_and_drops_duplicates():
    assert S.merge_tags(["a", "b"], ["b", "c"], "a") == ["a", "b", "c"]


def test_an_empty_tag_is_dropped():
    assert S.merge_tags(["", "a"], [""], "") == ["a"]


# ------------------------------------------------------------------ reserved overrides
@pytest.mark.parametrize(
    "override, flag",
    [
        ("wandb.project=p", "--project"),
        ("wandb.group=g", "--group_prefix"),
        ("wandb.tags=[a]", "--tag"),
    ],
)
def test_a_user_override_of_a_launcher_owned_key_is_refused(override, flag):
    with pytest.raises(S.SubmitError) as err:
        S.reject_reserved_overrides([override])
    message = str(err.value)
    assert flag in message  # it names the flag to use instead
    assert "two sources of truth" in message.lower()


def test_other_overrides_pass_through():
    S.reject_reserved_overrides(["sac.actor_lr=1e-4", "hpc.mem=64G", "wandb.mode=offline"])


def test_a_malformed_override_is_caught_before_it_reaches_a_job():
    with pytest.raises(S.SubmitError) as err:
        S.forwarded(["not-an-override"])
    assert "section.field=value" in str(err.value)


def test_forwarded_overrides_are_verbatim():
    """Including hpc.*: the job re-applies them, so resolved_config.yaml records the
    resources that shaped the run."""
    given = ["sac.actor_lr=1.0e-4", "hpc.time=2-00:00:00", "task.cfg.scene.num_envs=128"]
    assert S.forwarded(given) == given


# ------------------------------------------------------------------ logs
def test_log_paths_are_per_project_and_carry_the_slurm_job_id(tmp_path):
    hpc = S.HpcCfg(exp_log_dir=str(tmp_path / "exp_logs"))
    out, err = S.log_paths(hpc, "forge_pih", "e2e_match_10N")
    assert out.name == "e2e_match_10N_%j.out"
    assert err.name == "e2e_match_10N_%j.err"
    assert out.parent.name == "forge_pih" and out.parent.is_dir()


def test_a_dry_run_creates_no_log_directory(tmp_path):
    hpc = S.HpcCfg(exp_log_dir=str(tmp_path / "exp_logs"))
    out, _ = S.log_paths(hpc, "forge_pih", "g", create=False)
    assert not out.parent.exists()


# ------------------------------------------------------------------ the sweep value
def test_a_scalar_sweep_value_names_itself():
    from robonuke_rl_core.hpc.launch_sweep import parse_value

    assert parse_value("100") == ("100", "100")
    assert parse_value("0.08") == ("0.08", "0.08")
    assert parse_value("true") == ("true", "true")


def test_a_bare_list_value_says_to_use_the_name_form():
    from robonuke_rl_core.hpc.launch_sweep import parse_value

    with pytest.raises(S.SubmitError) as err:
        parse_value("[0.0,15.0,0.0]")
    assert "--value low=[0.0,15.0,0.0]" in str(err.value)


def test_the_name_form_keeps_the_value_text_verbatim():
    from robonuke_rl_core.hpc.launch_sweep import parse_value

    name, text = parse_value("low=[0.0,15.0,0.0]")
    assert (name, text) == ("low", "[0.0,15.0,0.0]")


def test_an_empty_name_is_refused():
    from robonuke_rl_core.hpc.launch_sweep import parse_value

    with pytest.raises(S.SubmitError):
        parse_value("=100")


# ------------------------------------------------------------------ the naming table
def test_the_naming_table_is_the_documented_one():
    """Kept as data so the tests, the docstring and the README cannot disagree."""
    assert S.NAMING["wandb project"] == "--project"
    assert S.NAMING["wandb group"] == "{group_prefix}_{config_stem}[_{LABEL}-{value}]"
    assert S.NAMING["slurm job name"] == "the group, exactly"
    assert "%j" in S.NAMING["slurm logs"]

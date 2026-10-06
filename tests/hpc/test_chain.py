"""The login-node chain reader: same `base` rules as `load_config`, without Isaac Lab.

The submitters cannot call `load_config` — step 4 needs the task's env cfg, which needs
Isaac Lab, which a login node does not have. So they read the chain through
`configfile.load_file_chain`, the *same* function `load_config` uses. These tests exist to
keep the two from drifting: if `base` ever means something different to the submitter than it
does to a training run, a job silently trains the wrong experiment.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core.configfile import chain, load_file_chain
from robonuke_rl_core.hpc.submit import SubmitError, read_submit_config, require_submit_fields

from hpc_helpers import tree


def test_it_follows_base_across_directories(tmp_path):
    outer = tmp_path / "base"
    inner = tmp_path / "exp" / "deep"
    outer.mkdir(parents=True)
    inner.mkdir(parents=True)
    (outer / "root.yaml").write_text("hpc:\n  time: 1-00:00:00\n")
    (inner / "mid.yaml").write_text("base: ../../base/root.yaml\nhpc:\n  mem: 64G\n")
    leaf = inner / "leaf.yaml"
    leaf.write_text("base: mid.yaml\nhpc:\n  cpus: 4\n")

    followed = chain(leaf)
    assert [p.name for p in followed] == ["root.yaml", "mid.yaml", "leaf.yaml"]
    cfg = read_submit_config(leaf)
    assert (cfg.hpc.time, cfg.hpc.mem, cfg.hpc.cpus) == ("1-00:00:00", "64G", 4)


def test_the_closest_file_wins_and_the_cli_wins_over_all(tmp_path):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  mem: 16G\n")
    config = paths["configs"]["alpha"]
    assert read_submit_config(config).hpc.mem == "16G"  # experiment beats base
    assert read_submit_config(config, ["hpc.mem=128G"]).hpc.mem == "128G"


def test_a_missing_base_names_the_file_that_asked_for_it(tmp_path):
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text("base: nope.yaml\n")
    with pytest.raises(FileNotFoundError) as err:
        chain(leaf)
    assert "nope.yaml" in str(err.value) and "relative to the file that names it" in str(err.value)


def test_a_cycle_prints_the_chain(tmp_path):
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("base: b.yaml\n")
    second.write_text("base: a.yaml\n")
    with pytest.raises(ValueError) as err:
        chain(first)
    assert "cycle" in str(err.value) and "a.yaml" in str(err.value)


def test_interpolation_is_rejected(tmp_path):
    path = tmp_path / "leaf.yaml"
    path.write_text("hpc:\n  mem: ${oc.env:MEM}\n")
    with pytest.raises(ValueError) as err:
        load_file_chain(path)
    assert "interpolation" in str(err.value)


def test_an_unknown_hpc_key_is_rejected_naming_the_file(tmp_path):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  wall_time: 1:00:00\n")
    with pytest.raises(SubmitError) as err:
        read_submit_config(paths["configs"]["alpha"])
    assert "hpc section error" in str(err.value) and "alpha.yaml" in str(err.value)


def test_a_bad_format_is_reported_against_the_config_path(tmp_path):
    paths = tree(tmp_path, names=("alpha",), extra="hpc:\n  time: tomorrow\n")
    with pytest.raises(SubmitError) as err:
        read_submit_config(paths["configs"]["alpha"])
    assert "hpc.time" in str(err.value) and "alpha.yaml" in str(err.value)


def test_it_reads_what_the_launchers_need_from_the_chain(tmp_path):
    paths = tree(tmp_path, names=("alpha",))
    cfg = read_submit_config(paths["configs"]["alpha"])
    assert cfg.stem == "alpha"
    assert cfg.entity == "hur"
    assert cfg.tags == ["base_tag"]
    assert cfg.num_agents == 2
    assert cfg.output_dir == str(paths["runs"])


def test_output_dir_falls_back_to_the_trainer_default(tmp_path):
    from robonuke_rl_core.learners.cfg import TrainerCfg

    path = tmp_path / "bare.yaml"
    path.write_text("hpc:\n  account: a\n")
    assert read_submit_config(path).output_dir == TrainerCfg().output_dir


def test_required_at_submit_names_the_field_and_where_to_set_it(tmp_path):
    path = tmp_path / "bare.yaml"
    path.write_text("hpc:\n  account: a\n")  # partitions, sif_image, cache_home all unset
    cfg = read_submit_config(path)
    cfg.hpc.validate(None)  # the SECTION is happy: formats only
    with pytest.raises(SubmitError) as err:
        require_submit_fields(cfg)
    message = str(err.value)
    # the list names exactly what is missing -- account is set, so it is not in it
    assert "hpc.partitions, hpc.sif_image, hpc.cache_home must be set" in message
    assert "configs/base/hpc.yaml" in message  # and it says where to put them
    assert "never an NFS home" in message  # cache_home's trap is called out


def test_a_complete_config_passes_the_submit_gate(tmp_path):
    paths = tree(tmp_path, names=("alpha",))
    require_submit_fields(read_submit_config(paths["configs"]["alpha"]))

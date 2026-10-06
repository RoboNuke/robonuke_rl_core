"""The `hpc` section: defaults, the format rules, and the round trip.

`hpc` is a `config.py`-owned section, so it loads in every normal training run — which is
exactly why **every field has a default**. A `MISSING` field here would make a local run that
never touches SLURM fail on a field it does not need. Required-ness belongs to the submitter.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core.hpc.cfg import SUBMIT_REQUIRED, HpcCfg


def test_the_defaults_are_valid_and_submit_nothing_on_their_own():
    cfg = HpcCfg()
    cfg.validate(None)  # a local run must never trip over this section
    # the fields a job cannot be built without are empty, so the submitter has to ask
    for name in SUBMIT_REQUIRED:
        assert getattr(cfg, name) == "", name


def test_the_resource_defaults_are_the_documented_ones():
    cfg = HpcCfg()
    assert (cfg.time, cfg.gpus, cfg.mem, cfg.cpus) == ("0-09:00:00", 1, "32G", 12)
    assert cfg.signal == "TERM@300"
    assert cfg.apptainer_bin == "apptainer"
    assert cfg.exp_log_dir == "exp_logs"
    assert cfg.binds == []


@pytest.mark.parametrize(
    "field, value, needle",
    [
        ("gpus", 0, "hpc.gpus"),
        ("cpus", 0, "hpc.cpus"),
        ("time", "9:00", "hpc.time"),
        ("time", "0-9:0:0", "hpc.time"),
        ("time", "tomorrow", "hpc.time"),
        ("signal", "TERM", "hpc.signal"),
        ("signal", "300", "hpc.signal"),
        ("binds", ["/a /b:/c"], "hpc.binds[0]"),
        ("binds", [""], "hpc.binds[0]"),
    ],
)
def test_each_format_rule_fails_on_a_bad_value(field, value, needle):
    cfg = HpcCfg(**{field: value})
    with pytest.raises(ValueError) as err:
        cfg.validate(None)
    assert needle in str(err.value)


@pytest.mark.parametrize("value", ["0-09:00:00", "12:00:00", "2-00:00:00", "7-23:59:59"])
def test_the_walltime_forms_slurm_accepts(value):
    HpcCfg(time=value).validate(None)


@pytest.mark.parametrize("value", ["TERM@300", "USR1@60", "B:TERM@120", "R:USR2@30"])
def test_the_signal_forms_slurm_accepts(value):
    HpcCfg(signal=value).validate(None)


def test_an_unset_value_is_not_an_error():
    """validate checks formats only, and only on values that are set."""
    HpcCfg(time="", signal="", account="", sif_image="").validate(None)


def test_the_section_is_registered_and_round_trips():
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import SECTIONS

    assert SECTIONS["hpc"] is HpcCfg
    node = OmegaConf.structured(HpcCfg(account="a", binds=["/x:/y"]))
    back = OmegaConf.to_object(OmegaConf.create(OmegaConf.to_yaml(node)))
    assert back["account"] == "a" and back["binds"] == ["/x:/y"]


def test_an_unknown_hpc_key_is_rejected():
    from omegaconf import OmegaConf

    node = OmegaConf.structured(HpcCfg)
    OmegaConf.set_struct(node, True)
    with pytest.raises(Exception):
        OmegaConf.merge(node, OmegaConf.create({"wall_time": "1:00:00"}))

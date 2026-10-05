"""Where eval gets a policy from: a wandb run (the usual way) or a local run directory.

The wandb side runs against a fake Api, so it needs no network and no login. What it pins is
the contract: plain run FILES, never the Artifacts API, and a local directory afterwards that
the local path can consume unchanged.
"""

from __future__ import annotations

import pytest

from robonuke_rl_core.config import RESOLVED_NAME
from robonuke_rl_core.evaluation import (
    checkpoint_file,
    fetch_wandb_run,
    find_checkpoint,
    resolved_config_path,
)
from robonuke_rl_core.learners.base import CHECKPOINT_BEST, checkpoint_name


# ------------------------------------------------------------------ the fake wandb Api
class FakeFile:
    def __init__(self, name: str, size: int, body: str = "x"):
        self.name, self.size, self.body = name, size, body
        self.downloaded_to: str | None = None

    def download(self, root: str, replace: bool = False):
        from pathlib import Path

        self.downloaded_to = root
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.body)
        return path


class FakeRun:
    def __init__(self, run_id: str, name: str, files: dict):
        self.id, self.display_name, self._files = run_id, name, files
        self.asked: list[str] = []
        self.uploaded: list[tuple[str, str]] = []

    def upload_file(self, path: str, root: str = "."):
        self.uploaded.append((path, root))

    def file(self, name: str):
        self.asked.append(name)
        return self._files.get(name) or FakeFile(name, 0)  # wandb returns an empty stub


class FakeApi:
    """Only ``run`` / ``runs`` / ``file`` — an Artifacts call would be an AttributeError."""

    def __init__(self, runs: dict, named: dict | None = None):
        self._runs, self._named = runs, named or {}

    def run(self, path: str):
        if path not in self._runs:
            raise ValueError(f"could not find run {path}")
        return self._runs[path]

    def runs(self, path: str, filters: dict):
        return list(self._named.get((path, filters.get("display_name")), []))


def a_run(run_id: str = "abc123", name: str = "grp_a0") -> FakeRun:
    return FakeRun(
        run_id,
        name,
        {
            RESOLVED_NAME: FakeFile(RESOLVED_NAME, 120, "experiment:\n  seed: 3\n"),
            CHECKPOINT_BEST: FakeFile(CHECKPOINT_BEST, 2048),
            checkpoint_name(40): FakeFile(checkpoint_name(40), 2048),
        },
    )


# ------------------------------------------------------------------ checkpoint naming
def test_a_checkpoint_argument_means_best_a_step_or_a_file_name():
    assert checkpoint_file("best") == CHECKPOINT_BEST
    assert checkpoint_file() == CHECKPOINT_BEST  # the default
    assert checkpoint_file("40") == checkpoint_name(40)
    assert checkpoint_file("ckpt_custom.ckpt") == "ckpt_custom.ckpt"


# ------------------------------------------------------------------ the wandb path
def test_a_wandb_run_downloads_into_a_local_run_directory(tmp_path):
    run = a_run()
    api = FakeApi({"ent/proj/abc123": run})

    directory = fetch_wandb_run("ent/proj/abc123", "best", tmp_path, api=api)

    assert directory == tmp_path / "ent" / "proj" / "abc123"
    # the result is shaped like a local run, so the local code path takes over unchanged
    assert resolved_config_path(directory) == directory / RESOLVED_NAME
    assert find_checkpoint(directory, "best") == directory / "checkpoints" / CHECKPOINT_BEST
    # only plain run files were touched
    assert run.asked == [RESOLVED_NAME, CHECKPOINT_BEST]


def test_a_named_run_is_found_when_the_id_does_not_match(tmp_path):
    run = a_run(run_id="xyz789", name="fgain_k100_a2")
    api = FakeApi({}, {("ent/proj", "fgain_k100_a2"): [run]})

    directory = fetch_wandb_run("ent/proj/fgain_k100_a2", "40", tmp_path, api=api)
    assert directory == tmp_path / "ent" / "proj" / "xyz789"
    assert find_checkpoint(directory, "40").name == checkpoint_name(40)


def test_an_ambiguous_or_missing_run_raises(tmp_path):
    api = FakeApi({}, {("ent/proj", "twice"): [a_run("a"), a_run("b")]})
    with pytest.raises(LookupError) as err:
        fetch_wandb_run("ent/proj/twice", cache_root=tmp_path, api=api)
    assert "2 runs" in str(err.value) and "twice" in str(err.value)

    with pytest.raises(LookupError) as err:
        fetch_wandb_run("ent/proj/nope", cache_root=tmp_path, api=api)
    assert "nope" in str(err.value)


def test_a_missing_file_on_the_run_raises_naming_it(tmp_path):
    run = a_run()
    api = FakeApi({"ent/proj/abc123": run})
    with pytest.raises(FileNotFoundError) as err:
        fetch_wandb_run("ent/proj/abc123", "ckpt_999.ckpt", tmp_path, api=api)
    assert "ckpt_999.ckpt" in str(err.value) and "ent/proj/abc123" in str(err.value)


@pytest.mark.parametrize("spec", ["proj/run", "ent/proj/run/extra", "", "justarun"])
def test_a_malformed_run_spec_raises(spec, tmp_path):
    with pytest.raises(ValueError) as err:
        fetch_wandb_run(spec, cache_root=tmp_path, api=FakeApi({}))
    assert "entity/project/run" in str(err.value)


# ------------------------------------------------------------------ the local path
def test_a_local_run_dir_finds_the_group_config_and_its_checkpoints(tmp_path):
    group = tmp_path / "proj" / "group"
    run_dir = group / "group_a0"
    (run_dir / "checkpoints").mkdir(parents=True)
    (group / RESOLVED_NAME).write_text("experiment:\n  seed: 3\n")  # written per group
    (run_dir / "checkpoints" / CHECKPOINT_BEST).write_text("x")
    (run_dir / "checkpoints" / checkpoint_name(80)).write_text("x")

    assert resolved_config_path(run_dir) == group / RESOLVED_NAME
    assert find_checkpoint(run_dir, "best").name == CHECKPOINT_BEST
    assert find_checkpoint(run_dir, "80").name == checkpoint_name(80)
    # an explicit path wins over any lookup
    explicit = run_dir / "checkpoints" / checkpoint_name(80)
    assert find_checkpoint(run_dir, str(explicit)) == explicit


def test_a_missing_local_config_or_checkpoint_raises_with_what_it_looked_for(tmp_path):
    run_dir = tmp_path / "group" / "group_a0"
    (run_dir / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as err:
        resolved_config_path(run_dir)
    assert RESOLVED_NAME in str(err.value) and "group" in str(err.value)

    (run_dir.parent / RESOLVED_NAME).write_text("experiment:\n  seed: 1\n")
    (run_dir / "checkpoints" / checkpoint_name(5)).write_text("x")
    with pytest.raises(FileNotFoundError) as err:
        find_checkpoint(run_dir, "best")
    assert CHECKPOINT_BEST in str(err.value)
    assert checkpoint_name(5) in str(err.value)  # it says what is there instead


# ------------------------------------------------------------------ results back to the run
def test_eval_results_upload_under_an_eval_prefix(tmp_path):
    from robonuke_rl_core.evaluation import upload_eval_results

    run = a_run()
    api = FakeApi({"ent/proj/abc123": run})
    out_dir = tmp_path / "eval" / "quick_20260101-000000"
    (out_dir / "videos").mkdir(parents=True)
    (out_dir / "summary.yaml").write_text("summary: {}\n")
    (out_dir / "quick.parquet").write_bytes(b"PAR1")
    (out_dir / "videos" / "round0_env0.mp4").write_bytes(b"mp4")

    names = upload_eval_results("ent/proj/abc123", out_dir, tmp_path, api=api)

    # stored paths carry the eval-config stem and timestamp, so repeat evals never collide
    assert names == [
        "eval/quick_20260101-000000/quick.parquet",
        "eval/quick_20260101-000000/summary.yaml",
        "eval/quick_20260101-000000/videos/round0_env0.mp4",
    ]
    assert [root for _, root in run.uploaded] == [str(tmp_path)] * 3


def test_uploading_nothing_raises(tmp_path):
    from robonuke_rl_core.evaluation import upload_eval_results

    api = FakeApi({"ent/proj/abc123": a_run()})
    empty = tmp_path / "eval" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        upload_eval_results("ent/proj/abc123", empty, tmp_path, api=api)

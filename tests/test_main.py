"""Independent representation construction and training entry points."""

from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest

import construct_data_representations as construction
import train_pipeline
from data_processing.music_representations.core.builder import BuildStatus
from trainer.spec import NGRAM_NOTE_TABLE, RNN_REMI
from trainer import tracking


@pytest.fixture()
def env(monkeypatch, tmp_path: Path):
    for key in (
        "MUSIC_REPR_INPUT_DIR", "MUSIC_REPR_OUTPUT_DIR", "MUSIC_REPR_REPRESENTATIONS",
        "MUSIC_REPR_OVERWRITE", "MUSIC_REPR_LIMIT", "MUSIC_REPR_CONFIG",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(construction, "DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setattr(train_pipeline, "DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("MUSIC_REPR_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("MUSIC_REPR_OUTPUT_DIR", str(tmp_path / "prepared"))
    return monkeypatch


def test_construction_only_builds(env):
    calls = []

    class Builder:
        def __init__(self, config):
            pass

        def build(self, names, overwrite=False):
            calls.append((names, overwrite))
            return [SimpleNamespace(representation=name, status=BuildStatus.SKIPPED,
                                    location="", message="", error=None) for name in names]

    env.setenv("MUSIC_REPR_REPRESENTATIONS", "bpe, tsd")
    env.setenv("MUSIC_REPR_OVERWRITE", "true")
    env.setattr(construction, "MusicDatasetBuilder", Builder)
    assert construction.main() == 0
    assert calls == [(["bpe", "tsd"], True)]


def test_training_does_not_build(env, tmp_path):
    seen = []
    env.setattr(train_pipeline, "run", lambda settings, **kwargs: seen.append((settings, kwargs)) or {"jsd": {}})
    config = train_pipeline.load_config(RNN_REMI)
    train_pipeline.train_pipeline(config, run_name="rnn-test")
    assert seen[0][0].representation == "remi"
    assert seen[0][1]["run_name"] == "rnn-test"
    assert seen[0][1]["run_id"]


def test_presets_and_repeated_names_get_distinct_artifact_paths(env):
    seen = []
    env.setattr(train_pipeline, "run", lambda settings, **kwargs: seen.append((settings, kwargs)) or {"jsd": {}})
    remi = train_pipeline.load_config(RNN_REMI)
    note = train_pipeline.load_config(NGRAM_NOTE_TABLE)
    assert (remi.model, remi.representation) == ("rnn_remi", "remi")
    assert (note.model, note.representation) == ("ngram_note_table", "note_table")
    train_pipeline.train_pipeline(remi, run_name="same name")
    train_pipeline.train_pipeline(remi, run_name="same name")
    assert seen[0][0].output_dir != seen[1][0].output_dir
    assert all("same-name" in str(settings.output_dir) for settings, _ in seen)


def test_failed_construction_reports_failure(env):
    class Builder:
        def __init__(self, config):
            pass

        def build(self, names, overwrite=False):
            return [SimpleNamespace(representation=name, status=BuildStatus.FAILED,
                                    location="", message="", error="boom") for name in names]

    env.setattr(construction, "MusicDatasetBuilder", Builder)
    assert construction.main() == 1


def test_explicit_name_is_used_for_mlflow(env):
    env.setenv("MLFLOW_TRACKING_URI", "http://example.invalid")
    calls = []
    fake_mlflow = SimpleNamespace(
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=lambda run_name: calls.append(("run_name", run_name)) or nullcontext(
            SimpleNamespace(info=SimpleNamespace(run_id="remote-id"))
        ),
        set_tags=lambda tags: calls.append(("tags", tags)),
        log_params=lambda params: calls.append(("params", params)),
    )
    env.setitem(sys.modules, "mlflow", fake_mlflow)
    with tracking.start_run(train_pipeline.load_config(RNN_REMI), "named-run", "local-id"):
        pass
    assert ("run_name", "named-run") in calls
    assert any(name == "tags" and value["local_run_id"] == "local-id" for name, value in calls)

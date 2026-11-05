"""``main.py``: .env-only configuration, building representations, then training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from data_processing.music_representations.core.builder import BuildStatus

ENV_KEYS = (
    "MUSIC_REPR_INPUT_DIR",
    "MUSIC_REPR_OUTPUT_DIR",
    "MUSIC_REPR_REPRESENTATIONS",
    "MUSIC_REPR_OVERWRITE",
    "MUSIC_REPR_LIMIT",
    "MUSIC_REPR_CONFIG",
    "MUSIC_TRAIN_MODEL",
    "MUSIC_TRAIN_OUTPUT_DIR",
    "MUSIC_TRAIN_REPRESENTATION",
)


@pytest.fixture()
def env(monkeypatch, tmp_path: Path):
    """A clean environment; the real .env is never read."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(main, "DOTENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("MUSIC_REPR_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("MUSIC_REPR_OUTPUT_DIR", str(tmp_path / "prepared"))
    return monkeypatch


def _fake_builder(monkeypatch, calls: list, status=BuildStatus.SKIPPED):
    class Builder:
        def __init__(self, config):
            self.config = config

        def build(self, names, overwrite=False):
            calls.append(("build", list(names), overwrite))
            return [
                SimpleNamespace(
                    representation=name, status=status, location=None,
                    message="", error="boom",
                )
                for name in names
            ]

    monkeypatch.setattr(main, "MusicDatasetBuilder", Builder)


def test_dotenv_file_is_loaded_without_overriding_real_env(env, tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'MUSIC_REPR_REPRESENTATIONS="bpe, tsd"\nMUSIC_REPR_OVERWRITE=yes\n',
        encoding="utf-8",
    )
    env.setattr(main, "DOTENV_PATH", dotenv)
    env.setenv("MUSIC_REPR_OVERWRITE", "false")  # real env wins
    calls: list = []
    _fake_builder(env, calls)
    assert main.main() == 0
    assert calls == [("build", ["bpe", "tsd"], False)]


def test_build_only_when_no_model_is_configured(env, monkeypatch):
    calls: list = []
    _fake_builder(monkeypatch, calls)
    monkeypatch.setattr(main, "run", lambda _settings: pytest.fail("must not train"))
    assert main.main() == 0
    assert calls == [("build", ["all"], False)]


def test_builds_the_training_representation_then_trains(env, monkeypatch, tmp_path):
    env.setenv("MUSIC_REPR_REPRESENTATIONS", "remi")
    env.setenv("MUSIC_TRAIN_MODEL", "rnn_remi")
    env.setenv("MUSIC_TRAIN_OUTPUT_DIR", str(tmp_path / "runs"))
    env.setenv("MUSIC_TRAIN_REPRESENTATION", "bpe")
    calls: list = []
    _fake_builder(monkeypatch, calls)

    def fake_run(settings):
        calls.append(("train", settings.model, settings.representation))
        return {"jsd": {"pitch": 0.1}}

    monkeypatch.setattr(main, "run", fake_run)
    assert main.main() == 0
    assert calls == [("build", ["remi", "bpe"], False), ("train", "rnn_remi", "bpe")]


def test_failed_build_stops_before_training(env, monkeypatch, tmp_path):
    env.setenv("MUSIC_TRAIN_MODEL", "rnn_remi")
    env.setenv("MUSIC_TRAIN_OUTPUT_DIR", str(tmp_path / "runs"))
    calls: list = []
    _fake_builder(monkeypatch, calls, status=BuildStatus.FAILED)
    monkeypatch.setattr(main, "run", lambda _settings: pytest.fail("must not train"))
    assert main.main() == 1


def test_bad_training_settings_fail_before_building(env, monkeypatch):
    env.setenv("MUSIC_TRAIN_MODEL", "transformer_9000")
    calls: list = []
    _fake_builder(monkeypatch, calls)
    with pytest.raises(ValueError, match="MUSIC_TRAIN_MODEL"):
        main.main()
    assert calls == []


def test_missing_directories_are_reported(env):
    env.delenv("MUSIC_REPR_INPUT_DIR")
    with pytest.raises(ValueError, match="MUSIC_REPR_INPUT_DIR"):
        main.main()


def test_resolve_representations(env):
    assert main.resolve_representations("remi") == ["all"]
    env.setenv("MUSIC_REPR_REPRESENTATIONS", " remi , bpe ,")
    assert main.resolve_representations() == ["remi", "bpe"]
    assert main.resolve_representations("remi") == ["remi", "bpe"]
    assert main.resolve_representations("tsd") == ["remi", "bpe", "tsd"]

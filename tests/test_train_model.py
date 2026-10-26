from __future__ import annotations

import logging
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.helpers.canonical_frames import (
    build_canonical_frames,
)
from data_processing.music_representations.helpers.canonical_writer import (
    write_canonical_dataset,
)
from models.ngram.model import Model
from models.rnn.model import MusicRNN
from trainer import MODEL_NAMES, TrainSettings, pipeline
from trainer import settings as settings_module
from trainer.factory import RunFactory
from trainer.runs import NgramRun, RNNRun


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_no_argument_settings_select_each_model(monkeypatch, tmp_path: Path, model: str) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda _path: None)
    monkeypatch.setenv("MUSIC_TRAIN_MODEL", model)
    monkeypatch.setenv("MUSIC_TRAIN_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MUSIC_REPR_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("MUSIC_REPR_OUTPUT_DIR", str(tmp_path / "prepared"))
    settings = TrainSettings.from_env()
    assert settings.model == model
    assert settings.output_dir == tmp_path
    assert settings.generated_pieces == 250


@pytest.mark.parametrize(
    ("model", "run_type", "checkpoint_name"),
    [
        ("ngram_remi", NgramRun, "model.pkl"),
        ("ngram_note_table", NgramRun, "model.pkl"),
        ("rnn_remi", RNNRun, "model.pt"),
    ],
)
def test_run_factory_selects_model_and_checkpoint(
    tmp_path: Path, model: str, run_type: type, checkpoint_name: str
) -> None:
    settings = TrainSettings(model, tmp_path, 11, 1, 2, 16, 16, 3, 5, 2, 1)
    dataloader = SimpleNamespace()
    decoder = SimpleNamespace()
    trained = RunFactory.create(settings, dataloader, decoder)
    assert isinstance(trained, run_type)
    assert trained.checkpoint_name == checkpoint_name
    if model == "rnn_remi":
        assert trained.decoder is decoder
    else:
        assert trained.dataloader is dataloader


def test_run_factory_rejects_unknown_model(tmp_path: Path) -> None:
    settings = TrainSettings("ngram_remi", tmp_path, 11, 1, 2, 16, 16, 3, 5, 2, 1)
    with pytest.raises(ValueError, match="Unknown training model 'unknown'"):
        RunFactory.create(replace(settings, model="unknown"), SimpleNamespace(), None)


@pytest.mark.parametrize("representation", ["remi", "note_table"])
def test_ngram_training_and_checkpoint(
    tmp_path: Path, representation: str
) -> None:
    settings = TrainSettings(
        model="ngram_remi" if representation == "remi" else "ngram_note_table",
        output_dir=tmp_path,
        seed=11,
        generated_pieces=1,
        max_attempts=2,
        remi_max_tokens=32,
        note_max_tokens=32,
        remi_ngram_order=3,
        note_ngram_order=5,
        knn_neighbors=2,
        rnn_epochs=1,
    )
    if representation == "remi":
        song = {"tokens": [4, 5, 6, 7, 8, 9]}
    else:
        song = {
            "pitch": [60, 62, 64, 65, 67, 69],
            "velocity_bin": [10] * 6,
            "delta_onset_bin": [1] * 6,
            "duration_bin": [2] * 6,
        }
    loader = [[song]]
    dataloader = SimpleNamespace(
        load_music=object(),
        load_note_table_music=object(),
        loader=lambda **_kwargs: loader,
    )
    trained = NgramRun(dataloader, settings)
    trained.train()
    assert len(trained.model.models) == trained.model.n - 1
    baseline = Model(loader=loader, n=trained.model.n)
    if representation == "remi":
        baseline.remi_train()
    else:
        baseline.note_table_train()
    assert trained.model.models == baseline.models
    assert trained.generate(0, 0) is not None
    checkpoint = tmp_path / "model.pkl"
    trained.save(checkpoint)
    with checkpoint.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["representation"] == representation
    assert len(payload["counts"]) == trained.model.n - 1


def test_rnn_training_sampling_and_checkpoint(tmp_path: Path) -> None:
    settings = TrainSettings(
        model="rnn_remi",
        output_dir=tmp_path,
        seed=11,
        generated_pieces=1,
        max_attempts=2,
        remi_max_tokens=8,
        note_max_tokens=8,
        remi_ngram_order=3,
        note_ngram_order=5,
        knn_neighbors=2,
        rnn_epochs=1,
    )
    songs = [{"tokens": [4, 5] * 129} for _ in range(32)]
    class Loader:
        dataset = songs

        def __iter__(self):
            return iter([songs])

    dataloader = SimpleNamespace(loader=lambda **_kwargs: Loader(), load_music=object())

    class Tokenizer:
        def __init__(self):
            self.vocab = {f"Pitch_{index}": index for index in range(16)}

        def __len__(self):
            return len(self.vocab)

    trained = RNNRun(dataloader, SimpleNamespace(tokenizer=Tokenizer()), settings)
    trained.train()
    assert len(trained.epoch_losses) == 1
    assert len(trained.generate(0, 0)) <= settings.remi_max_tokens
    checkpoint = tmp_path / "model.pt"
    trained.save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["vocab_size"] == 16
    restored = MusicRNN(payload["vocab_size"])
    restored.load_state_dict(payload["state_dict"])


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_shared_workflow_uses_validation_and_writes_artifacts(
    monkeypatch, tmp_path: Path, caplog, model: str
) -> None:
    caplog.set_level(logging.INFO)
    settings = TrainSettings(
        model=model,
        output_dir=tmp_path / "runs",
        seed=11,
        generated_pieces=1,
        max_attempts=2,
        remi_max_tokens=16,
        note_max_tokens=16,
        remi_ngram_order=3,
        note_ngram_order=5,
        knn_neighbors=2,
        rnn_epochs=1,
    )
    config = MusicRepresentationConfig(tmp_path / "input", tmp_path / "prepared")
    canonical_root = config.output_dir / "canonical"

    def frames(piece_id: str, split: str):
        return build_canonical_frames(
            piece_id=piece_id,
            notes=[
                {"track_id": 0, "onset_tick": i * 240, "duration_tick": 120,
                 "pitch": 60 + i, "velocity": 80}
                for i in range(3)
            ],
            metadata={"maestro_split": split},
        )

    write_canonical_dataset(
        canonical_root,
        [frames("train", "train"), frames("valid", "validation"), frames("test", "test")],
        config=config,
        source_dataset_fingerprint="test-fingerprint",
    )
    remi_root = config.output_dir / "remi"
    remi_root.mkdir()
    (remi_root / "tokenizer.json").write_text("{}", encoding="utf-8")

    class Storage:
        def materialize(self, representation):
            return config.output_dir / representation

    class Dataloader:
        def __init__(self):
            self.storage = Storage()
            self.config = config
            self.train_df = pd.DataFrame({"piece_id": ["train"]})

        def prepare_dataset(self, representation):
            self.representation = representation

    class Contract:
        def describe_loss(self):
            return []

        def to_dict(self):
            return {}

    class Decoder:
        def __init__(self, representation):
            self.representation_dir = config.output_dir / representation
            self.config = config
            self.contract = Contract()

        def tuple_contract(self):
            return self.contract

        def decode(self, _tokens, context):
            return frames(context.piece_id, "generated")

    class Trained:
        def __init__(self, selected_model):
            self.epoch_losses = []
            self.checkpoint_name = (
                "model.pt" if selected_model == "rnn_remi" else "model.pkl"
            )

        def train(self):
            pass

        def generate(self, _piece_index, _attempt):
            return [1, 2, 3]

        def save(self, path):
            path.write_bytes(b"checkpoint")

    monkeypatch.setattr(pipeline, "RawDataloader", Dataloader)
    monkeypatch.setattr(
        pipeline, "create_decoder", lambda representation, _config, _storage: Decoder(representation)
    )
    monkeypatch.setattr(
        pipeline.RunFactory,
        "create",
        lambda selected_settings, _loader, _decoder: Trained(selected_settings.model),
    )
    result = pipeline.run(settings)
    assert result["reference_split"] == "validation"
    assert result["reference_pieces"] == 1
    assert result["generated_pieces"] == 1
    assert len(result["jsd"]) == 6
    assert (settings.output_dir / model / "metrics.json").is_file()
    assert (settings.output_dir / model / "generated_canonical" / "manifest.json").is_file()
    assert Path(result["checkpoint"]).is_file()
    assert Path(result["checkpoint"]).name == (
        "model.pt" if model == "rnn_remi" else "model.pkl"
    )
    assert "Prepare dataset started" in caplog.text
    assert "Generated 1 pieces" in caplog.text
    assert "Validation JSD scores" in caplog.text
    assert "Saved model to" in caplog.text


def test_pipeline_calls_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    settings = TrainSettings("ngram_remi", tmp_path, 11, 1, 2, 16, 16, 3, 5, 2, 1)
    events = []
    validation = SimpleNamespace(pieces=SimpleNamespace(height=2))
    generated_root = tmp_path / "generated_canonical"
    trained = object()

    def prepare(_settings):
        events.append("prepare_dataset")
        return object(), validation, object(), tmp_path / "canonical", "remi"

    def train_stage(_settings, _loader, _decoder):
        events.append("train")
        return trained

    def generate(*_args):
        events.append("generate_dataset")
        return generated_root

    def evaluate(_validation, _root):
        events.append("evaluate")
        return {"score": 0.0}, 1

    def save(*_args):
        events.append("save")
        return {"jsd": {"score": 0.0}}

    monkeypatch.setattr(pipeline, "prepare_dataset", prepare)
    monkeypatch.setattr(pipeline, "train", train_stage)
    monkeypatch.setattr(pipeline, "generate_dataset", generate)
    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    monkeypatch.setattr(pipeline, "save", save)
    assert pipeline.run(settings)["jsd"] == {"score": 0.0}
    assert events == [
        "prepare_dataset", "train", "generate_dataset", "evaluate", "save"
    ]


@pytest.mark.parametrize("failure_stage", ["generate_dataset", "evaluate"])
def test_failed_validation_does_not_save_checkpoint(
    monkeypatch, tmp_path: Path, caplog, failure_stage: str
) -> None:
    caplog.set_level(logging.INFO)
    settings = TrainSettings("ngram_remi", tmp_path, 11, 1, 2, 16, 16, 3, 5, 2, 1)
    validation = SimpleNamespace(pieces=SimpleNamespace(height=1))
    generated_root = tmp_path / "ngram_remi" / "generated_canonical"

    class Trained:
        checkpoint_name = "model.pkl"

        def save(self, path):
            path.write_bytes(b"checkpoint")

    def generate(*_args):
        if failure_stage == "generate_dataset":
            raise RuntimeError("generation failed")
        return generated_root

    def evaluate(_validation, _root):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(
        pipeline,
        "prepare_dataset",
        lambda _settings: (object(), validation, object(), tmp_path, "remi"),
    )
    monkeypatch.setattr(pipeline, "train", lambda *_args: Trained())
    monkeypatch.setattr(pipeline, "generate_dataset", generate)
    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.run(settings)
    assert not (tmp_path / "ngram_remi" / "model.pkl").exists()
    assert (
        "Generate dataset failed" if failure_stage == "generate_dataset" else "Evaluate failed"
    ) in caplog.text

from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import pytest

from data_processing.music_representations.adapters import (
    canonical_piece_to_muspy,
    canonical_piece_to_symusic,
)
from data_processing.music_representations.builder import BuildStatus
from data_processing.music_representations.canonical import CanonicalBuilder
from data_processing.music_representations.canonical_loader import (
    load_canonical_dataset,
)
from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.dataset_builder import MusicDatasetBuilder
from data_processing.music_representations.registry import (
    REPRESENTATION_NAMES,
    create_builder,
)
from data_processing.music_representations.segmentation import build_segments


def _build_canonical(config: MusicRepresentationConfig) -> Path:
    result = CanonicalBuilder(config).build()
    assert result.status is BuildStatus.BUILT
    return result.output_dir


def test_registry_contains_every_supported_representation() -> None:
    assert set(REPRESENTATION_NAMES) == {
        "canonical",
        "note_table",
        "piano_roll",
        "midilike",
        "tsd",
        "remi",
        "remi_plus",
        "structured",
        "cpword",
        "octuple",
        "pertok",
        "bpe",
        "unigram",
        "wordpiece",
    }


def test_canonical_timing_is_not_quantized_and_split_preserved(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = _build_canonical(base_config)
    dataset = load_canonical_dataset(canonical_dir)
    train_piece = dataset.pieces.filter(pl.col("maestro_split") == "train").row(
        0, named=True
    )
    note = dataset.notes.filter(pl.col("piece_id") == train_piece["piece_id"]).row(
        0, named=True
    )
    assert note["onset_tick"] == 123
    assert note["duration_tick"] == 457
    assert train_piece["maestro_split"] == "train"


def test_velocities_and_control_changes_survive_canonical_conversion(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = _build_canonical(base_config)
    dataset = load_canonical_dataset(canonical_dir)
    train_piece_id = dataset.pieces.filter(pl.col("maestro_split") == "train").row(
        0, named=True
    )["piece_id"]
    note = dataset.notes.filter(pl.col("piece_id") == train_piece_id).row(0, named=True)
    assert note["velocity"] == 93
    cc_numbers = set(dataset.control_changes.get_column("number").to_list())
    assert {64, 66}.issubset(cc_numbers)


def test_canonical_adapters_preserve_expected_events(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = _build_canonical(base_config)
    dataset = load_canonical_dataset(canonical_dir)
    piece_id = dataset.pieces.sort("piece_id").row(0, named=True)["piece_id"]
    score = canonical_piece_to_symusic(dataset, piece_id)
    music = canonical_piece_to_muspy(dataset, piece_id)
    assert score.note_num() == 1
    assert score.tracks[0].controls[0].number == 64
    assert len(music.tracks) == 1
    assert music.tracks[0].notes[0].pitch == 60


def test_segmentation_identifiers_are_stable(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = _build_canonical(base_config)
    dataset = load_canonical_dataset(canonical_dir)
    first = build_segments(dataset.pieces, dataset.notes, base_config)
    second = build_segments(dataset.pieces, dataset.notes, base_config)
    assert first.equals(second)


def test_subword_vocabulary_is_trained_only_on_training_split(
    base_config: MusicRepresentationConfig,
) -> None:
    results = MusicDatasetBuilder(base_config).build(["bpe"], overwrite=False)
    assert [item.representation for item in results] == ["canonical", "bpe"]
    manifest = json.loads(
        (base_config.output_dir / "bpe" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["subword_training_splits"] == ["train"]


def test_existing_output_folder_is_skipped_without_modification(
    base_config: MusicRepresentationConfig,
) -> None:
    result = CanonicalBuilder(base_config).build()
    assert result.status is BuildStatus.BUILT
    target = base_config.output_dir / "canonical" / "manifest.json"
    before = target.stat().st_mtime_ns
    time.sleep(0.01)
    skipped = CanonicalBuilder(base_config).build()
    after = target.stat().st_mtime_ns
    assert skipped.status is BuildStatus.SKIPPED
    assert before == after


def test_overwrite_rebuilds_only_requested_target(
    base_config: MusicRepresentationConfig,
) -> None:
    builder = MusicDatasetBuilder(base_config)
    first_results = builder.build(["canonical", "note_table"], overwrite=False)
    assert [item.status for item in first_results] == [
        BuildStatus.BUILT,
        BuildStatus.BUILT,
    ]
    canonical_manifest = base_config.output_dir / "canonical" / "manifest.json"
    note_manifest = base_config.output_dir / "note_table" / "manifest.json"
    canonical_before = canonical_manifest.stat().st_mtime_ns
    note_before = note_manifest.stat().st_mtime_ns
    time.sleep(0.01)
    second_results = builder.build(["note_table"], overwrite=True)
    canonical_after = canonical_manifest.stat().st_mtime_ns
    note_after = note_manifest.stat().st_mtime_ns
    assert second_results[0].representation == "note_table"
    assert canonical_before == canonical_after
    assert note_after > note_before


def test_failed_build_does_not_replace_valid_existing_directory(
    base_config: MusicRepresentationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = create_builder("note_table", base_config)
    canonical_result = CanonicalBuilder(base_config).build()
    assert canonical_result.status is BuildStatus.BUILT
    good = builder.build()
    assert good.status is BuildStatus.BUILT
    manifest_path = base_config.output_dir / "note_table" / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    def explode(_output_dir: Path) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(builder, "_build_into", explode)
    failed = builder.build(overwrite=True)
    assert failed.status is BuildStatus.FAILED
    assert manifest_path.read_text(encoding="utf-8") == original


def test_cli_selection_and_all_dependency_ordering(
    base_config: MusicRepresentationConfig,
) -> None:
    results = MusicDatasetBuilder(base_config).build(["all"], overwrite=False)
    assert results[0].representation == "canonical"
    assert results[0].status is BuildStatus.BUILT
    assert len(results) == len(REPRESENTATION_NAMES)

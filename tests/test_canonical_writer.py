from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.helpers.canonical_frames import (
    build_canonical_frames,
)
from data_processing.music_representations.helpers.canonical_loader import (
    canonical_dataset_from_frames,
    filter_canonical_dataset,
    load_canonical_dataset,
)
from data_processing.music_representations.helpers.canonical_writer import (
    canonical_dataset_fingerprint,
    write_canonical_dataset,
)
from data_processing.music_representations.schemas import TABLE_SCHEMAS

MELODIES = {
    "alpha": [60, 62, 64, 65],
    "beta": [48, 55, 64],
}


def _config(tmp_path: Path) -> MusicRepresentationConfig:
    # Deliberately not the shared ``base_config`` fixture: that one builds MIDI
    # files these tests never read. The writer only needs the config to render
    # ``to_dict()`` / ``config_hash()`` and to look up the canonical contract.
    return MusicRepresentationConfig(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output"
    )


def _frames(piece_id: str, pitches: list[int]) -> dict[str, pl.DataFrame]:
    return build_canonical_frames(
        piece_id=piece_id,
        notes=[
            {
                "onset_tick": index * 240,
                "duration_tick": 240,
                "pitch": pitch,
                "velocity": 80,
            }
            for index, pitch in enumerate(pitches)
        ],
        time_signatures=[{"time_tick": 0, "numerator": 4, "denominator": 4}],
        metadata={"maestro_split": "generated"},
    )


def _melody_frames() -> list[dict[str, pl.DataFrame]]:
    return [_frames(piece_id, pitches) for piece_id, pitches in MELODIES.items()]


def test_written_tables_match_the_canonical_schemas(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(root, _melody_frames(), config=_config(tmp_path))

    for table_name, schema in TABLE_SCHEMAS.items():
        written = pq.read_schema(root / f"{table_name}.parquet")
        assert written.equals(schema), table_name


def test_round_trips_through_load_canonical_dataset(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(root, _melody_frames(), config=_config(tmp_path))
    dataset = load_canonical_dataset(root)

    assert dataset.pieces.height == len(MELODIES)
    note_counts = dict(
        dataset.notes.group_by("piece_id").len().sort("piece_id").iter_rows()
    )
    assert note_counts == {
        piece_id: len(pitches) for piece_id, pitches in MELODIES.items()
    }
    assert dataset.manifest["representation"] == "canonical"


def test_empty_frames_write_eight_valid_empty_tables(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(root, [], config=_config(tmp_path))
    dataset = load_canonical_dataset(root)

    assert dataset.pieces.height == 0
    assert dataset.notes.height == 0
    assert dataset.manifest["pieces_per_split"] == {}
    assert len(sorted(root.glob("*.parquet"))) == len(TABLE_SCHEMAS)


def test_duplicate_piece_ids_are_rejected(tmp_path: Path) -> None:
    frames = [_frames("alpha", [60]), _frames("alpha", [62])]

    with pytest.raises(ValueError, match="alpha"):
        write_canonical_dataset(
            tmp_path / "generated", frames, config=_config(tmp_path)
        )


def test_missing_frame_key_is_reported_by_name(tmp_path: Path) -> None:
    incomplete = dict(_frames("alpha", [60]))
    del incomplete["tempos"]

    with pytest.raises(ValueError, match="tempos"):
        write_canonical_dataset(
            tmp_path / "generated", [incomplete], config=_config(tmp_path)
        )


def test_atomic_write_skips_an_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generated"
    root.mkdir()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    result = write_canonical_dataset(
        root, _melody_frames(), config=_config(tmp_path), atomic=True, overwrite=False
    )

    assert result is None
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert not (root / "pieces.parquet").exists()


def test_atomic_write_failure_leaves_no_target_directory(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    def exploding() -> Iterator[dict[str, pl.DataFrame]]:
        yield _frames("alpha", [60])
        raise RuntimeError("forced failure")

    with pytest.raises(RuntimeError, match="forced failure"):
        write_canonical_dataset(
            root, exploding(), config=_config(tmp_path), atomic=True
        )

    assert not root.exists()


def test_atomic_write_failure_leaves_an_existing_dataset_intact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generated"
    write_canonical_dataset(
        root, _melody_frames(), config=_config(tmp_path), atomic=True
    )
    original = (root / "manifest.json").read_text(encoding="utf-8")

    def exploding() -> Iterator[dict[str, pl.DataFrame]]:
        yield _frames("gamma", [72])
        raise RuntimeError("forced failure")

    with pytest.raises(RuntimeError, match="forced failure"):
        write_canonical_dataset(
            root, exploding(), config=_config(tmp_path), atomic=True, overwrite=True
        )

    assert (root / "manifest.json").read_text(encoding="utf-8") == original
    assert load_canonical_dataset(root).pieces.height == len(MELODIES)


def test_non_atomic_write_fills_an_existing_directory_in_place(tmp_path: Path) -> None:
    root = tmp_path / "generated"
    root.mkdir()
    sentinel = root / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    result = write_canonical_dataset(
        root, _melody_frames(), config=_config(tmp_path), atomic=False
    )

    assert result == root
    assert sentinel.exists()
    assert load_canonical_dataset(root).pieces.height == len(MELODIES)


def test_manifest_carries_extra_and_information_loss(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(
        root,
        _melody_frames(),
        config=_config(tmp_path),
        known_information_loss_extra=["velocities come from the model, not a score"],
        extra={"generated_by": {"model": "ngram", "n": 5}},
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_by"] == {"model": "ngram", "n": 5}
    assert (
        "velocities come from the model, not a score"
        in manifest["known_information_loss"]
    )


def test_fingerprint_defaults_to_the_written_pieces(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(root, _melody_frames(), config=_config(tmp_path))
    dataset = load_canonical_dataset(root)

    assert dataset.manifest[
        "source_dataset_fingerprint"
    ] == canonical_dataset_fingerprint(dataset.pieces)


def test_explicit_fingerprint_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "generated"

    write_canonical_dataset(
        root,
        _melody_frames(),
        config=_config(tmp_path),
        source_dataset_fingerprint="training-corpus-fingerprint",
    )

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_dataset_fingerprint"] == "training-corpus-fingerprint"


def test_canonical_dataset_from_frames_matches_the_on_disk_round_trip(
    tmp_path: Path,
) -> None:
    root = tmp_path / "generated"
    write_canonical_dataset(root, _melody_frames(), config=_config(tmp_path))

    from_disk = load_canonical_dataset(root)
    in_memory = canonical_dataset_from_frames(_melody_frames(), root=root)

    for table_name in TABLE_SCHEMAS:
        assert getattr(in_memory, table_name).equals(
            getattr(from_disk, table_name)
        ), table_name


def test_canonical_dataset_from_frames_handles_no_pieces(tmp_path: Path) -> None:
    dataset = canonical_dataset_from_frames([], root=tmp_path)

    assert dataset.manifest == {}
    for table_name in TABLE_SCHEMAS:
        assert getattr(dataset, table_name).height == 0


def test_filter_canonical_dataset_subsets_every_table(tmp_path: Path) -> None:
    dataset = canonical_dataset_from_frames(_melody_frames(), root=tmp_path)

    filtered = filter_canonical_dataset(dataset, ["alpha"])

    assert filtered.pieces.get_column("piece_id").to_list() == ["alpha"]
    assert filtered.notes.height == len(MELODIES["alpha"])
    assert filtered.tracks.get_column("piece_id").unique().to_list() == ["alpha"]


def test_filter_canonical_dataset_with_an_empty_selection(tmp_path: Path) -> None:
    dataset = canonical_dataset_from_frames(_melody_frames(), root=tmp_path)

    filtered = filter_canonical_dataset(dataset, [])

    for table_name in TABLE_SCHEMAS:
        assert getattr(filtered, table_name).height == 0, table_name
    assert filtered.root == tmp_path
    assert filtered.manifest == dataset.manifest

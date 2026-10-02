from __future__ import annotations

import polars as pl
import python3_midi

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.helpers.canonical_loader import (
    load_canonical_dataset,
)
from data_processing.music_representations.representations.canonical import (
    CanonicalBuilder,
)
from evaluation.parsers import MGEvalAdapter, MusPyAdapter


def test_muspy_adapter_converts_all_pieces(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = CanonicalBuilder(base_config).build().output_dir
    canonical = load_canonical_dataset(canonical_dir)

    converted = MusPyAdapter().convert(canonical)

    expected_ids = canonical.pieces.get_column("piece_id").to_list()
    assert list(converted.pieces) == expected_ids
    assert len(converted) == len(expected_ids)
    assert converted[0] is converted.pieces[converted.piece_id(0)]
    assert converted[0].tracks[0].notes[0].pitch == 60


def test_mgeval_adapter_builds_both_midi_views(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical_dir = CanonicalBuilder(base_config).build().output_dir
    canonical = load_canonical_dataset(canonical_dir)
    train_piece_id = canonical.pieces.filter(pl.col("maestro_split") == "train").item(
        0, "piece_id"
    )

    converted = MGEvalAdapter().convert(canonical)
    piece = converted.pieces[train_piece_id]

    assert len(converted) == canonical.pieces.height
    assert piece["pretty_midi"].resolution == 480
    assert piece["pretty_midi"].instruments[0].notes[0].pitch == 60
    # MGEval reads the pattern view, so it must be a python3-midi Pattern whose
    # track 0 is metadata and whose notes live on a later track.
    assert isinstance(piece["midi_pattern"], python3_midi.Pattern)
    assert piece["midi_pattern"].resolution == 480
    assert len(piece["midi_pattern"]) >= 2

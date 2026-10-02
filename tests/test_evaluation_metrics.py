from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.core.canonical_data import CanonicalDataset
from data_processing.music_representations.helpers.canonical_frames import (
    build_canonical_frames,
)
from data_processing.music_representations.helpers.canonical_loader import (
    canonical_dataset_from_frames,
    filter_canonical_dataset,
    load_canonical_dataset,
)
from data_processing.music_representations.representations.canonical import (
    CanonicalBuilder,
)
from evaluation.metrics import DatasetEvaluation, MGEvalMetric, MusPyMetrics
from evaluation.parsers import MGEvalAdapter, MusPyAdapter

MUSPY_COLUMNS = (
    "polyphony_rate",
    "scale_consistency",
)
MGEVAL_COLUMNS = (
    "pitch_class_distribution",
    "avg_pitch_shift",
    "avg_IOI",
    "note_length_hist",
)
EVALUATED_METRICS = {
    "avg_pitch_shift",
    "avg_IOI",
    "polyphony_rate",
    "scale_consistency",
    "pitch_class_distribution",
    "note_length_hist",
}

# The shared MAESTRO fixture gives every piece a single note, which leaves the
# interval-based metrics undefined, so the metric tests build their own pieces.
MELODIES = {
    "c_major": [60, 62, 64, 65, 67, 69, 71, 72],
    "arpeggio": [48, 55, 64, 72, 64, 55, 48, 55],
    "chromatic": [60, 61, 62, 63, 64, 65, 66, 67],
}


def _piece_frames(piece_id: str, pitches: list[int]) -> dict[str, pl.DataFrame]:
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
        metadata={"maestro_split": "test"},
    )


def _melodies(tmp_path: Path) -> CanonicalDataset:
    return canonical_dataset_from_frames(
        [_piece_frames(piece_id, pitches) for piece_id, pitches in MELODIES.items()],
        root=tmp_path,
    )


def _joined_metrics(canonical: CanonicalDataset) -> pl.DataFrame:
    muspy_metrics = MusPyMetrics().compute(MusPyAdapter().convert(canonical))
    mgeval_metrics = MGEvalMetric().compute(MGEvalAdapter().convert(canonical))
    return muspy_metrics.join(mgeval_metrics, on="piece_id", how="inner")


def test_muspy_metrics_return_one_row_per_piece(
    base_config: MusicRepresentationConfig,
) -> None:
    canonical = load_canonical_dataset(CanonicalBuilder(base_config).build().output_dir)

    metrics = MusPyMetrics().compute(MusPyAdapter().convert(canonical))

    expected_ids = canonical.pieces.get_column("piece_id").to_list()
    assert metrics.get_column("piece_id").to_list() == expected_ids
    assert metrics.columns == ["piece_id", *MUSPY_COLUMNS]


def test_mgeval_metrics_return_fixed_length_histograms(tmp_path: Path) -> None:
    canonical = _melodies(tmp_path)

    metrics = MGEvalMetric().compute(MGEvalAdapter().convert(canonical))

    assert metrics.get_column("piece_id").to_list() == list(MELODIES)
    assert metrics.columns == ["piece_id", *MGEVAL_COLUMNS]
    for row in metrics.iter_rows(named=True):
        assert len(row["pitch_class_distribution"]) == 12
        assert len(row["note_length_hist"]) == 12
        assert not math.isnan(row["avg_pitch_shift"])
        assert not math.isnan(row["avg_IOI"])


def test_dataset_evaluation_of_a_dataset_against_itself_is_zero(
    tmp_path: Path,
) -> None:
    metrics = _joined_metrics(_melodies(tmp_path))

    results = DatasetEvaluation(metrics, metrics).evaluate()

    assert set(results) == EVALUATED_METRICS
    for name, value in results.items():
        assert value == 0.0, f"{name} should be identical to itself, got {value}"


def test_dataset_evaluation_of_disjoint_subsets_stays_in_unit_range(
    tmp_path: Path,
) -> None:
    canonical = _melodies(tmp_path)
    first = filter_canonical_dataset(canonical, ["c_major"])
    rest = filter_canonical_dataset(canonical, ["arpeggio", "chromatic"])
    assert first.notes.height and rest.notes.height

    results = DatasetEvaluation(
        _joined_metrics(first), _joined_metrics(rest)
    ).evaluate()

    assert set(results) == EVALUATED_METRICS
    for name, value in results.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

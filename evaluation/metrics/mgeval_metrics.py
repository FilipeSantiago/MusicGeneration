import polars as pl

from evaluation.model import MGDataset
from evaluation.vendor.mgeval_core import Feature, MgevalMetrics

# The two MIDI views an MGEval feature carries are indexed differently. The
# PrettyMIDI view holds one instrument per track, so the notes are instrument 0.
# The pattern view comes from Mido, whose track 0 is the metadata track, which
# puts the notes on track 1.
PRETTY_MIDI_TRACK = 0
PATTERN_TRACK = 1


class MGEvalMetric:
    name = "mgeval"

    def compute(self, dataset: MGDataset) -> pl.DataFrame:
        metrics = MgevalMetrics()
        rows = []

        for piece_id, piece in dataset.pieces.items():
            feature = Feature(piece["pretty_midi"], piece["midi_pattern"])
            rows.append(
                {
                    "piece_id": piece_id,
                    "pitch_class_distribution": metrics.total_pitch_class_histogram(
                        feature, PRETTY_MIDI_TRACK
                    ),
                    "avg_pitch_shift": metrics.avg_pitch_shift(feature, PATTERN_TRACK),
                    "avg_IOI": metrics.avg_IOI(feature, PRETTY_MIDI_TRACK),
                    "note_length_hist": metrics.note_length_hist(
                        feature, track_num=PATTERN_TRACK
                    ),
                }
            )

        return pl.DataFrame(rows)

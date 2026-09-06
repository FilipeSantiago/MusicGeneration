import polars as pl
from evaluation.model import MGDataset


class MGEvalMetric:
    name = "average_pitch_interval"

    def compute(self, dataset: MGDataset) -> pl.DataFrame:
        rows = []

        for piece_id, feature in dataset.pieces.items():
            rows.append(
                {
                    "piece_id": piece_id,
                    "pitch_count": feature.total_used_pitch(),
                    "pitch_class_distribution": feature.total_pitch_class_histogram(),
                    "pitch_class_transitions": feature.pitch_class_transition_matrix(),
                    "pitch_range": feature.pitch_range(),
                    "avg_pitch_shift": feature.avg_pitch_shift(),
                    "note_count": feature.total_used_note(),
                    "avg_IOI": feature.avg_IOI(),
                    "note_length_hist": feature.note_length_hist(),
                    "duration_transitions": feature.note_length_transition_matrix(),
                }
            )

        return pl.DataFrame(rows)
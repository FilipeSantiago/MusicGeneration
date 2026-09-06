import muspy
import polars as pl

from evaluation.model.muspy_dataset import MusPyDataset


class MusPyMetrics:

    def compute(
        self,
        dataset: MusPyDataset,
    ) -> pl.DataFrame:

        return pl.DataFrame(
            {
                "piece_id": piece_id,
                "pitch_range": muspy.pitch_range(music),
                "pitch_entropy": muspy.pitch_entropy(music),
                "polyphony": muspy.polyphony(music),
                "polyphony_rate": muspy.polyphony_rate(music),
                "scale_consistency": muspy.scale_consistency(music),
            }
            for piece_id, music in dataset.pieces.items()
        )

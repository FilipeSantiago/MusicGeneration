from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon


class DatasetEvaluation:

    def __init__(self, dataset1_metrics, dataset2_metrics):
        self.dataset1_metrics = dataset1_metrics
        self.dataset2_metrics = dataset2_metrics

    def evaluate(self) -> dict[str, float]:
        scalar_variables = ["avg_pitch_shift", "avg_IOI", "polyphony_rate", "scale_consistency"]
        histogram_variables = ["pitch_class_distribution", "note_length_hist"]

        results = {}

        for column in scalar_variables:
            results[column] = self.__scalar_jsd__(self.dataset1_metrics[column], self.dataset2_metrics[column], column)
        for column in histogram_variables:
            results[column] = self.__histogram_jsd__(self.dataset1_metrics[column], self.dataset2_metrics[column], column)
        return results

    @staticmethod
    def __scalar_jsd__(
        reference: pl.DataFrame,
        generated: pl.DataFrame,
        column: str,
        bins: int = 30,
    ) -> float:

        ref = reference[column].to_numpy()
        gen = generated[column].to_numpy()

        edges = np.histogram_bin_edges(
            np.concatenate([ref, gen]),
            bins=bins,
        )

        ref_hist, _ = np.histogram(ref, bins=edges)
        gen_hist, _ = np.histogram(gen, bins=edges)

        return jensenshannon(ref_hist, gen_hist, base=2,) ** 2

    @staticmethod
    def __histogram_jsd__(
            reference: pl.DataFrame,
            generated: pl.DataFrame,
            column: str,
    ) -> float:

        ref = np.stack(reference[column].to_list()).sum(axis=0)
        gen = np.stack(generated[column].to_list()).sum(axis=0)

        return jensenshannon(
            ref,
            gen,
            base=2,
        ) ** 2
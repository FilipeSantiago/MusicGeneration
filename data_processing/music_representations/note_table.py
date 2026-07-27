from __future__ import annotations

import polars as pl

from .base import BaseRepresentationBuilder
from .manifest import write_manifest
from .segmentation import build_segments


class NoteTableBuilder(BaseRepresentationBuilder):
    """Export a model-friendly note table from the canonical dataset."""

    representation_name = "note_table"

    def _build_into(self, output_dir):
        dataset = self.load_canonical()
        segments = build_segments(dataset.pieces, dataset.notes, self.config)
        notes = dataset.notes.join(
            dataset.tracks, on=["piece_id", "track_id"], how="left"
        ).join(
            dataset.pieces.select(["piece_id", "maestro_split"]),
            on="piece_id",
            how="left",
        )
        if self.config.segmentation.enabled:
            notes = notes.join(
                segments, on=["piece_id", "maestro_split"], how="inner"
            ).filter(
                (pl.col("onset_tick") < pl.col("end_tick"))
                & (
                    (pl.col("onset_tick") + pl.col("duration_tick"))
                    > pl.col("start_tick")
                )
            )
        else:
            notes = notes.with_columns(
                pl.lit(None, dtype=pl.String).alias("segment_id")
            )
        note_table = notes.select(
            [
                "piece_id",
                "segment_id",
                "maestro_split",
                pl.col("pitch"),
                pl.col("velocity"),
                pl.col("onset_tick").alias("onset"),
                pl.col("duration_tick").alias("duration"),
                "track_id",
                "program",
                "is_drum",
                "track_name",
            ]
        )
        note_table.write_parquet(output_dir / "notes.parquet")
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=dataset.manifest[
                "source_canonical_schema_version"
            ],
            source_dataset_fingerprint=dataset.manifest["source_dataset_fingerprint"],
            config=self.config,
            pieces=dataset.pieces,
            segments=segments,
            extra={"time_unit": self.config.note_table.time_unit},
        )

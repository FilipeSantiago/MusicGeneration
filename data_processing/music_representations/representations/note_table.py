from __future__ import annotations

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from ..core.base import BaseRepresentationBuilder
from ..helpers.canonical_loader import (
    iter_canonical_pieces,
    load_canonical_manifest,
    load_canonical_pieces,
)
from ..helpers.segmentation import build_piece_segments
from ..manifest import write_manifest


class NoteTableBuilder(BaseRepresentationBuilder):
    """Export a model-friendly note table from the canonical dataset."""

    representation_name = "note_table"

    def _build_into(self, output_dir):
        bin_size = self.config.note_table.bin_size
        pieces = load_canonical_pieces(self.canonical_dir)
        manifest = load_canonical_manifest(self.canonical_dir)
        segment_rows: list[dict[str, object]] = []
        writer = pq.ParquetWriter(
            output_dir / "notes.parquet", NOTE_TABLE_SCHEMA, compression="zstd"
        )
        try:
            for piece_data in tqdm(
                iter_canonical_pieces(self.canonical_dir),
                total=pieces.height,
                desc="note_table",
                unit="piece",
                leave=False,
            ):
                piece = piece_data.piece
                segments = build_piece_segments(piece, piece_data.notes, self.config)
                segment_rows.extend(segments)
                notes = piece_data.notes.join(
                    piece_data.tracks, on=["piece_id", "track_id"], how="left"
                ).with_columns(
                    pl.lit(piece["maestro_split"]).alias("maestro_split")
                )
                if self.config.segmentation.enabled:
                    piece_segments = pl.DataFrame(segments)
                    notes = notes.join(
                        piece_segments, on=["piece_id", "maestro_split"], how="inner"
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
                note_table = (
                    notes.select(
                        [
                            "piece_id",
                            "segment_id",
                            "maestro_split",
                            pl.col("pitch"),
                            pl.col("velocity"),
                            pl.col("onset_quarter").alias("onset"),
                            pl.col("duration_quarter").alias("duration"),
                            "track_id",
                            "program",
                            "is_drum",
                            "track_name",
                        ]
                    )
                    .sort(["piece_id", "segment_id", "onset"])
                    .with_columns(
                        pl.col("onset")
                        .diff()
                        .over(["piece_id", "segment_id"])
                        .fill_null(0.0)
                        .alias("delta_onset")
                    )
                    .with_columns(
                        (pl.col("duration") / bin_size)
                        .round(0)
                        .cast(pl.Int32)
                        .alias("duration_bin"),
                        (pl.col("delta_onset") / bin_size)
                        .round(0)
                        .cast(pl.Int32)
                        .alias("delta_onset_bin"),
                    )
                    .drop(["duration", "delta_onset"])
                )
                writer.write_table(
                    note_table.to_arrow()
                    .select(NOTE_TABLE_SCHEMA.names)
                    .cast(NOTE_TABLE_SCHEMA)
                )
        finally:
            writer.close()
        segments_df = pl.DataFrame(segment_rows) if segment_rows else pl.DataFrame(schema=SEGMENTS_SCHEMA)
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=manifest["source_canonical_schema_version"],
            source_dataset_fingerprint=manifest["source_dataset_fingerprint"],
            config=self.config,
            pieces=pieces,
            segments=segments_df,
            extra={
                "time_unit": self.config.note_table.time_unit,
                "bin_size": bin_size,
            },
        )


NOTE_TABLE_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("segment_id", pa.string()),
        ("maestro_split", pa.string()),
        ("pitch", pa.uint8()),
        ("velocity", pa.uint8()),
        ("onset", pa.float64()),
        ("delta_onset_bin", pa.int32()),
        ("duration_bin", pa.int32()),
        ("track_id", pa.int16()),
        ("program", pa.int16()),
        ("is_drum", pa.bool_()),
        ("track_name", pa.string()),
    ]
)

SEGMENTS_SCHEMA = [
    ("piece_id", pl.String),
    ("segment_id", pl.String),
    ("maestro_split", pl.String),
    ("start_tick", pl.Int64),
    ("end_tick", pl.Int64),
    ("configuration_hash", pl.String),
]

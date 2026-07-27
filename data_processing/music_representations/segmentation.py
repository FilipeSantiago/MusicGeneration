from __future__ import annotations

from hashlib import sha256

import polars as pl

from .config import MusicRepresentationConfig


def build_segments(
    pieces: pl.DataFrame, notes: pl.DataFrame, config: MusicRepresentationConfig
) -> pl.DataFrame:
    if not config.segmentation.enabled:
        return pieces.select(
            [
                "piece_id",
                pl.lit(None, dtype=pl.String).alias("segment_id"),
                "maestro_split",
                pl.lit(0).alias("start_tick"),
                pl.col("end_tick").alias("end_tick"),
                pl.lit(config.config_hash()).alias("configuration_hash"),
            ]
        )

    segment_ticks = config.segmentation.segment_ticks
    hop_ticks = config.segmentation.hop_ticks or segment_ticks
    if segment_ticks is None:
        raise ValueError("Segmentation enabled but segment_ticks is missing")
    assert hop_ticks is not None
    hop_ticks = int(hop_ticks)

    segments: list[dict[str, object]] = []
    notes_by_piece = notes.partition_by("piece_id", as_dict=True)
    for piece in pieces.iter_rows(named=True):
        piece_id = piece["piece_id"]
        end_tick = int(piece["end_tick"])
        current = 0
        piece_notes = notes_by_piece.get((piece_id,), pl.DataFrame(schema=notes.schema))
        while current < end_tick or (current == 0 and end_tick == 0):
            segment_end = min(current + segment_ticks, end_tick)
            if segment_end <= current:
                break
            in_segment = piece_notes.filter(
                (pl.col("onset_tick") < segment_end)
                & ((pl.col("onset_tick") + pl.col("duration_tick")) > current)
            )
            if in_segment.height >= config.segmentation.min_notes:
                stable_key = (
                    f"{piece_id}:{current}:{segment_end}:{config.config_hash()}"
                )
                segments.append(
                    {
                        "piece_id": piece_id,
                        "segment_id": sha256(stable_key.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "maestro_split": piece["maestro_split"],
                        "start_tick": current,
                        "end_tick": segment_end,
                        "configuration_hash": config.config_hash(),
                    }
                )
            if (
                segment_end == end_tick
                and not config.segmentation.allow_incomplete_final_segment
            ):
                break
            current += hop_ticks
            if current >= end_tick:
                break
    return pl.DataFrame(segments)

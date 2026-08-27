from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ..contracts import RoundTripContract, note_table_tuple_contract
from ..helpers.canonical_frames import build_canonical_frames
from .base import DecodeContext, RepresentationDecoder

TUPLE_FIELDS = ("pitch", "velocity_bin", "delta_onset_bin", "duration_bin")


class NoteTableDecoder(RepresentationDecoder):
    """Decode the note table, either from its parquet rows or from model output.

    The two forms are not equally faithful: the stored table keeps an exact float
    `onset` and an exact `velocity`, while the `(pitch, velocity_bin,
    delta_onset_bin, duration_bin)` tuples the n-gram model emits keep neither.
    :meth:`tuple_contract` states the weaker promise.
    """

    representation_name = "note_table"

    def tuple_contract(self) -> RoundTripContract:
        return note_table_tuple_contract(self.config)

    def decode(
        self, payload: Any, context: DecodeContext | None = None
    ) -> dict[str, pl.DataFrame]:
        context = context or DecodeContext()
        if isinstance(payload, pl.DataFrame):
            return self._decode_frame(payload, context)
        return self.decode_tuples(payload, context)

    def load_payload(
        self, *, piece_id: str, segment_id: str | None = None
    ) -> pl.DataFrame:
        dataset = ds.dataset(self.representation_dir / "notes.parquet", format="parquet")
        column = "segment_id" if segment_id is not None else "piece_id"
        value = segment_id if segment_id is not None else piece_id
        return pl.from_arrow(dataset.to_table(filter=pc.field(column) == value))

    def decode_tuples(
        self,
        rows: Sequence[Any],
        context: DecodeContext | None = None,
        *,
        fields: tuple[str, ...] = TUPLE_FIELDS,
    ) -> dict[str, pl.DataFrame]:
        """Rebuild canonical frames from the flat note tuples a model emits.

        Non-numeric tuples (the `<BOS>` / `<EOF>` sentinels) are dropped rather than
        raising, so generated sequences can be passed in verbatim.
        """
        context = context or DecodeContext()
        tpq = context.ticks_per_quarter
        bin_size = self.config.note_table.bin_size
        velocity_bin_size = self.config.note_table.velocity_bin_size

        notes: list[dict[str, Any]] = []
        onset_quarters = 0.0
        for row in rows:
            values = self._as_field_map(row, fields)
            if values is None:
                continue
            onset_quarters += values.get("delta_onset_bin", 0) * bin_size
            if "velocity" in values:
                velocity = int(values["velocity"])
            else:
                velocity = (
                    int(values.get("velocity_bin", 0)) * velocity_bin_size
                    + velocity_bin_size // 2
                )
            notes.append(
                {
                    "track_id": 0,
                    "onset_tick": context.start_tick + round(onset_quarters * tpq),
                    "duration_tick": max(
                        1, round(values.get("duration_bin", 0) * bin_size * tpq)
                    ),
                    "pitch": int(values["pitch"]),
                    "velocity": min(127, max(1, velocity)),
                }
            )

        return build_canonical_frames(
            piece_id=context.piece_id,
            ticks_per_quarter=tpq,
            tracks=[
                {
                    "track_id": 0,
                    "track_name": context.track_name,
                    "program": context.program,
                    "is_drum": context.is_drum,
                }
            ],
            notes=notes,
            metadata=context.metadata,
            default_qpm=context.default_qpm,
        )

    def _decode_frame(
        self, frame: pl.DataFrame, context: DecodeContext
    ) -> dict[str, pl.DataFrame]:
        tpq = context.ticks_per_quarter
        bin_size = self.config.note_table.bin_size
        notes = [
            {
                "track_id": int(row["track_id"]),
                "onset_tick": round(float(row["onset"]) * tpq),
                "duration_tick": max(1, round(int(row["duration_bin"]) * bin_size * tpq)),
                "pitch": int(row["pitch"]),
                "velocity": int(row["velocity"]),
            }
            for row in frame.iter_rows(named=True)
        ]
        tracks = [
            {
                "track_id": int(row["track_id"]),
                "track_name": row["track_name"] or "",
                "program": int(row["program"]),
                "is_drum": bool(row["is_drum"]),
            }
            for row in frame.select(
                ["track_id", "track_name", "program", "is_drum"]
            ).unique().sort("track_id").iter_rows(named=True)
        ]
        return build_canonical_frames(
            piece_id=context.piece_id,
            ticks_per_quarter=tpq,
            tracks=tracks or None,
            notes=notes,
            metadata=context.metadata,
            default_qpm=context.default_qpm,
        )

    @staticmethod
    def _as_field_map(row: Any, fields: tuple[str, ...]) -> dict[str, float] | None:
        if isinstance(row, dict):
            values = {name: row[name] for name in fields if name in row}
        else:
            values = dict(zip(fields, row, strict=False))
        if "pitch" not in values:
            return None
        try:
            return {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError):
            return None  # BOS / EOF sentinels and anything else non-numeric

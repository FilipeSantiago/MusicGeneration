from __future__ import annotations

import pyarrow as pa

PIECES_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("source_midi", pa.string()),
        ("source_sha256", pa.string()),
        ("composer", pa.string()),
        ("title", pa.string()),
        ("maestro_split", pa.string()),
        ("ticks_per_quarter", pa.int32()),
        ("end_tick", pa.int64()),
        ("duration_sec", pa.float64()),
        ("schema_version", pa.string()),
    ]
)

TRACKS_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("track_id", pa.int16()),
        ("track_name", pa.string()),
        ("program", pa.int16()),
        ("is_drum", pa.bool_()),
    ]
)

NOTES_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("track_id", pa.int16()),
        ("note_id", pa.int32()),
        ("onset_tick", pa.int64()),
        ("duration_tick", pa.int64()),
        ("onset_quarter", pa.float64()),
        ("duration_quarter", pa.float64()),
        ("onset_sec", pa.float64()),
        ("duration_sec", pa.float64()),
        ("pitch", pa.uint8()),
        ("velocity", pa.uint8()),
    ]
)

TEMPOS_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("time_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("qpm", pa.float64()),
    ]
)

TIME_SIGNATURES_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("time_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("numerator", pa.uint8()),
        ("denominator", pa.uint8()),
    ]
)

KEY_SIGNATURES_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("time_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("key", pa.int8()),
        ("tonality", pa.int8()),
    ]
)

CONTROL_CHANGES_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("track_id", pa.int16()),
        ("time_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("number", pa.uint8()),
        ("value", pa.int32()),
    ]
)

PITCH_BENDS_SCHEMA = pa.schema(
    [
        ("piece_id", pa.string()),
        ("track_id", pa.int16()),
        ("time_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("value", pa.int32()),
    ]
)


TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "pieces": PIECES_SCHEMA,
    "tracks": TRACKS_SCHEMA,
    "notes": NOTES_SCHEMA,
    "tempos": TEMPOS_SCHEMA,
    "time_signatures": TIME_SIGNATURES_SCHEMA,
    "key_signatures": KEY_SIGNATURES_SCHEMA,
    "control_changes": CONTROL_CHANGES_SCHEMA,
    "pitch_bends": PITCH_BENDS_SCHEMA,
}

TABLE_NAMES = tuple(TABLE_SCHEMAS)

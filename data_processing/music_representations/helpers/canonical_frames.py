from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import polars as pl
import pyarrow as pa
from symusic import Score

from ..schemas import TABLE_SCHEMAS
from .adapters import canonical_frames_to_symusic

DEFAULT_TICKS_PER_QUARTER = 480
DEFAULT_QPM = 120.0
DEFAULT_VELOCITY = 64

PIECE_METADATA_FIELDS = (
    "source_midi",
    "source_sha256",
    "composer",
    "title",
    "maestro_split",
    "schema_version",
)


def build_canonical_frames(
    *,
    piece_id: str,
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER,
    notes: Iterable[dict[str, Any]],
    tracks: Iterable[dict[str, Any]] | None = None,
    tempos: Iterable[dict[str, Any]] = (),
    time_signatures: Iterable[dict[str, Any]] = (),
    key_signatures: Iterable[dict[str, Any]] = (),
    control_changes: Iterable[dict[str, Any]] = (),
    pitch_bends: Iterable[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
    default_qpm: float = DEFAULT_QPM,
) -> dict[str, pl.DataFrame]:
    """Build schema-valid canonical frames from tick-level rows.

    Every producer of canonical data goes through here, so the derived columns
    (``*_quarter`` and ``*_sec``) are computed in exactly one place. Seconds are
    read back from ``symusic`` rather than from a hand-rolled tempo map, which is
    what makes them agree with the values written when parsing MIDI directly.
    """
    tpq = int(ticks_per_quarter)
    if tpq <= 0:
        raise ValueError(f"ticks_per_quarter must be > 0, got {ticks_per_quarter}")

    note_rows = _note_rows(piece_id, notes)
    control_rows = _track_event_rows(
        piece_id, control_changes, ("number", "value"), defaults={"number": 64, "value": 0}
    )
    bend_rows = _track_event_rows(piece_id, pitch_bends, ("value",), defaults={"value": 0})
    track_rows = _track_rows(piece_id, tracks, note_rows, control_rows, bend_rows)

    tempo_rows = _global_event_rows(piece_id, tempos, ("qpm",), defaults={"qpm": default_qpm})
    if not tempo_rows:
        tempo_rows = [{"piece_id": piece_id, "time_tick": 0, "qpm": float(default_qpm)}]
    time_signature_rows = _global_event_rows(
        piece_id,
        time_signatures,
        ("numerator", "denominator"),
        defaults={"numerator": 4, "denominator": 4},
    )
    key_signature_rows = _global_event_rows(
        piece_id, key_signatures, ("key", "tonality"), defaults={"key": 0, "tonality": 0}
    )

    # Cast through the schemas so empty tables still carry their columns.
    tick_frames = {
        "piece": _frame([{"piece_id": piece_id, "ticks_per_quarter": tpq}], "pieces"),
        "tracks": _frame(track_rows, "tracks"),
        "notes": _frame(note_rows, "notes"),
        "tempos": _frame(tempo_rows, "tempos"),
        "time_signatures": _frame(time_signature_rows, "time_signatures"),
        "key_signatures": _frame(key_signature_rows, "key_signatures"),
        "control_changes": _frame(control_rows, "control_changes"),
        "pitch_bends": _frame(bend_rows, "pitch_bends"),
    }
    score = canonical_frames_to_symusic(tick_frames)
    _fill_seconds(score, tpq, note_rows, control_rows, bend_rows, tempo_rows,
                  time_signature_rows, key_signature_rows)

    piece_row: dict[str, Any] = {
        "piece_id": piece_id,
        "ticks_per_quarter": tpq,
        "end_tick": int(score.end()),
        "duration_sec": float(score.to("second").end()),
    }
    for field in PIECE_METADATA_FIELDS:
        piece_row[field] = (metadata or {}).get(field)

    return {
        "piece": _frame([piece_row], "pieces"),
        "tracks": _frame(track_rows, "tracks"),
        "notes": _frame(note_rows, "notes"),
        "tempos": _frame(tempo_rows, "tempos"),
        "time_signatures": _frame(time_signature_rows, "time_signatures"),
        "key_signatures": _frame(key_signature_rows, "key_signatures"),
        "control_changes": _frame(control_rows, "control_changes"),
        "pitch_bends": _frame(bend_rows, "pitch_bends"),
    }


def symusic_to_canonical_frames(
    score: Score,
    *,
    piece_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, pl.DataFrame]:
    """Inverse of :func:`~..helpers.adapters.canonical_frames_to_symusic`."""
    tpq = int(score.ticks_per_quarter)
    tracks: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    control_changes: list[dict[str, Any]] = []
    pitch_bends: list[dict[str, Any]] = []

    for track_id, track in enumerate(score.tracks):
        tracks.append(
            {
                "track_id": track_id,
                "track_name": track.name,
                "program": track.program,
                "is_drum": track.is_drum,
            }
        )
        for note in track.notes:
            notes.append(
                {
                    "track_id": track_id,
                    "onset_tick": note.time,
                    "duration_tick": note.duration,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                }
            )
        for control in track.controls:
            control_changes.append(
                {
                    "track_id": track_id,
                    "time_tick": control.time,
                    "number": control.number,
                    "value": control.value,
                }
            )
        for bend in track.pitch_bends:
            pitch_bends.append(
                {"track_id": track_id, "time_tick": bend.time, "value": bend.value}
            )

    return build_canonical_frames(
        piece_id=piece_id,
        ticks_per_quarter=tpq,
        tracks=tracks,
        notes=notes,
        tempos=[{"time_tick": t.time, "qpm": t.qpm} for t in score.tempos],
        time_signatures=[
            {
                "time_tick": ts.time,
                "numerator": ts.numerator,
                "denominator": ts.denominator,
            }
            for ts in score.time_signatures
        ],
        key_signatures=[
            {"time_tick": ks.time, "key": ks.key, "tonality": ks.tonality}
            for ks in score.key_signatures
        ],
        control_changes=control_changes,
        pitch_bends=pitch_bends,
        metadata=metadata,
    )


def _frame(rows: Sequence[dict[str, Any]], table_name: str) -> pl.DataFrame:
    schema = TABLE_SCHEMAS[table_name]
    normalized = [{name: row.get(name) for name in schema.names} for row in rows]
    return pl.from_arrow(pa.Table.from_pylist(normalized, schema=schema))


def _note_rows(piece_id: str, notes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "piece_id": piece_id,
            "track_id": int(note.get("track_id", 0)),
            "onset_tick": int(note["onset_tick"]),
            "duration_tick": int(note["duration_tick"]),
            "pitch": int(note["pitch"]),
            "velocity": int(note.get("velocity", DEFAULT_VELOCITY)),
        }
        for note in notes
    ]
    rows.sort(key=lambda row: (row["track_id"], row["onset_tick"], row["pitch"]))
    note_id_by_track: dict[int, int] = {}
    for row in rows:
        track_id = row["track_id"]
        row["note_id"] = note_id_by_track.get(track_id, 0)
        note_id_by_track[track_id] = row["note_id"] + 1
    return rows


def _track_event_rows(
    piece_id: str,
    events: Iterable[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        row = {
            "piece_id": piece_id,
            "track_id": int(event.get("track_id", 0)),
            "time_tick": int(event["time_tick"]),
        }
        for field in fields:
            row[field] = int(event.get(field, defaults[field]))
        rows.append(row)
    rows.sort(key=lambda row: (row["track_id"], row["time_tick"]))
    return rows


def _global_event_rows(
    piece_id: str,
    events: Iterable[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    defaults: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        row = {"piece_id": piece_id, "time_tick": int(event["time_tick"])}
        for field in fields:
            value = event.get(field, defaults[field])
            row[field] = float(value) if field == "qpm" else int(value)
        rows.append(row)
    rows.sort(key=lambda row: row["time_tick"])
    return rows


def _track_rows(
    piece_id: str,
    tracks: Iterable[dict[str, Any]] | None,
    *event_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if tracks is not None:
        rows = [
            {
                "piece_id": piece_id,
                "track_id": int(track.get("track_id", index)),
                "track_name": track.get("track_name") or "",
                "program": int(track.get("program", 0)),
                "is_drum": bool(track.get("is_drum", False)),
            }
            for index, track in enumerate(tracks)
        ]
    else:
        rows = []
    known = {row["track_id"] for row in rows}
    referenced = {row["track_id"] for rows_ in event_rows for row in rows_}
    for track_id in sorted(referenced - known):
        rows.append(
            {
                "piece_id": piece_id,
                "track_id": track_id,
                "track_name": "",
                "program": 0,
                "is_drum": False,
            }
        )
    if not rows:
        rows.append(
            {
                "piece_id": piece_id,
                "track_id": 0,
                "track_name": "",
                "program": 0,
                "is_drum": False,
            }
        )
    rows.sort(key=lambda row: row["track_id"])
    return rows


def _fill_seconds(
    score: Score,
    tpq: int,
    note_rows: Sequence[dict[str, Any]],
    control_rows: Sequence[dict[str, Any]],
    bend_rows: Sequence[dict[str, Any]],
    tempo_rows: Sequence[dict[str, Any]],
    time_signature_rows: Sequence[dict[str, Any]],
    key_signature_rows: Sequence[dict[str, Any]],
) -> None:
    score_sec = score.to("second")
    seconds_notes = [note for track in score_sec.tracks for note in track.notes]
    seconds_controls = [control for track in score_sec.tracks for control in track.controls]
    seconds_bends = [bend for track in score_sec.tracks for bend in track.pitch_bends]

    for row, event in _pair(note_rows, seconds_notes, "notes"):
        row["onset_quarter"] = row["onset_tick"] / tpq
        row["duration_quarter"] = row["duration_tick"] / tpq
        row["onset_sec"] = float(event.time)
        row["duration_sec"] = float(event.duration)
    for label, rows, events in (
        ("control changes", control_rows, seconds_controls),
        ("pitch bends", bend_rows, seconds_bends),
        ("tempos", tempo_rows, score_sec.tempos),
        ("time signatures", time_signature_rows, score_sec.time_signatures),
        ("key signatures", key_signature_rows, score_sec.key_signatures),
    ):
        for row, event in _pair(rows, events, label):
            row["time_quarter"] = row["time_tick"] / tpq
            row["time_sec"] = float(event.time)


def _pair(rows: Sequence[dict[str, Any]], events: Sequence[Any], label: str):
    if len(rows) != len(events):
        raise ValueError(
            f"{label}: event count mismatch after symusic round trip: "
            f"{len(rows)} != {len(events)}"
        )
    return zip(rows, events, strict=True)

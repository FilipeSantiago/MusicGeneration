from __future__ import annotations

from pathlib import Path

import muspy
import polars as pl
from symusic import (
    ControlChange,
    KeySignature,
    Note,
    PitchBend,
    Score,
    Tempo,
    TimeSignature,
    Track,
)

from ..core.canonical_data import CanonicalDataset
from .canonical_loader import CanonicalPiece


def piece_frames(dataset: CanonicalDataset, piece_id: str) -> dict[str, pl.DataFrame]:
    return {
        "piece": dataset.pieces.filter(pl.col("piece_id") == piece_id),
        "tracks": dataset.tracks.filter(pl.col("piece_id") == piece_id),
        "notes": dataset.notes.filter(pl.col("piece_id") == piece_id),
        "tempos": dataset.tempos.filter(pl.col("piece_id") == piece_id),
        "time_signatures": dataset.time_signatures.filter(
            pl.col("piece_id") == piece_id
        ),
        "key_signatures": dataset.key_signatures.filter(pl.col("piece_id") == piece_id),
        "control_changes": dataset.control_changes.filter(
            pl.col("piece_id") == piece_id
        ),
        "pitch_bends": dataset.pitch_bends.filter(pl.col("piece_id") == piece_id),
    }


def piece_frames_from_canonical_piece(piece: CanonicalPiece) -> dict[str, pl.DataFrame]:
    return {
        "piece": pl.DataFrame([piece.piece]),
        "tracks": piece.tracks,
        "notes": piece.notes,
        "tempos": piece.tempos,
        "time_signatures": piece.time_signatures,
        "key_signatures": piece.key_signatures,
        "control_changes": piece.control_changes,
        "pitch_bends": piece.pitch_bends,
    }


def canonical_piece_to_symusic(dataset: CanonicalDataset, piece_id: str):
    frames = piece_frames(dataset, piece_id)
    return canonical_frames_to_symusic(frames)


def canonical_piece_data_to_symusic(piece: CanonicalPiece):
    return canonical_frames_to_symusic(piece_frames_from_canonical_piece(piece))


def canonical_frames_to_symusic(frames: dict[str, pl.DataFrame]):
    piece = frames["piece"].row(0, named=True)
    score = Score(int(piece["ticks_per_quarter"]))
    track_map: dict[int, Track] = {}

    for row in frames["tracks"].sort("track_id").iter_rows(named=True):
        track = Track(
            name=row["track_name"] or "",
            program=int(row["program"]),
            is_drum=bool(row["is_drum"]),
        )
        track_map[int(row["track_id"])] = track
        score.tracks.append(track)

    for row in (
        frames["notes"]
        .sort(["track_id", "onset_tick", "note_id"])
        .iter_rows(named=True)
    ):
        track_map[int(row["track_id"])].notes.append(
            Note(
                int(row["onset_tick"]),
                int(row["duration_tick"]),
                int(row["pitch"]),
                int(row["velocity"]),
            )
        )

    for row in (
        frames["control_changes"].sort(["track_id", "time_tick"]).iter_rows(named=True)
    ):
        control = ControlChange(
            int(row["time_tick"]), int(row["number"]), int(row["value"])
        )
        track_map[int(row["track_id"])].controls.append(control)

    for row in (
        frames["pitch_bends"].sort(["track_id", "time_tick"]).iter_rows(named=True)
    ):
        track_map[int(row["track_id"])].pitch_bends.append(
            PitchBend(int(row["time_tick"]), int(row["value"]))
        )

    for row in frames["tempos"].sort("time_tick").iter_rows(named=True):
        score.tempos.append(Tempo(int(row["time_tick"]), float(row["qpm"])))

    for row in frames["time_signatures"].sort("time_tick").iter_rows(named=True):
        score.time_signatures.append(
            TimeSignature(
                int(row["time_tick"]), int(row["numerator"]), int(row["denominator"])
            )
        )

    for row in frames["key_signatures"].sort("time_tick").iter_rows(named=True):
        score.key_signatures.append(
            KeySignature(int(row["time_tick"]), int(row["key"]), int(row["tonality"]))
        )

    return score


def canonical_piece_to_muspy(dataset: CanonicalDataset, piece_id: str) -> muspy.Music:
    frames = piece_frames(dataset, piece_id)
    return canonical_frames_to_muspy(frames)


def canonical_piece_data_to_muspy(piece: CanonicalPiece) -> muspy.Music:
    return canonical_frames_to_muspy(piece_frames_from_canonical_piece(piece))


def canonical_frames_to_muspy(frames: dict[str, pl.DataFrame]) -> muspy.Music:
    piece = frames["piece"].row(0, named=True)
    music = muspy.Music(resolution=int(piece["ticks_per_quarter"]))
    for row in frames["tempos"].sort("time_tick").iter_rows(named=True):
        music.tempos.append(
            muspy.Tempo(time=int(row["time_tick"]), qpm=float(row["qpm"]))
        )
    for row in frames["time_signatures"].sort("time_tick").iter_rows(named=True):
        music.time_signatures.append(
            muspy.TimeSignature(
                time=int(row["time_tick"]),
                numerator=int(row["numerator"]),
                denominator=int(row["denominator"]),
            )
        )
    for row in frames["key_signatures"].sort("time_tick").iter_rows(named=True):
        mode = "major" if int(row["tonality"]) == 0 else "minor"
        music.key_signatures.append(
            muspy.KeySignature(
                time=int(row["time_tick"]), root=int(row["key"]), mode=mode
            )
        )
    track_map: dict[int, muspy.Track] = {}
    for row in frames["tracks"].sort("track_id").iter_rows(named=True):
        track = muspy.Track(
            program=int(row["program"]),
            is_drum=bool(row["is_drum"]),
            name=row["track_name"] or None,
        )
        track_map[int(row["track_id"])] = track
        music.tracks.append(track)
    for row in (
        frames["notes"]
        .sort(["track_id", "onset_tick", "note_id"])
        .iter_rows(named=True)
    ):
        track_map[int(row["track_id"])].notes.append(
            muspy.Note(
                time=int(row["onset_tick"]),
                pitch=int(row["pitch"]),
                duration=int(row["duration_tick"]),
                velocity=int(row["velocity"]),
            )
        )
    return music


def canonical_frames_to_midi(
    frames: dict[str, pl.DataFrame],
    path: Path,
    *,
    ticks_per_quarter: int | None = None,
) -> Path:
    """The only canonical -> MIDI exit in the codebase."""
    score = canonical_frames_to_symusic(frames)
    if ticks_per_quarter is not None and ticks_per_quarter != score.ticks_per_quarter:
        score = score.resample(ticks_per_quarter)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(path))
    return path


def canonical_piece_to_midi(
    dataset: CanonicalDataset, piece_id: str, path: Path, **kwargs
) -> Path:
    return canonical_frames_to_midi(piece_frames(dataset, piece_id), path, **kwargs)


def canonical_piece_data_to_midi(piece: CanonicalPiece, path: Path, **kwargs) -> Path:
    return canonical_frames_to_midi(
        piece_frames_from_canonical_piece(piece), path, **kwargs
    )

"""Time queries over a canonical piece, precomputed once as numpy arrays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import polars as pl

SUSTAIN_CC = 64
SUSTAIN_THRESHOLD = 64
DEFAULT_QPM = 120.0
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class _StepFunction:
    """A piecewise-constant boolean signal: ``states[i]`` holds from ``times[i]``."""

    times: np.ndarray
    states: np.ndarray

    def at(self, t: float) -> bool:
        index = int(np.searchsorted(self.times, t, side="right")) - 1
        return bool(self.states[index]) if index >= 0 else False


@dataclass(frozen=True)
class Timeline:
    """Note, pedal, tempo and metre lookups by time in seconds.

    Notes are sorted by onset so window queries are two binary searches plus a
    filter over the (small) candidate slice.
    """

    onset: np.ndarray
    end: np.ndarray
    pitch: np.ndarray
    velocity: np.ndarray
    track: np.ndarray
    duration_sec: float
    piece_id: str = ""
    title: str | None = None
    composer: str | None = None
    sustain: Mapping[int, _StepFunction] = field(default_factory=dict)
    tempo_sec: np.ndarray = field(default_factory=lambda: np.zeros(1))
    tempo_quarter: np.ndarray = field(default_factory=lambda: np.zeros(1))
    tempo_qpm: np.ndarray = field(default_factory=lambda: np.array([DEFAULT_QPM]))
    ts_quarter: np.ndarray = field(default_factory=lambda: np.zeros(1))
    ts_bar_length: np.ndarray = field(default_factory=lambda: np.array([4.0]))
    ts_beat_length: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    ts_bar_base: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    max_note_duration: float = 0.0

    @classmethod
    def from_frames(cls, frames: Mapping[str, pl.DataFrame]) -> Timeline:
        notes = frames["notes"].sort("onset_sec", maintain_order=True)
        onset = notes["onset_sec"].to_numpy().astype(np.float64)
        pitch = notes["pitch"].to_numpy().astype(np.int16)
        end = _truncate_restrikes(
            onset, onset + notes["duration_sec"].to_numpy().astype(np.float64), pitch
        )
        # A key struck twice at the same instant leaves a zero-length note: drop it.
        keep = end > onset + _EPS
        onset, end, pitch = onset[keep], end[keep], pitch[keep]
        velocity = notes["velocity"].to_numpy().astype(np.int16)[keep]
        track = notes["track_id"].to_numpy().astype(np.int16)[keep]
        duration = end - onset

        piece = frames.get("piece")
        meta: dict[str, object] = {}
        if piece is not None and piece.height:
            meta = piece.row(0, named=True)
        piece_duration = _as_float(meta.get("duration_sec"))
        notes_end = float(end.max()) if end.size else 0.0
        duration_sec = max(piece_duration or 0.0, notes_end)

        tempo = _tempo_arrays(frames.get("tempos"))
        metre = _metre_arrays(frames.get("time_signatures"))

        return cls(
            onset=onset,
            end=end,
            pitch=pitch,
            velocity=velocity,
            track=track,
            duration_sec=duration_sec,
            piece_id=str(meta.get("piece_id") or ""),
            title=_as_str(meta.get("title")),
            composer=_as_str(meta.get("composer")),
            sustain=_sustain_signals(frames.get("control_changes")),
            tempo_sec=tempo[0],
            tempo_quarter=tempo[1],
            tempo_qpm=tempo[2],
            ts_quarter=metre[0],
            ts_bar_length=metre[1],
            ts_beat_length=metre[2],
            ts_bar_base=metre[3],
            max_note_duration=float(duration.max()) if duration.size else 0.0,
        )

    @property
    def note_count(self) -> int:
        return int(self.onset.size)

    def visible(self, t: float, lookahead: float) -> np.ndarray:
        """Indices of notes overlapping ``[t, t + lookahead)``."""
        lo = int(np.searchsorted(self.onset, t - self.max_note_duration, side="left"))
        hi = int(np.searchsorted(self.onset, t + lookahead, side="left"))
        candidates = np.arange(lo, hi)
        return candidates[self.end[lo:hi] > t]

    def active(self, t: float) -> np.ndarray:
        """Indices of notes sounding at ``t`` (``onset <= t < end``)."""
        lo = int(np.searchsorted(self.onset, t - self.max_note_duration, side="left"))
        hi = int(np.searchsorted(self.onset, t, side="right"))
        candidates = np.arange(lo, hi)
        return candidates[self.end[lo:hi] > t]

    def busiest_window(self, length: float) -> float:
        """Start time of the ``length``-second window with the most note onsets."""
        n = self.note_count
        if n == 0 or self.duration_sec <= length:
            return 0.0
        ends = np.searchsorted(self.onset, self.onset + length, side="left")
        counts = ends - np.arange(n)
        counts[self.onset > self.duration_sec - length] = -1
        if counts.max() < 0:
            return max(self.duration_sec - length, 0.0)
        return float(self.onset[int(np.argmax(counts))])

    def sustain_at(self, t: float, track: int | None = None) -> bool:
        """Whether the sustain pedal (CC64 >= 64) is down, on ``track`` or any track."""
        if track is not None:
            signal = self.sustain.get(track)
            return signal.at(t) if signal is not None else False
        return any(signal.at(t) for signal in self.sustain.values())

    def quarter_at(self, t: float) -> float:
        index = max(int(np.searchsorted(self.tempo_sec, t, side="right")) - 1, 0)
        return float(
            self.tempo_quarter[index]
            + (t - self.tempo_sec[index]) * self.tempo_qpm[index] / 60.0
        )

    def bar_beat_at(self, t: float) -> tuple[int, int]:
        """1-based (bar, beat) at ``t``, honouring tempo and time-signature changes."""
        quarter = max(self.quarter_at(t), 0.0)
        index = max(
            int(np.searchsorted(self.ts_quarter, quarter + _EPS, side="right")) - 1, 0
        )
        relative = quarter - self.ts_quarter[index] + _EPS
        bar_length = self.ts_bar_length[index]
        bars = math.floor(relative / bar_length)
        beat = math.floor((relative - bars * bar_length) / self.ts_beat_length[index])
        return int(self.ts_bar_base[index]) + bars + 1, beat + 1


def _truncate_restrikes(
    onset: np.ndarray, end: np.ndarray, pitch: np.ndarray
) -> np.ndarray:
    """End each note no later than the next onset on the same key.

    A piano key can only sound once; overlapping notes on one pitch (common in
    generated output) would otherwise be drawn stacked on top of each other.
    """
    end = end.copy()
    if onset.size < 2:
        return end
    order = np.lexsort((onset, pitch))
    same_key = pitch[order][1:] == pitch[order][:-1]
    earlier, later = order[:-1][same_key], order[1:][same_key]
    end[earlier] = np.minimum(end[earlier], onset[later])
    return end


def _tempo_arrays(
    tempos: pl.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tempos is None or tempos.height == 0:
        return np.zeros(1), np.zeros(1), np.array([DEFAULT_QPM])
    tempos = tempos.sort("time_sec")
    sec = tempos["time_sec"].to_numpy().astype(np.float64)
    quarter = tempos["time_quarter"].to_numpy().astype(np.float64)
    qpm = tempos["qpm"].to_numpy().astype(np.float64)
    if sec[0] > 0:
        # Before the first tempo event, extrapolate backwards with its tempo.
        sec = np.concatenate([[0.0], sec])
        quarter = np.concatenate([[quarter[0] - sec[1] * qpm[0] / 60.0], quarter])
        qpm = np.concatenate([[qpm[0]], qpm])
    return sec, quarter, qpm


def _metre_arrays(
    time_signatures: pl.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    if time_signatures is not None and time_signatures.height:
        for row in time_signatures.sort("time_quarter").iter_rows(named=True):
            numerator = max(int(row["numerator"]), 1)
            denominator = max(int(row["denominator"]), 1)
            beat = 4.0 / denominator
            rows.append((float(row["time_quarter"]), numerator * beat, beat))
    if not rows or rows[0][0] > 0:
        rows.insert(0, (0.0, 4.0, 1.0))
    # A later signature at the same position replaces the earlier one.
    deduped: list[tuple[float, float, float]] = []
    for segment in rows:
        if deduped and abs(deduped[-1][0] - segment[0]) < _EPS:
            deduped[-1] = segment
        else:
            deduped.append(segment)

    starts = np.array([segment[0] for segment in deduped])
    bar_lengths = np.array([segment[1] for segment in deduped])
    beat_lengths = np.array([segment[2] for segment in deduped])
    bar_base = np.zeros(len(deduped), dtype=np.int64)
    for i in range(1, len(deduped)):
        span = starts[i] - starts[i - 1]
        # A change mid-bar still starts a new bar; the partial bar counts.
        bar_base[i] = bar_base[i - 1] + math.ceil(span / bar_lengths[i - 1] - _EPS)
    return starts, bar_lengths, beat_lengths, bar_base


def _sustain_signals(control_changes: pl.DataFrame | None) -> dict[int, _StepFunction]:
    if control_changes is None or control_changes.height == 0:
        return {}
    pedal = control_changes.filter(pl.col("number") == SUSTAIN_CC).sort(
        ["track_id", "time_sec"], maintain_order=True
    )
    signals: dict[int, _StepFunction] = {}
    for (track_id,), group in pedal.group_by(["track_id"], maintain_order=True):
        times = group["time_sec"].to_numpy().astype(np.float64)
        states = group["value"].to_numpy() >= SUSTAIN_THRESHOLD
        signals[int(track_id)] = _StepFunction(times, states)  # type: ignore[arg-type]
    return signals


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

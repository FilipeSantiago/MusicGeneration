from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ..helpers.canonical_frames import build_canonical_frames
from .base import DecodeContext, RepresentationDecoder


@dataclass(slots=True)
class PianoRollArrays:
    active: np.ndarray
    onsets: np.ndarray
    velocities: np.ndarray
    sustain: np.ndarray
    start_tick: int = 0


class PianoRollDecoder(RepresentationDecoder):
    """Rebuild notes from the frame grid and CC64 from the sustain channel."""

    representation_name = "piano_roll"

    def decode(
        self, payload: PianoRollArrays | dict[str, Any], context: DecodeContext | None = None
    ) -> dict[str, pl.DataFrame]:
        context = context or DecodeContext()
        arrays = self._as_arrays(payload)
        tpq = context.ticks_per_quarter
        resolution = self.config.piano_roll.resolution
        base_tick = context.start_tick or arrays.start_tick

        def to_tick(frame_index: float) -> int:
            return base_tick + round(frame_index * tpq / resolution)

        notes: list[dict[str, Any]] = []
        length, num_pitches = arrays.active.shape
        for pitch in range(num_pitches):
            active = arrays.active[:, pitch]
            onsets = arrays.onsets[:, pitch]
            start: int | None = None
            for frame in range(length):
                # A new onset both closes the running note and opens the next one,
                # which is the only way re-articulated pitches survive.
                if start is not None and (not active[frame] or onsets[frame]):
                    notes.append(self._note(arrays, pitch, start, frame, to_tick))
                    start = None
                if onsets[frame] or (active[frame] and start is None):
                    start = frame
            if start is not None:
                notes.append(self._note(arrays, pitch, start, length, to_tick))

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
            control_changes=self._sustain_controls(arrays.sustain, to_tick),
            metadata=context.metadata,
            default_qpm=context.default_qpm,
        )

    def load_payload(
        self, *, piece_id: str, segment_id: str | None = None
    ) -> PianoRollArrays:
        row = self._segment_row(piece_id, segment_id)
        with np.load(self.representation_dir / row["array_file"]) as payload:
            return PianoRollArrays(
                active=payload["active"],
                onsets=payload["onsets"],
                velocities=payload["velocities"],
                sustain=payload["sustain"],
                start_tick=int(row["start_tick"]),
            )

    @staticmethod
    def _note(arrays, pitch: int, start: int, end: int, to_tick) -> dict[str, Any]:
        onset_tick = to_tick(start)
        return {
            "track_id": 0,
            "onset_tick": onset_tick,
            "duration_tick": max(1, to_tick(end) - onset_tick),
            "pitch": pitch,
            "velocity": max(1, int(arrays.velocities[start, pitch])),
        }

    @staticmethod
    def _sustain_controls(sustain: np.ndarray, to_tick) -> list[dict[str, Any]]:
        controls: list[dict[str, Any]] = []
        previous = 0
        for frame, value in enumerate(sustain.tolist()):
            value = int(value)
            if frame == 0 or value != previous:
                if frame == 0 and value == 0:
                    previous = value
                    continue
                controls.append(
                    {
                        "track_id": 0,
                        "time_tick": to_tick(frame),
                        "number": 64,
                        "value": 127 if value else 0,
                    }
                )
            previous = value
        return controls

    @staticmethod
    def _as_arrays(payload: PianoRollArrays | dict[str, Any]) -> PianoRollArrays:
        if isinstance(payload, PianoRollArrays):
            return payload
        return PianoRollArrays(
            active=np.asarray(payload["active"]),
            onsets=np.asarray(payload["onsets"]),
            velocities=np.asarray(payload["velocities"]),
            sustain=np.asarray(payload["sustain"]),
            start_tick=int(payload.get("start_tick", 0)),
        )

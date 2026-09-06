from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import mido
import pretty_midi


class MGEvalPiece(TypedDict):
    """The two MIDI views consumed by MGEval-style metrics."""

    pretty_midi: pretty_midi.PrettyMIDI
    midi_pattern: mido.MidiFile


class MGDataset:
    """In-memory MGEval feature dataset keyed by canonical piece ID."""

    def __init__(self, pieces: Mapping[str, MGEvalPiece]) -> None:
        self.pieces = dict(pieces)
        self._piece_ids = tuple(self.pieces)

    def __len__(self) -> int:
        return len(self._piece_ids)

    def __getitem__(self, index: int) -> MGEvalPiece:
        return self.pieces[self._piece_ids[index]]

    def piece_id(self, index: int) -> str:
        """Return the canonical piece ID associated with a positional index."""
        return self._piece_ids[index]

from __future__ import annotations

from io import BytesIO

import muspy
import pretty_midi
import python3_midi

from data_processing.music_representations.core.canonical_data import CanonicalDataset
from data_processing.music_representations.helpers.adapters import (
    canonical_piece_to_muspy,
    canonical_piece_to_symusic,
)
from evaluation.model.mgeval_dataset import MGDataset, MGEvalPiece


class MGEvalAdapter:
    """Convert canonical pieces into the feature inputs used by MGEval."""

    def convert(
        self,
        dataset: CanonicalDataset,
    ) -> MGDataset:
        pieces: dict[str, MGEvalPiece] = {}

        for piece_id in dataset.pieces.get_column("piece_id"):
            pieces[piece_id] = self._convert_piece(dataset, piece_id)

        return MGDataset(pieces=pieces)

    def _convert_piece(
        self,
        dataset: CanonicalDataset,
        piece_id: str,
    ) -> MGEvalPiece:
        score = canonical_piece_to_symusic(dataset, piece_id)
        midi_bytes = score.dumps_midi()
        music = canonical_piece_to_muspy(dataset, piece_id)

        # MGEval's pattern metrics expect track 0 to hold global metadata and
        # the notes to live on a later track. MusPy's Mido exporter guarantees
        # that layout even when the source MIDI used a single combined track,
        # so round-trip through it to get the python3-midi Pattern MGEval reads.
        buffer = BytesIO()
        muspy.to_mido(music).save(file=buffer)
        buffer.seek(0)

        return {
            "pretty_midi": pretty_midi.PrettyMIDI(BytesIO(midi_bytes)),
            "midi_pattern": python3_midi.read_midifile(buffer),
        }

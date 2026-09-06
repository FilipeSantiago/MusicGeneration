from __future__ import annotations

import muspy

from data_processing.music_representations.core.canonical_data import CanonicalDataset
from data_processing.music_representations.helpers.adapters import (
    canonical_piece_to_muspy,
)
from evaluation.model.muspy_dataset import MusPyDataset


class MusPyAdapter:
    """Convert a canonical dataset into an in-memory MusPy dataset."""

    def convert(
        self,
        dataset: CanonicalDataset,
    ) -> MusPyDataset:
        pieces: dict[str, muspy.Music] = {}

        for piece_id in dataset.pieces.get_column("piece_id"):
            pieces[piece_id] = self._convert_piece(dataset, piece_id)

        return MusPyDataset(pieces=pieces)

    def _convert_piece(
        self,
        dataset: CanonicalDataset,
        piece_id: str,
    ) -> muspy.Music:
        return canonical_piece_to_muspy(dataset, piece_id)

from __future__ import annotations

from collections.abc import Mapping

import muspy


class MusPyDataset(muspy.Dataset):
    """In-memory MusPy dataset keyed by canonical piece ID."""

    def __init__(self, pieces: Mapping[str, muspy.Music]) -> None:
        self.pieces = dict(pieces)
        self._piece_ids = tuple(self.pieces)

    def __len__(self) -> int:
        return len(self._piece_ids)

    def __getitem__(self, index: int) -> muspy.Music:
        return self.pieces[self._piece_ids[index]]

    def piece_id(self, index: int) -> str:
        """Return the canonical piece ID associated with a positional index."""
        return self._piece_ids[index]

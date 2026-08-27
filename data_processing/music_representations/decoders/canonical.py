from __future__ import annotations

import polars as pl

from ..helpers.adapters import piece_frames_from_canonical_piece
from ..helpers.canonical_loader import CanonicalPiece, load_canonical_piece
from .base import DecodeContext, RepresentationDecoder


class CanonicalDecoder(RepresentationDecoder):
    """Identity decoder, so the registry is total and every path ends at canonical."""

    representation_name = "canonical"

    def decode(
        self, payload: CanonicalPiece | dict[str, pl.DataFrame],
        context: DecodeContext | None = None,
    ) -> dict[str, pl.DataFrame]:
        if isinstance(payload, CanonicalPiece):
            return piece_frames_from_canonical_piece(payload)
        return payload

    def load_payload(
        self, *, piece_id: str, segment_id: str | None = None
    ) -> CanonicalPiece:
        return load_canonical_piece(self.representation_dir, piece_id)

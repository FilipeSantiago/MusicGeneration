from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from storage import LocalStorage, Storage

from ..config import MusicRepresentationConfig
from ..contracts import RoundTripContract, contract_for
from ..helpers.adapters import canonical_frames_to_midi
from ..helpers.canonical_frames import DEFAULT_QPM, DEFAULT_TICKS_PER_QUARTER
from ..helpers.canonical_loader import load_canonical_pieces


class DecodeError(RuntimeError):
    """Raised when a payload cannot be turned back into canonical data."""


@dataclass(frozen=True, slots=True)
class DecodeContext:
    """Everything a representation dropped and therefore cannot supply itself."""

    piece_id: str = "decoded"
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER
    start_tick: int = 0
    program: int = 0
    is_drum: bool = False
    track_name: str = ""
    default_qpm: float = DEFAULT_QPM
    metadata: dict[str, Any] = field(default_factory=dict)


class RepresentationDecoder(ABC):
    """Turns one representation back into canonical frames.

    Decoding is split so that both callers are first class: :meth:`decode` is pure
    and in-memory (what a model's output needs), while :meth:`decode_stored` reads
    the representation's own build output.
    """

    representation_name: ClassVar[str]

    def __init__(
        self,
        config: MusicRepresentationConfig,
        storage: Storage | None = None,
        representation_dir: Path | None = None,
    ):
        self.config = config
        self.storage = storage if storage is not None else LocalStorage(config.output_dir)
        self._representation_dir = representation_dir

    @property
    def representation_dir(self) -> Path:
        if self._representation_dir is None:
            self._representation_dir = self.storage.materialize(self.representation_name)
        return self._representation_dir

    @property
    def contract(self) -> RoundTripContract:
        return contract_for(self.representation_name, self.config)

    @abstractmethod
    def decode(
        self, payload: Any, context: DecodeContext | None = None
    ) -> dict[str, pl.DataFrame]:
        """Decode an in-memory payload into canonical frames."""

    @abstractmethod
    def load_payload(self, *, piece_id: str, segment_id: str | None = None) -> Any:
        """Read this representation's stored output for one piece or segment."""

    def decode_stored(
        self,
        *,
        piece_id: str,
        segment_id: str | None = None,
        context: DecodeContext | None = None,
    ) -> dict[str, pl.DataFrame]:
        payload = self.load_payload(piece_id=piece_id, segment_id=segment_id)
        return self.decode(payload, context or self.stored_context(piece_id))

    def decode_to_midi(
        self, payload: Any, path: Path, context: DecodeContext | None = None
    ) -> Path:
        return canonical_frames_to_midi(self.decode(payload, context), path)

    def stored_context(self, piece_id: str) -> DecodeContext:
        """Recover what the representation dropped from the canonical source tables.

        Only `ticks_per_quarter` and the piece metadata are taken from canonical —
        never musical content — so this stays a context lookup, not a shortcut that
        would make a round trip pass for the wrong reason.
        """
        context = DecodeContext(piece_id=piece_id)
        try:
            pieces = load_canonical_pieces(
                self.storage.materialize(self.config.canonical_dir_name)
            )
        except (FileNotFoundError, OSError):
            return context
        matching = pieces.filter(pl.col("piece_id") == piece_id)
        if matching.is_empty():
            return context
        row = matching.row(0, named=True)
        return replace(
            context,
            ticks_per_quarter=int(row["ticks_per_quarter"]),
            metadata={
                "maestro_split": row["maestro_split"],
                "composer": row["composer"],
                "title": row["title"],
                "schema_version": self.config.schema_version,
            },
        )

    def _segments(self) -> pl.DataFrame:
        path = self.representation_dir / "segments.parquet"
        if not path.exists():
            raise DecodeError(f"{self.representation_name}: {path} is missing")
        return pl.read_parquet(path)

    def _segment_row(self, piece_id: str, segment_id: str | None) -> dict[str, Any]:
        segments = self._segments()
        if segment_id is not None:
            matching = segments.filter(pl.col("segment_id") == segment_id)
        else:
            matching = segments.filter(pl.col("piece_id") == piece_id)
        if matching.is_empty():
            raise DecodeError(
                f"{self.representation_name}: no segment for "
                f"piece_id={piece_id!r} segment_id={segment_id!r}"
            )
        return matching.row(0, named=True)

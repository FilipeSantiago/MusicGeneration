from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ..core.canonical_data import CanonicalDataset


@dataclass(slots=True)
class CanonicalPiece:
    piece: dict[str, object]
    tracks: pl.DataFrame
    notes: pl.DataFrame
    tempos: pl.DataFrame
    time_signatures: pl.DataFrame
    key_signatures: pl.DataFrame
    control_changes: pl.DataFrame
    pitch_bends: pl.DataFrame


def load_canonical_dataset(root: Path) -> CanonicalDataset:
    root = root.resolve()
    return CanonicalDataset(
        root=root,
        pieces=pl.read_parquet(root / "pieces.parquet"),
        tracks=pl.read_parquet(root / "tracks.parquet"),
        notes=pl.read_parquet(root / "notes.parquet"),
        tempos=pl.read_parquet(root / "tempos.parquet"),
        time_signatures=pl.read_parquet(root / "time_signatures.parquet"),
        key_signatures=pl.read_parquet(root / "key_signatures.parquet"),
        control_changes=pl.read_parquet(root / "control_changes.parquet"),
        pitch_bends=pl.read_parquet(root / "pitch_bends.parquet"),
        manifest=json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )


def load_canonical_manifest(root: Path) -> dict:
    return json.loads((root.resolve() / "manifest.json").read_text(encoding="utf-8"))


def load_canonical_pieces(root: Path) -> pl.DataFrame:
    return pl.read_parquet(root.resolve() / "pieces.parquet")


def iter_canonical_pieces(root: Path) -> Iterator[CanonicalPiece]:
    root = root.resolve()
    for piece in load_canonical_pieces(root).iter_rows(named=True):
        piece_id = str(piece["piece_id"])
        yield CanonicalPiece(
            piece=piece,
            tracks=_load_piece_frame(root, "tracks", piece_id),
            notes=_load_piece_frame(root, "notes", piece_id),
            tempos=_load_piece_frame(root, "tempos", piece_id),
            time_signatures=_load_piece_frame(root, "time_signatures", piece_id),
            key_signatures=_load_piece_frame(root, "key_signatures", piece_id),
            control_changes=_load_piece_frame(root, "control_changes", piece_id),
            pitch_bends=_load_piece_frame(root, "pitch_bends", piece_id),
        )


def load_canonical_piece(root: Path, piece_id: str) -> CanonicalPiece:
    root = root.resolve()
    piece = (
        load_canonical_pieces(root).filter(pl.col("piece_id") == piece_id).row(0, named=True)
    )
    return CanonicalPiece(
        piece=piece,
        tracks=_load_piece_frame(root, "tracks", piece_id),
        notes=_load_piece_frame(root, "notes", piece_id),
        tempos=_load_piece_frame(root, "tempos", piece_id),
        time_signatures=_load_piece_frame(root, "time_signatures", piece_id),
        key_signatures=_load_piece_frame(root, "key_signatures", piece_id),
        control_changes=_load_piece_frame(root, "control_changes", piece_id),
        pitch_bends=_load_piece_frame(root, "pitch_bends", piece_id),
    )


def _load_piece_frame(root: Path, table_name: str, piece_id: str) -> pl.DataFrame:
    dataset = ds.dataset(root / f"{table_name}.parquet", format="parquet")
    table = dataset.to_table(filter=pc.field("piece_id") == piece_id)
    return pl.from_arrow(table)

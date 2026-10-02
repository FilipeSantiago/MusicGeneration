from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ..core.canonical_data import CanonicalDataset
from ..schemas import CANONICAL_FRAME_NAMES, TABLE_NAMES, TABLE_SCHEMAS

# Frame name (as produced by ``build_canonical_frames``) to the matching
# ``CanonicalDataset`` field; the two differ only for "piece" / "pieces".
_FRAME_TO_FIELD = dict(zip(CANONICAL_FRAME_NAMES, TABLE_NAMES, strict=True))


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


def canonical_dataset_from_frames(
    frames: Sequence[Mapping[str, pl.DataFrame]],
    root: Path,
    manifest: dict | None = None,
) -> CanonicalDataset:
    """Concatenate per-piece canonical frames into an in-memory dataset.

    Writes nothing: `root` and `manifest` are recorded as-is, for callers that
    have frames in hand and only need the eight tables (the evaluation adapters
    never read anything else). Use `write_canonical_dataset` to persist.
    """
    tables = {
        field_name: _concat_frames(field_name, [piece[frame_name] for piece in frames])
        for frame_name, field_name in _FRAME_TO_FIELD.items()
    }
    return CanonicalDataset(root=root, manifest=manifest or {}, **tables)


def _concat_frames(table_name: str, frames: list[pl.DataFrame]) -> pl.DataFrame:
    # pl.concat raises on an empty list, so an empty dataset has to take its
    # correctly typed empty frame from the schema instead.
    if not frames:
        return pl.from_arrow(TABLE_SCHEMAS[table_name].empty_table())
    return pl.concat(frames)


def filter_canonical_dataset(
    dataset: CanonicalDataset, piece_ids: Iterable[str]
) -> CanonicalDataset:
    """Restrict every canonical table to `piece_ids`, preserving row order.

    `root` and `manifest` still describe the full dataset on disk.
    """
    # An explicit dtype keeps is_in working when the selection is empty; the
    # implode marks it as a collection to match against rather than row-wise.
    wanted = pl.Series("piece_id", list(piece_ids), dtype=pl.String).implode()
    return replace(
        dataset,
        **{
            name: getattr(dataset, name).filter(pl.col("piece_id").is_in(wanted))
            for name in TABLE_NAMES
        },
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

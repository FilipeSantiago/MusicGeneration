from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .canonical_data import CanonicalDataset


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

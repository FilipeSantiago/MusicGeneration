from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(slots=True)
class CanonicalDataset:
    root: Path
    pieces: pl.DataFrame
    tracks: pl.DataFrame
    notes: pl.DataFrame
    tempos: pl.DataFrame
    time_signatures: pl.DataFrame
    key_signatures: pl.DataFrame
    control_changes: pl.DataFrame
    pitch_bends: pl.DataFrame
    manifest: dict

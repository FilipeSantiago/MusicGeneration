from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import polars as pl

from .config import MusicRepresentationConfig
from .helpers.io import list_output_files, utc_now_iso, write_json


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in [
        "symusic",
        "polars",
        "muspy",
        "pypianoroll",
        "miditok",
        "pretty_midi",
        "pyarrow",
    ]:
        module = importlib.import_module(module_name)
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def split_counts(
    frame: pl.DataFrame, split_column: str = "maestro_split"
) -> dict[str, int]:
    if frame.is_empty():
        return {}
    rows = frame.group_by(split_column).len().sort(split_column).iter_rows()
    return {split: count for split, count in rows}


def write_manifest(
    output_dir: Path,
    *,
    representation: str,
    representation_schema_version: str,
    source_canonical_schema_version: str,
    source_dataset_fingerprint: str,
    config: MusicRepresentationConfig,
    pieces: pl.DataFrame,
    segments: pl.DataFrame | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "representation": representation,
        "representation_schema_version": representation_schema_version,
        "created_at": utc_now_iso(),
        "source_canonical_schema_version": source_canonical_schema_version,
        "source_dataset_fingerprint": source_dataset_fingerprint,
        "effective_configuration": config.to_dict(),
        "dependency_versions": dependency_versions(),
        "pieces_per_split": split_counts(pieces),
        "segments_per_split": split_counts(segments) if segments is not None else {},
        "output_files": list_output_files(output_dir),
        "configuration_hash": config.config_hash(),
        "known_information_loss": [],
    }
    if extra:
        payload.update(extra)
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, payload)
    return manifest_path

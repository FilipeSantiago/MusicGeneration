from __future__ import annotations

import csv
import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar

import polars as pl
import pyarrow.parquet as pq
from symusic import Score
from tqdm.auto import tqdm

from ..core.base import BaseRepresentationBuilder
from ..helpers.canonical_frames import symusic_to_canonical_frames
from ..manifest import write_manifest
from ..schemas import (
    CONTROL_CHANGES_SCHEMA,
    KEY_SIGNATURES_SCHEMA,
    NOTES_SCHEMA,
    PIECES_SCHEMA,
    PITCH_BENDS_SCHEMA,
    TABLE_SCHEMAS,
    TEMPOS_SCHEMA,
    TIME_SIGNATURES_SCHEMA,
    TRACKS_SCHEMA,
)

# Canonical frames key the single-row pieces table as "piece" (see helpers.adapters).
FRAME_KEYS = {name: "piece" if name == "pieces" else name for name in TABLE_SCHEMAS}


class CanonicalBuilder(BaseRepresentationBuilder):
    """Build the canonical dataset from raw MIDI plus MAESTRO metadata."""

    representation_name = "canonical"
    MIDI_SUFFIXES: ClassVar[frozenset[str]] = frozenset({".mid", ".midi"})

    def _metadata_index(self) -> dict[str, dict[str, str]]:
        if self.config.metadata_csv is None:
            candidate = self.config.input_dir / "maestro-v3.0.0.csv"
            if candidate.exists():
                metadata_csv = candidate
            else:
                return {}
        else:
            metadata_csv = self.config.metadata_csv
        rows: dict[str, dict[str, str]] = {}
        with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                midi_filename = (
                    row.get("midi_filename") or row.get("canonical_composer") or ""
                )
                if midi_filename:
                    rows[midi_filename] = row
        return rows

    def _iter_midi_files(self) -> list[Path]:
        files = sorted(
            path
            for path in self.config.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.MIDI_SUFFIXES
        )
        if self.config.limit is not None:
            files = files[: self.config.limit]
        return files

    def _build_into(self, output_dir: Path) -> None:
        metadata = self._metadata_index()
        midi_files = self._iter_midi_files()
        with ExitStack() as stack:
            writers = {
                "pieces": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "pieces.parquet", PIECES_SCHEMA, compression="zstd"
                    )
                ),
                "tracks": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "tracks.parquet", TRACKS_SCHEMA, compression="zstd"
                    )
                ),
                "notes": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "notes.parquet", NOTES_SCHEMA, compression="zstd"
                    )
                ),
                "tempos": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "tempos.parquet", TEMPOS_SCHEMA, compression="zstd"
                    )
                ),
                "time_signatures": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "time_signatures.parquet",
                        TIME_SIGNATURES_SCHEMA,
                        compression="zstd",
                    )
                ),
                "key_signatures": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "key_signatures.parquet",
                        KEY_SIGNATURES_SCHEMA,
                        compression="zstd",
                    )
                ),
                "control_changes": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "control_changes.parquet",
                        CONTROL_CHANGES_SCHEMA,
                        compression="zstd",
                    )
                ),
                "pitch_bends": stack.enter_context(
                    pq.ParquetWriter(
                        output_dir / "pitch_bends.parquet",
                        PITCH_BENDS_SCHEMA,
                        compression="zstd",
                    )
                ),
            }
            for midi_path in tqdm(
                midi_files,
                desc="canonical",
                unit="midi",
                leave=False,
            ):
                frames = self._build_piece_frames(midi_path, metadata)
                for table_name, writer in writers.items():
                    table = frames[FRAME_KEYS[table_name]].to_arrow()
                    writer.write_table(table.cast(TABLE_SCHEMAS[table_name]))

        pieces_df = pl.read_parquet(output_dir / "pieces.parquet")
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=self.config.schema_version,
            source_dataset_fingerprint=self._dataset_fingerprint(pieces_df),
            config=self.config,
            pieces=pieces_df,
            segments=None,
            extra={
                "known_information_loss_extra": [
                    "lyrics and markers are not persisted in the canonical tables"
                ]
            },
        )

    def _build_piece_frames(
        self, midi_path: Path, metadata: dict[str, dict[str, str]]
    ) -> dict[str, pl.DataFrame]:
        relative_midi = midi_path.relative_to(self.config.input_dir)
        source_key = relative_midi.as_posix()
        meta = metadata.get(source_key, {})
        split = meta.get(
            "split", relative_midi.parts[0] if relative_midi.parts else "unknown"
        )
        return symusic_to_canonical_frames(
            Score(midi_path),
            piece_id=hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16],
            metadata={
                "source_midi": source_key,
                "source_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
                "composer": meta.get("canonical_composer") or meta.get("composer"),
                "title": meta.get("title"),
                "maestro_split": split,
                "schema_version": self.config.schema_version,
            },
        )

    @staticmethod
    def _dataset_fingerprint(pieces: pl.DataFrame) -> str:
        payload = "|".join(
            f"{row['piece_id']}:{row['source_sha256']}"
            for row in pieces.sort("piece_id").iter_rows(named=True)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

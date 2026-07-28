from __future__ import annotations

import csv
import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from symusic import Score
from tqdm.auto import tqdm

from ..core.base import BaseRepresentationBuilder
from ..manifest import write_manifest
from ..schemas import (
    CONTROL_CHANGES_SCHEMA,
    KEY_SIGNATURES_SCHEMA,
    NOTES_SCHEMA,
    PIECES_SCHEMA,
    PITCH_BENDS_SCHEMA,
    TEMPOS_SCHEMA,
    TIME_SIGNATURES_SCHEMA,
    TRACKS_SCHEMA,
)


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

    @staticmethod
    def _pair_events(tick_events, second_events, label: str):
        if len(tick_events) != len(second_events):
            raise ValueError(
                f"{label}: event count mismatch: {len(tick_events)} != {len(second_events)}"
            )
        return zip(tick_events, second_events, strict=True)

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
                piece_rows = self._build_piece_rows(midi_path, metadata)
                for table_name, rows in piece_rows.items():
                    self._write_rows(writers[table_name], rows, TABLE_SCHEMAS[table_name])

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
                "known_information_loss": [
                    "lyrics and markers are not persisted in the canonical tables"
                ]
            },
        )

    @staticmethod
    def _write_rows(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema) -> None:
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)

    def _build_piece_rows(
        self, midi_path: Path, metadata: dict[str, dict[str, str]]
    ) -> dict[str, list[dict]]:
        relative_midi = midi_path.relative_to(self.config.input_dir)
        source_key = relative_midi.as_posix()
        meta = metadata.get(source_key, {})
        split = meta.get(
            "split", relative_midi.parts[0] if relative_midi.parts else "unknown"
        )
        source_hash = hashlib.sha256(midi_path.read_bytes()).hexdigest()
        piece_id = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]

        score_tick = Score(midi_path)
        score_sec = score_tick.to("second")
        tpq = int(score_tick.ticks_per_quarter)
        rows: dict[str, list[dict]] = {
            "pieces": [
                {
                    "piece_id": piece_id,
                    "source_midi": source_key,
                    "source_sha256": source_hash,
                    "composer": meta.get("canonical_composer") or meta.get("composer"),
                    "title": meta.get("title"),
                    "maestro_split": split,
                    "ticks_per_quarter": tpq,
                    "end_tick": int(score_tick.end()),
                    "duration_sec": float(score_sec.end()),
                    "schema_version": self.config.schema_version,
                }
            ],
            "tracks": [],
            "notes": [],
            "tempos": [],
            "time_signatures": [],
            "key_signatures": [],
            "control_changes": [],
            "pitch_bends": [],
        }

        for track_id, (track_tick, track_sec) in enumerate(
            zip(score_tick.tracks, score_sec.tracks, strict=True)
        ):
            rows["tracks"].append(
                {
                    "piece_id": piece_id,
                    "track_id": track_id,
                    "track_name": track_tick.name,
                    "program": track_tick.program,
                    "is_drum": track_tick.is_drum,
                }
            )
            for note_id, (tick, sec) in enumerate(
                self._pair_events(track_tick.notes, track_sec.notes, "notes")
            ):
                rows["notes"].append(
                    {
                        "piece_id": piece_id,
                        "track_id": track_id,
                        "note_id": note_id,
                        "onset_tick": tick.time,
                        "duration_tick": tick.duration,
                        "onset_quarter": tick.time / tpq,
                        "duration_quarter": tick.duration / tpq,
                        "onset_sec": sec.time,
                        "duration_sec": sec.duration,
                        "pitch": tick.pitch,
                        "velocity": tick.velocity,
                    }
                )
            for tick, sec in self._pair_events(
                track_tick.controls, track_sec.controls, "control changes"
            ):
                rows["control_changes"].append(
                    {
                        "piece_id": piece_id,
                        "track_id": track_id,
                        "time_tick": tick.time,
                        "time_quarter": tick.time / tpq,
                        "time_sec": sec.time,
                        "number": tick.number,
                        "value": tick.value,
                    }
                )
            for tick, sec in self._pair_events(
                track_tick.pitch_bends, track_sec.pitch_bends, "pitch bends"
            ):
                rows["pitch_bends"].append(
                    {
                        "piece_id": piece_id,
                        "track_id": track_id,
                        "time_tick": tick.time,
                        "time_quarter": tick.time / tpq,
                        "time_sec": sec.time,
                        "value": tick.value,
                    }
                )

        for tick, sec in self._pair_events(score_tick.tempos, score_sec.tempos, "tempos"):
            rows["tempos"].append(
                {
                    "piece_id": piece_id,
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "qpm": tick.qpm,
                }
            )
        for tick, sec in self._pair_events(
            score_tick.time_signatures, score_sec.time_signatures, "time signatures"
        ):
            rows["time_signatures"].append(
                {
                    "piece_id": piece_id,
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "numerator": tick.numerator,
                    "denominator": tick.denominator,
                }
            )
        for tick, sec in self._pair_events(
            score_tick.key_signatures, score_sec.key_signatures, "key signatures"
        ):
            rows["key_signatures"].append(
                {
                    "piece_id": piece_id,
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "key": tick.key,
                    "tonality": tick.tonality,
                }
            )
        return rows

    @staticmethod
    def _dataset_fingerprint(pieces: pl.DataFrame) -> str:
        payload = "|".join(
            f"{row['piece_id']}:{row['source_sha256']}"
            for row in pieces.sort("piece_id").iter_rows(named=True)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
TABLE_SCHEMAS: dict[str, pa.Schema] = {
    "pieces": PIECES_SCHEMA,
    "tracks": TRACKS_SCHEMA,
    "notes": NOTES_SCHEMA,
    "tempos": TEMPOS_SCHEMA,
    "time_signatures": TIME_SIGNATURES_SCHEMA,
    "key_signatures": KEY_SIGNATURES_SCHEMA,
    "control_changes": CONTROL_CHANGES_SCHEMA,
    "pitch_bends": PITCH_BENDS_SCHEMA,
}

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from ..core.base import BaseRepresentationBuilder
from ..helpers.canonical_loader import (
    iter_canonical_pieces,
    load_canonical_manifest,
    load_canonical_pieces,
)
from ..helpers.segmentation import build_piece_segments
from ..manifest import write_manifest


class PianoRollBuilder(BaseRepresentationBuilder):
    """Export piano-roll tensors and aligned expressive side channels."""

    representation_name = "piano_roll"

    def _build_into(self, output_dir: Path) -> None:
        pieces = load_canonical_pieces(self.canonical_dir)
        manifest = load_canonical_manifest(self.canonical_dir)
        arrays_dir = output_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: list[dict[str, object]] = []
        for piece_data in tqdm(
            iter_canonical_pieces(self.canonical_dir),
            total=pieces.height,
            desc="piano_roll",
            unit="piece",
            leave=False,
        ):
            piece = piece_data.piece
            for segment in build_piece_segments(piece, piece_data.notes, self.config):
                segment_id = segment["segment_id"] or f"{segment['piece_id']}_full"
                resolution = self.config.piano_roll.resolution
                tpq = int(piece["ticks_per_quarter"])
                start_tick = int(segment["start_tick"])
                end_tick = int(segment["end_tick"])
                length = max(1, int(np.ceil((end_tick - start_tick) * resolution / tpq)))
                active = np.zeros((length, 128), dtype=np.uint8)
                onsets = np.zeros((length, 128), dtype=np.uint8)
                velocities = np.zeros((length, 128), dtype=np.uint8)
                sustain = np.zeros((length,), dtype=np.uint8)

                notes = piece_data.notes.filter(
                    (pl.col("onset_tick") < end_tick)
                    & ((pl.col("onset_tick") + pl.col("duration_tick")) > start_tick)
                )
                for row in notes.iter_rows(named=True):
                    onset = max(start_tick, int(row["onset_tick"]))
                    offset = min(
                        end_tick, int(row["onset_tick"]) + int(row["duration_tick"])
                    )
                    start_idx = min(
                        length - 1, int((onset - start_tick) * resolution / tpq)
                    )
                    end_idx = max(
                        start_idx + 1,
                        int(np.ceil((offset - start_tick) * resolution / tpq)),
                    )
                    pitch = int(row["pitch"])
                    active[start_idx:end_idx, pitch] = 1
                    onsets[start_idx, pitch] = 1
                    velocities[start_idx:end_idx, pitch] = int(row["velocity"])

                cc64 = piece_data.control_changes.filter(
                    (pl.col("number") == 64)
                    & (pl.col("time_tick") >= start_tick)
                    & (pl.col("time_tick") < end_tick)
                ).sort("time_tick")
                sustain_state = 0
                cursor = 0
                for row in cc64.iter_rows(named=True):
                    idx = min(
                        length - 1,
                        int((int(row["time_tick"]) - start_tick) * resolution / tpq),
                    )
                    sustain[cursor : idx + 1] = sustain_state
                    sustain_state = 1 if int(row["value"]) >= 64 else 0
                    cursor = idx
                sustain[cursor:] = sustain_state

                base_name = f"{segment_id}.npz"
                np.savez_compressed(
                    arrays_dir / base_name,
                    active=active,
                    onsets=onsets,
                    velocities=velocities,
                    sustain=sustain,
                )
                metadata_rows.append(
                    {
                        "piece_id": segment["piece_id"],
                        "segment_id": segment["segment_id"],
                        "maestro_split": segment["maestro_split"],
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "array_file": f"arrays/{base_name}",
                    }
                )
        segments_df = pl.DataFrame(metadata_rows)
        segments_df.write_parquet(output_dir / "segments.parquet")
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=manifest["source_canonical_schema_version"],
            source_dataset_fingerprint=manifest["source_dataset_fingerprint"],
            config=self.config,
            pieces=pieces,
            segments=segments_df,
            extra={
                "tensor_spec": {
                    "axes": ["time", "pitch"],
                    "channels": {
                        "active": {"dtype": "uint8", "range": [0, 1]},
                        "onsets": {"dtype": "uint8", "range": [0, 1]},
                        "velocities": {"dtype": "uint8", "range": [0, 127]},
                        "sustain": {"dtype": "uint8", "range": [0, 1]},
                    },
                    "resolution_frames_per_quarter": self.config.piano_roll.resolution,
                },
                "known_information_loss": [
                    "tempo changes, non-sustain control changes, and pitch bends are stored only in canonical source tables"
                ],
            },
        )

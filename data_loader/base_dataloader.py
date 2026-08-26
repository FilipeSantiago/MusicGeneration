import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from data_processing.music_representations import MusicRepresentationConfig
from storage import build_storage

NOTE_TABLE_COLUMNS = [
    "piece_id",
    "segment_id",
    "pitch",
    "velocity",
    "onset",
    "duration_bin",
    "delta_onset_bin",
]


class BaseDataloader(ABC):

    def __init__(self):
        self.representation_root = None
        self.train_df = None
        self.note_groups = None

    def prepare_dataset(self, representation):
        output_dir = Path(os.environ.get("MUSIC_REPR_OUTPUT_DIR"))
        config = MusicRepresentationConfig(
            input_dir=Path(os.environ.get("MUSIC_REPR_INPUT_DIR")),
            output_dir=output_dir,
        )
        storage = build_storage(config)
        self.representation_root = storage.materialize(representation)
        
        if representation in ["note_table"]:
            segments_df = pd.read_parquet(self.representation_root / "notes.parquet")
        else:
            segments_df = pd.read_parquet(self.representation_root / "segments.parquet")
        self.train_df = segments_df[segments_df["maestro_split"] == "train"]

        if representation in ["note_table"]:
            self.train_df = (
                self.train_df[["piece_id", "segment_id"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

    def load_music(self, row, representation_root):
        token_path = representation_root / row["token_file"]
        payload = json.loads(token_path.read_text(encoding="utf-8"))

        return {
            "piece_id": row["piece_id"],
            "segment_id": row["segment_id"],
            "tokens": payload["ids"],
        }

    def load_note_table_music(self, row, representation_root):
        segment_id = row["segment_id"]
        grouped = self._note_table_groups(representation_root)

        if pd.isna(segment_id):
            notes = grouped["piece_id"].get_group(row["piece_id"])
        else:
            notes = grouped["segment_id"].get_group(segment_id)

        notes = notes.sort_values("onset")
        notes["velocity_bin"] = notes["velocity"] // 8

        return {
            "piece_id": row["piece_id"],
            "segment_id": segment_id,
            "pitch": notes["pitch"].tolist(),
            "velocity": notes["velocity"].tolist(),
            "velocity_bin": notes["velocity_bin"].tolist(),
            "onset": notes["onset"].tolist(),
            "duration_bin": notes["duration_bin"].tolist(),
            "delta_onset_bin": notes["delta_onset_bin"].tolist(),
        }

    def _note_table_groups(self, representation_root):
        if self.note_groups is None:
            notes_df = pd.read_parquet(
                representation_root / "notes.parquet", columns=NOTE_TABLE_COLUMNS
            )
            self.note_groups = {
                "piece_id": notes_df.groupby("piece_id", sort=False),
                "segment_id": notes_df.groupby("segment_id", sort=False),
            }
        return self.note_groups

    @abstractmethod
    def loader(self, batch_size) -> DataLoader:
        pass

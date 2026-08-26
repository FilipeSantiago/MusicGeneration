import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from data_processing.music_representations import MusicRepresentationConfig
from storage import build_storage


class BaseDataloader(ABC):

    def __init__(self):
        self.representation_root = None
        self.train_df = None

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

    def load_music(self, row, representation_root):
        token_path = representation_root / row["token_file"]
        payload = json.loads(token_path.read_text(encoding="utf-8"))

        return {
            "piece_id": row["piece_id"],
            "segment_id": row["segment_id"],
            "tokens": payload["ids"],
        }

    @abstractmethod
    def loader(self, batch_size) -> DataLoader:
        pass

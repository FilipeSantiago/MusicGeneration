import json
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

NOTE_TABLE_COLUMNS = [
    "piece_id", "segment_id", "pitch", "velocity", "onset",
    "duration_bin", "delta_onset_bin",
]


def load_token_music(row, representation_root: Path) -> dict:
    """Read one local token file without capturing the parent dataloader."""
    payload = json.loads((representation_root / row["token_file"]).read_text(encoding="utf-8"))
    return {
        "piece_id": row["piece_id"],
        "segment_id": row["segment_id"],
        "tokens": payload["ids"],
    }


class MusicDataset(Dataset):
    def __init__(self, df, representation_root):
        self.df = df.reset_index(drop=True)
        self.representation_root = Path(representation_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return load_token_music(row, self.representation_root)


class NoteTableDataset(Dataset):
    """Read local notes with a cache built separately inside each worker."""

    def __init__(self, df, representation_root, velocity_bin_size: int):
        self.df = df.reset_index(drop=True)
        self.representation_root = Path(representation_root)
        self.velocity_bin_size = velocity_bin_size
        self._groups = None

    def __len__(self):
        return len(self.df)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_groups"] = None
        return state

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self._groups is None:
            notes = pd.read_parquet(
                self.representation_root / "notes.parquet", columns=NOTE_TABLE_COLUMNS
            )
            self._groups = {
                "piece_id": notes.groupby("piece_id", sort=False),
                "segment_id": notes.groupby("segment_id", sort=False),
            }
        segment_id = row["segment_id"]
        if pd.isna(segment_id):
            notes = self._groups["piece_id"].get_group(row["piece_id"])
        else:
            notes = self._groups["segment_id"].get_group(segment_id)
        notes = notes.sort_values("onset")
        return {
            "piece_id": row["piece_id"],
            "segment_id": segment_id,
            "pitch": notes["pitch"].tolist(),
            "velocity": notes["velocity"].tolist(),
            "velocity_bin": (notes["velocity"] // self.velocity_bin_size).tolist(),
            "onset": notes["onset"].tolist(),
            "duration_bin": notes["duration_bin"].tolist(),
            "delta_onset_bin": notes["delta_onset_bin"].tolist(),
        }

"""Worker datasets must not serialize the parent's S3 client."""

import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from data_loader.raw_dataloader import RawDataloader
from data_processing.music_representations.config import MusicRepresentationConfig


@pytest.mark.parametrize("representation", ["remi", "note_table"])
def test_worker_dataset_reads_local_data_with_forkserver(tmp_path, representation):
    root = tmp_path / representation
    root.mkdir()
    loader = RawDataloader()
    loader.representation_root = root
    loader.config = MusicRepresentationConfig(tmp_path / "input", tmp_path)
    loader.config.note_table.velocity_bin_size = 7
    # A boto3 client cannot be pickled. This stands in for the parent-only
    # storage object and catches accidental bound-method capture.
    loader.storage = SimpleNamespace(_client=lambda: None)

    if representation == "remi":
        (root / "tokens.json").write_text(json.dumps({"ids": [3, 4, 5]}), encoding="utf-8")
        loader.train_df = pd.DataFrame({
            "piece_id": ["piece"], "segment_id": ["segment"],
            "token_file": ["tokens.json"],
        })
    else:
        loader.train_df = pd.DataFrame({
            "piece_id": ["piece"], "segment_id": ["segment"],
        })
        pd.DataFrame({
            "piece_id": ["piece", "piece"],
            "segment_id": ["segment", "segment"],
            "pitch": [64, 60],
            "velocity": [80, 70],
            "onset": [2, 1],
            "duration_bin": [3, 2],
            "delta_onset_bin": [1, 0],
        }).to_parquet(root / "notes.parquet")

    batches = loader.loader(
        batch_size=1,
        representation=representation,
        num_workers=2,
        multiprocessing_context="forkserver",
    )
    if representation == "note_table":
        assert batches.dataset[0]["pitch"] == [60, 64]
    pickle.dumps(batches.dataset)
    song = next(iter(batches))[0]
    assert song["piece_id"] == "piece"
    assert song["segment_id"] == "segment"
    if representation == "remi":
        assert song["tokens"] == [3, 4, 5]
    else:
        assert song["pitch"] == [60, 64]
        assert song["velocity_bin"] == [10, 11]
        assert song["duration_bin"] == [2, 3]

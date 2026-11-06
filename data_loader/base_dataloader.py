import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from data_processing.music_representations import MusicRepresentationConfig
from storage import build_storage

LOGGER = logging.getLogger(__name__)

class BaseDataloader(ABC):

    def __init__(self):
        self.representation_root = None
        self.train_df = None
        self.config = None
        self.storage = None

    def prepare_dataset(self, representation):
        output_dir = Path(os.environ["MUSIC_REPR_OUTPUT_DIR"])
        config = MusicRepresentationConfig(
            input_dir=Path(os.environ["MUSIC_REPR_INPUT_DIR"]),
            output_dir=output_dir,
        )
        storage = build_storage(config)
        LOGGER.info(
            "Checking completed %s representation at %s",
            representation,
            storage.location(representation),
        )
        if not storage.exists(representation):
            raise FileNotFoundError(
                f"Missing completed {representation!r} representation. Run "
                "construct_data_representations.py first."
            )
        LOGGER.info("Found completed %s representation; materializing it", representation)
        self.representation_root = storage.materialize(representation)
        LOGGER.info("%s representation available at %s", representation, self.representation_root)
        manifest_path = self.representation_root / "manifest.json"
        LOGGER.info("Reading %s manifest from %s", representation, manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        LOGGER.info("Loaded %s manifest", representation)
        if manifest["representation"] != representation:
            raise ValueError(
                f"Expected {representation!r} manifest at {manifest_path}, "
                f"found {manifest['representation']!r}"
            )
        config = MusicRepresentationConfig.from_dict(dict(manifest["effective_configuration"]))
        # The current output root locates artifacts; the saved build settings
        # determine how training interprets those artifacts.
        config.output_dir = output_dir
        LOGGER.info(
            "Checking completed %s representation at %s",
            config.canonical_dir_name,
            storage.location(config.canonical_dir_name),
        )
        if not storage.exists(config.canonical_dir_name):
            raise FileNotFoundError(
                f"Missing completed {config.canonical_dir_name!r} representation. "
                "Run construct_data_representations.py first."
            )
        LOGGER.info("Found completed %s representation", config.canonical_dir_name)
        self.source_manifest = {
            "representation": manifest["representation"],
            "configuration_hash": manifest["configuration_hash"],
            "source_dataset_fingerprint": manifest["source_dataset_fingerprint"],
            "representation_schema_version": manifest["representation_schema_version"],
        }
        self.config = config
        self.storage = storage
        
        if representation in ["note_table"]:
            table_path = self.representation_root / "notes.parquet"
        else:
            table_path = self.representation_root / "segments.parquet"
        LOGGER.info("Loading %s training table from %s", representation, table_path)
        segments_df = pd.read_parquet(table_path)
        self.train_df = segments_df[segments_df["maestro_split"] == "train"]
        if representation in ["note_table"]:
            self.train_df = (
                self.train_df[["piece_id", "segment_id"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        LOGGER.info("Loaded %s training rows for %s", len(self.train_df), representation)

    @abstractmethod
    def loader(self, batch_size) -> DataLoader:
        pass

"""N-gram training, sampling, and checkpoint behavior."""

from __future__ import annotations

import logging
import pickle
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from data_loader.raw_dataloader import RawDataloader
from models.ngram.knn import KNN
from models.ngram.model import Model
from trainer.settings import TrainSettings

LOGGER = logging.getLogger(__name__)

NOTE_BOS = ("<BOS>",) * 4
NOTE_EOS = ("<EOF>",) * 4
REMI_BOS = -1
REMI_EOS = -1000


class NgramRun:
    checkpoint_name = "model.pkl"

    def __init__(self, dataloader: RawDataloader, settings: TrainSettings):
        self.dataloader = dataloader
        self.settings = settings
        self.representation = (
            "remi" if settings.model == "ngram_remi" else "note_table"
        )
        self.model: Model | None = None
        self.epoch_losses: list[float] = []

    def train(self) -> None:
        load_fn = (
            self.dataloader.load_music
            if self.representation == "remi"
            else self.dataloader.load_note_table_music
        )
        loader = self.dataloader.loader(load_music=load_fn)
        neighbors = None
        if self.representation == "note_table":
            LOGGER.info("Counting note-table tokens for nearest neighbors")
            tokens_by_pitch: dict[int, Counter] = defaultdict(Counter)
            for batch in tqdm(
                loader,
                total=len(loader) if hasattr(loader, "__len__") else None,
                desc="Note-table neighbors",
                unit="batch",
                disable=None,
            ):
                for song in batch:
                    tokens = zip(
                        song["pitch"],
                        song["velocity_bin"],
                        song["delta_onset_bin"],
                        song["duration_bin"],
                        strict=True,
                    )
                    for token in tokens:
                        tokens_by_pitch[token[0]][token] += 1
            neighbors = KNN(
                n=self.settings.knn_neighbors, min_count=10
            ).get_nearest_neighbors(tokens_by_pitch)
            LOGGER.info(
                "Neighbor map ready: %s pitches, %s tokens",
                len(tokens_by_pitch),
                len(neighbors),
            )
        order = (
            self.settings.remi_ngram_order
            if self.representation == "remi"
            else self.settings.note_ngram_order
        )
        self.model = Model(loader=loader, neighbors=neighbors, n=order)
        if self.representation == "remi":
            self.model.remi_train(show_progress=True)
        else:
            self.model.note_table_train(show_progress=True)

    def _trained_model(self) -> Model:
        if self.model is None:
            raise RuntimeError("Train the n-gram model before generating or saving")
        return self.model

    def generate(self, piece_index: int, attempt: int) -> list[Any] | None:
        model = self._trained_model()
        random.seed(f"{self.settings.seed}:{piece_index}:{attempt}")
        if self.representation == "remi":
            context = deque([REMI_BOS] * (model.n - 1), maxlen=model.n - 1)
            eos, limit, predict = (
                REMI_EOS,
                self.settings.remi_max_tokens,
                model.remi_predict,
            )
        else:
            context = deque([NOTE_BOS] * (model.n - 1), maxlen=model.n - 1)
            eos, limit, predict = (
                NOTE_EOS,
                self.settings.note_max_tokens,
                model.note_table_predict,
            )
        result = []
        for _ in range(limit):
            prediction = predict(tuple(context))
            if not prediction:
                return None
            token = prediction[0]
            if token == eos:
                return result
            result.append(token)
            context.append(token)
        return result

    def save(self, path: Path) -> None:
        model = self._trained_model()
        # The model's DataLoader is deliberately excluded from the checkpoint.
        with path.open("wb") as stream:
            pickle.dump(
                {
                    "representation": self.representation,
                    "order": model.n,
                    "counts": model.models,
                    "neighbors": model.neighbors,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

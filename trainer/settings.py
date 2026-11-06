"""Code-defined settings for a training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trainer.spec import MODEL_SPECS

MODEL_NAMES = tuple(MODEL_SPECS)


@dataclass(frozen=True)
class TrainSettings:
    model: str
    output_dir: Path
    seed: int
    generated_pieces: int
    max_attempts: int
    remi_max_tokens: int
    note_max_tokens: int
    remi_ngram_order: int
    note_ngram_order: int
    knn_neighbors: int
    rnn_epochs: int
    representation: str = "remi"

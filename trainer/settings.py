"""Environment-backed settings for a no-argument training run."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from main import DOTENV_PATH, load_dotenv

MODEL_NAMES = ("ngram_remi", "ngram_note_table", "rnn_remi")


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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

    @classmethod
    def from_env(cls) -> TrainSettings:
        load_dotenv(DOTENV_PATH)
        model = os.environ.get("MUSIC_TRAIN_MODEL", "").strip().lower()
        if model not in MODEL_NAMES:
            raise ValueError(
                "MUSIC_TRAIN_MODEL must be one of " + ", ".join(MODEL_NAMES)
            )
        output = os.environ.get("MUSIC_TRAIN_OUTPUT_DIR", "").strip()
        if not output:
            raise ValueError("Set MUSIC_TRAIN_OUTPUT_DIR in .env")
        if not os.environ.get("MUSIC_REPR_INPUT_DIR") or not os.environ.get(
            "MUSIC_REPR_OUTPUT_DIR"
        ):
            raise ValueError("Set MUSIC_REPR_INPUT_DIR and MUSIC_REPR_OUTPUT_DIR in .env")
        seed_raw = os.environ.get("MUSIC_TRAIN_SEED", "20260912").strip()
        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError(f"MUSIC_TRAIN_SEED must be an integer, got {seed_raw!r}") from exc
        remi_order = _positive_int("MUSIC_TRAIN_REMI_N", 3)
        note_order = _positive_int("MUSIC_TRAIN_NOTE_N", 5)
        if min(remi_order, note_order) < 2:
            raise ValueError("N-gram orders must be at least 2")
        return cls(
            model=model,
            output_dir=Path(output).expanduser().resolve(),
            seed=seed,
            generated_pieces=_positive_int("MUSIC_TRAIN_GENERATED_PIECES", 250),
            max_attempts=_positive_int("MUSIC_TRAIN_MAX_ATTEMPTS", 10),
            remi_max_tokens=_positive_int("MUSIC_TRAIN_REMI_MAX_TOKENS", 2048),
            note_max_tokens=_positive_int("MUSIC_TRAIN_NOTE_MAX_TOKENS", 512),
            remi_ngram_order=remi_order,
            note_ngram_order=note_order,
            knn_neighbors=_positive_int("MUSIC_TRAIN_KNN_K", 5),
            rnn_epochs=_positive_int("MUSIC_TRAIN_RNN_EPOCHS", 30),
        )

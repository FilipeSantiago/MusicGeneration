"""Run a code-configured experiment against existing representations."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from tqdm.contrib.logging import logging_redirect_tqdm

from trainer import TrainSettings, run
from trainer.spec import ModelSpec, RNN_REMI

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"


def load_config(model: ModelSpec = RNN_REMI) -> TrainSettings:
    """Create a reusable preset; use ``dataclasses.replace`` to vary settings."""
    return TrainSettings(
        model=model.name,
        representation=model.representation,
        output_dir=ROOT_DIR.parent / "data/MusicGeneration/trained_models",
        seed=20260912,
        generated_pieces=250,
        max_attempts=10,
        remi_max_tokens=2048,
        note_max_tokens=512,
        remi_ngram_order=3,
        note_ngram_order=5,
        knn_neighbors=5,
        rnn_epochs=30,
    )


def train_pipeline(config: TrainSettings, run_name: str | None = None) -> dict:
    """Train and save one uniquely identified run of ``config``."""
    load_dotenv(DOTENV_PATH)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    label = run_name or f"{config.model}-baseline"
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not slug:
        raise ValueError("run_name must contain letters or numbers")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    settings = replace(config, output_dir=config.output_dir / slug / run_id)
    with logging_redirect_tqdm():
        result = run(settings, run_name=label, run_id=run_id)
    LOGGER.info("Validation JSD: %s", result["jsd"])
    return result


if __name__ == "__main__":
    train_pipeline(load_config(RNN_REMI), run_name="rnn-remi-baseline")

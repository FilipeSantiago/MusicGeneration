"""Single entry point: build the configured representations, then train.

Everything is configured through ``.env`` (see ``.env.example``):

- ``MUSIC_REPR_*`` -- which representations to build and where
  (``INPUT_DIR``, ``OUTPUT_DIR``, ``REPRESENTATIONS``, ``OVERWRITE``,
  ``LIMIT``, ``CONFIG``). Representations that already exist are skipped
  unless ``MUSIC_REPR_OVERWRITE`` is true.
- ``MUSIC_TRAIN_*`` -- the model run (``MODEL``, ``REPRESENTATION``,
  ``OUTPUT_DIR``, ...). Leave ``MUSIC_TRAIN_MODEL`` empty to only build data.

The representation the model trains on is always built, even when it is not
listed in ``MUSIC_REPR_REPRESENTATIONS``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.contrib.logging import logging_redirect_tqdm

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.core.builder import BuildStatus
from data_processing.music_representations.core.dataset_builder import (
    MusicDatasetBuilder,
)
from trainer import TrainSettings, run

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"


def resolve_representations(training_representation: str | None = None) -> list[str]:
    """``MUSIC_REPR_REPRESENTATIONS`` (default ``all``) plus the training one."""
    raw = os.getenv("MUSIC_REPR_REPRESENTATIONS") or "all"
    names = [item.strip() for item in raw.split(",") if item.strip()] or ["all"]
    if (
        training_representation
        and "all" not in names
        and training_representation not in names
    ):
        names.append(training_representation)
    return names


def load_config() -> MusicRepresentationConfig:
    input_dir = os.getenv("MUSIC_REPR_INPUT_DIR")
    output_dir = os.getenv("MUSIC_REPR_OUTPUT_DIR")
    if not input_dir or not output_dir:
        raise ValueError("Set MUSIC_REPR_INPUT_DIR and MUSIC_REPR_OUTPUT_DIR in .env")
    limit = os.getenv("MUSIC_REPR_LIMIT")
    payload: dict = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "limit": int(limit) if limit else None,
    }
    config_path = os.getenv("MUSIC_REPR_CONFIG")
    if config_path:
        payload.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    return MusicRepresentationConfig.from_dict(payload)


def training_settings() -> TrainSettings | None:
    """Settings for the training run, or ``None`` when no model is configured."""
    if not os.getenv("MUSIC_TRAIN_MODEL"):
        return None
    return TrainSettings.from_env()


def build_representations(config: MusicRepresentationConfig, names: list[str]) -> bool:
    """Build ``names``; log a summary and return whether every build succeeded."""
    overwrite = os.getenv("MUSIC_REPR_OVERWRITE", "false").lower() == "true"
    results = MusicDatasetBuilder(config).build(names, overwrite=overwrite)
    by_status = {
        status: [r.representation for r in results if r.status is status]
        for status in BuildStatus
    }
    LOGGER.info(
        "built=%s skipped=%s locked=%s failed=%s",
        by_status[BuildStatus.BUILT],
        by_status[BuildStatus.SKIPPED],
        by_status[BuildStatus.LOCKED],
        by_status[BuildStatus.FAILED],
    )
    for result in results:
        if result.status is BuildStatus.BUILT:
            LOGGER.info("%s -> %s", result.representation, result.location)
        elif result.status is BuildStatus.LOCKED:
            LOGGER.info("%s skipped: %s", result.representation, result.message)
        elif result.status is BuildStatus.FAILED:
            LOGGER.error("%s: %s", result.representation, result.error)
    return not by_status[BuildStatus.FAILED]


def main() -> int:
    load_dotenv(DOTENV_PATH)  # real environment variables take precedence
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )
    settings = training_settings()  # validate .env before any long build

    names = resolve_representations(settings.representation if settings else None)
    if not build_representations(config, names):
        return 1

    if settings is None:
        LOGGER.info("MUSIC_TRAIN_MODEL is empty: representations built, no training")
        return 0
    with logging_redirect_tqdm():
        result = run(settings)
    LOGGER.info("Validation JSD: %s", result["jsd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

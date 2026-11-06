"""Build the representations selected by ``MUSIC_REPR_*`` in ``.env``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.core.builder import BuildStatus
from data_processing.music_representations.core.dataset_builder import (
    MusicDatasetBuilder,
)

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"


def resolve_representations() -> list[str]:
    """Return ``MUSIC_REPR_REPRESENTATIONS`` (default ``all``)."""
    raw = os.getenv("MUSIC_REPR_REPRESENTATIONS") or "all"
    names = [item.strip() for item in raw.split(",") if item.strip()] or ["all"]
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
    return 0 if build_representations(config, resolve_representations()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

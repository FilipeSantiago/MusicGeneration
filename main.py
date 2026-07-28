from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from data_processing.music_representations.core.builder import BuildStatus
from data_processing.music_representations.config import MusicRepresentationConfig
from data_processing.music_representations.core.dataset_builder import MusicDatasetBuilder
from data_processing.music_representations.registry import REPRESENTATION_NAMES

LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = ROOT_DIR / ".env"
ENV_PREFIX = "MUSIC_REPR_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build music representations from MIDI or canonical data."
    )
    parser.add_argument(
        "--representation",
        action="append",
        default=[],
        choices=[*REPRESENTATION_NAMES, "all"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def env_value(name: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}")


def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str) -> int | None:
    value = env_value(name)
    if value is None or value == "":
        return None
    return int(value)


def resolve_representations(cli_values: list[str]) -> list[str]:
    if cli_values:
        return cli_values
    env_representations = env_value("REPRESENTATIONS")
    if not env_representations:
        return ["all"]
    return [item.strip() for item in env_representations.split(",") if item.strip()]


def load_config(args: argparse.Namespace) -> MusicRepresentationConfig:
    load_dotenv(DOTENV_PATH)
    input_dir = args.input_dir or env_value("INPUT_DIR")
    output_dir = args.output_dir or env_value("OUTPUT_DIR")
    if not input_dir or not output_dir:
        raise ValueError(
            "Missing input/output directories. Set --input-dir/--output-dir or define "
            "MUSIC_REPR_INPUT_DIR and MUSIC_REPR_OUTPUT_DIR in .env."
        )
    payload: dict = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "limit": args.limit if args.limit is not None else env_int("LIMIT"),
    }
    config_path = args.config or env_value("CONFIG")
    if config_path:
        payload.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    return MusicRepresentationConfig.from_dict(payload)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    overwrite = args.overwrite or env_bool("OVERWRITE")
    results = MusicDatasetBuilder(config).build(
        resolve_representations(args.representation),
        overwrite=overwrite,
    )
    built = [
        result.representation
        for result in results
        if result.status is BuildStatus.BUILT
    ]
    skipped = [
        result.representation
        for result in results
        if result.status is BuildStatus.SKIPPED
    ]
    failed = [result for result in results if result.status is BuildStatus.FAILED]
    LOGGER.info(
        "built=%s skipped=%s failed=%s",
        built,
        skipped,
        [item.representation for item in failed],
    )
    if failed:
        for item in failed:
            LOGGER.error("%s: %s", item.representation, item.error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

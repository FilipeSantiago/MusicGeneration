"""No-argument entry point for a configured music-model training run."""

import logging

from tqdm.contrib.logging import logging_redirect_tqdm

from trainer import TrainSettings, run

LOGGER = logging.getLogger(__name__)


def main() -> int:
    settings = TrainSettings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with logging_redirect_tqdm():
        result = run(settings)
    LOGGER.info("Validation JSD: %s", result["jsd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

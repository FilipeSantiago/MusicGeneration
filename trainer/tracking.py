"""MLflow experiment tracking, enabled when ``MLFLOW_TRACKING_URI`` is set.

``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME`` are read by MLflow itself
from the environment (``.env`` is loaded by ``main.py``). Without a tracking URI
every helper here is a no-op, so tests and offline runs need no server.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from trainer.settings import TrainSettings

LOGGER = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def run_name(settings: TrainSettings, now: datetime | None = None) -> str:
    """``<model>_<representation>_<date>``, e.g. ``rnn_remi_2026-11-05_14-03-22``.

    The model family is the part of ``MUSIC_TRAIN_MODEL`` before the first
    underscore, so ``ngram_note_table`` on ``note_table`` gives ``ngram_note_table_...``
    rather than repeating the representation.
    """
    family = settings.model.split("_", 1)[0]
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{family}_{settings.representation}_{stamp}"


@contextmanager
def start_run(settings: TrainSettings) -> Iterator[None]:
    """Open an MLflow run for ``settings`` and log its parameters."""
    if not enabled():
        yield
        return
    import mlflow

    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if experiment:
        mlflow.set_experiment(experiment)
    name = run_name(settings)
    with mlflow.start_run(run_name=name) as active:
        LOGGER.info("MLflow run %s (%s)", name, active.info.run_id)
        mlflow.set_tags(
            {"model": settings.model, "representation": settings.representation}
        )
        mlflow.log_params(
            {**asdict(settings), "output_dir": str(settings.output_dir)}
        )
        yield


def _active() -> bool:
    if not enabled():
        return False
    import mlflow

    return mlflow.active_run() is not None


def log_metric(key: str, value: float, step: int | None = None) -> None:
    if _active():
        import mlflow

        mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    if _active():
        import mlflow

        mlflow.log_metrics(dict(metrics), step=step)


def log_artifact(path: Path) -> None:
    if _active():
        import mlflow

        mlflow.log_artifact(str(path))

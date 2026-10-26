"""Training workflows for music generation models."""

from .pipeline import run
from .settings import MODEL_NAMES, TrainSettings

__all__ = ["MODEL_NAMES", "TrainSettings", "run"]

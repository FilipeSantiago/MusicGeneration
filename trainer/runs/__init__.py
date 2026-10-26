"""Model run interfaces and implementations."""

from .interface import TrainedModel
from .ngram import NgramRun
from .rnn import RNNRun

__all__ = ["NgramRun", "RNNRun", "TrainedModel"]

"""Select a model run from the configured training model."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from data_loader.raw_dataloader import RawDataloader
from trainer.runs import NgramRun, RNNRun, TrainedModel
from trainer.settings import TrainSettings

type RunBuilder = Callable[[RawDataloader, object, TrainSettings], TrainedModel]


class RunFactory:
    _BUILDERS: ClassVar[dict[str, RunBuilder]] = {
        "ngram_remi": lambda dataloader, _decoder, settings: NgramRun(
            dataloader, settings
        ),
        "ngram_note_table": lambda dataloader, _decoder, settings: NgramRun(
            dataloader, settings
        ),
        "rnn_remi": lambda dataloader, decoder, settings: RNNRun(
            dataloader, decoder, settings
        ),
    }

    @classmethod
    def create(
        cls, settings: TrainSettings, dataloader: RawDataloader, decoder: object
    ) -> TrainedModel:
        try:
            builder = cls._BUILDERS[settings.model]
        except KeyError as exc:
            choices = ", ".join(sorted(cls._BUILDERS))
            raise ValueError(
                f"Unknown training model {settings.model!r}; choose one of {choices}"
            ) from exc
        return builder(dataloader, decoder, settings)

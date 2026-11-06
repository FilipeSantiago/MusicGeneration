"""Select a model run from the configured training model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from data_loader.raw_dataloader import RawDataloader
from trainer.runs import NgramRun, RNNRun, TrainedModel
from trainer.settings import TrainSettings
from trainer.spec import MODEL_SPECS

type RunBuilder = Callable[[RawDataloader, object, TrainSettings], TrainedModel]


@dataclass(frozen=True)
class RunDefinition:
    representation: str
    build: RunBuilder


class RunFactory:
    _RUNS: ClassVar[dict[str, RunDefinition]] = {
        "ngram_remi": RunDefinition(
            MODEL_SPECS["ngram_remi"].representation,
            lambda dataloader, _decoder, settings: NgramRun(dataloader, settings),
        ),
        "ngram_note_table": RunDefinition(
            MODEL_SPECS["ngram_note_table"].representation,
            lambda dataloader, _decoder, settings: NgramRun(dataloader, settings),
        ),
        "rnn_remi": RunDefinition(
            MODEL_SPECS["rnn_remi"].representation,
            lambda dataloader, decoder, settings: RNNRun(dataloader, decoder, settings),
        ),
    }

    @classmethod
    def create(
        cls, settings: TrainSettings, dataloader: RawDataloader, decoder: object
    ) -> TrainedModel:
        try:
            definition = cls._RUNS[settings.model]
        except KeyError as exc:
            choices = ", ".join(sorted(cls._RUNS))
            raise ValueError(
                f"Unknown training model {settings.model!r}; choose one of {choices}"
            ) from exc
        required = definition.representation
        if settings.representation != required:
            raise ValueError(f"{settings.model} trains on {required!r} but settings specify {settings.representation!r}")
        return definition.build(dataloader, decoder, settings)

"""Shared train, generate, validate, and artifact workflow."""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from data_loader.raw_dataloader import RawDataloader
from data_processing.music_representations.decoders import DecodeContext, create_decoder
from data_processing.music_representations.helpers.canonical_loader import (
    filter_canonical_dataset,
    load_canonical_dataset,
    load_canonical_manifest,
)
from data_processing.music_representations.helpers.canonical_writer import (
    write_canonical_dataset,
)
from evaluation.metrics import DatasetEvaluation, MGEvalMetric, MusPyMetrics
from evaluation.parsers import MGEvalAdapter, MusPyAdapter
from trainer.factory import RunFactory
from trainer.runs import TrainedModel
from trainer.settings import TrainSettings

LOGGER = logging.getLogger(__name__)


def _metrics_table(dataset) -> pl.DataFrame:
    LOGGER.info("Extracting MusPy features from %s pieces", dataset.pieces.height)
    muspy = MusPyMetrics().compute(MusPyAdapter().convert(dataset))
    LOGGER.info("Extracting MGEval features from %s pieces", dataset.pieces.height)
    mgeval = MGEvalMetric().compute(MGEvalAdapter().convert(dataset))
    return muspy.join(mgeval, on="piece_id", how="inner")


def _generated_frames(
    trained: TrainedModel, decoder, settings: TrainSettings
) -> Iterator[dict[str, pl.DataFrame]]:
    rejected_attempts = 0
    decode_failures = 0
    retried_pieces = 0
    progress = tqdm(
        range(settings.generated_pieces),
        desc="Generate",
        unit="piece",
        disable=None,
    )
    for piece_index in progress:
        for attempt in range(settings.max_attempts):
            tokens = trained.generate(piece_index, attempt)
            if tokens is None:
                rejected_attempts += 1
                continue
            try:
                frames = decoder.decode(
                    tokens,
                    DecodeContext(
                        piece_id=f"{settings.model}_{piece_index}",
                        metadata={
                            "maestro_split": "generated",
                            "schema_version": decoder.config.schema_version,
                        },
                    ),
                )
            except (ValueError, RuntimeError) as exc:
                decode_failures += 1
                LOGGER.debug(
                    "Could not decode piece %s attempt %s: %s",
                    piece_index,
                    attempt,
                    exc,
                )
                continue
            if frames["notes"].height >= 2:
                if attempt:
                    retried_pieces += 1
                if (piece_index + 1) % 10 == 0:
                    progress.set_postfix(retried=retried_pieces)
                yield frames
                break
            rejected_attempts += 1
        else:
            raise RuntimeError(
                f"Could not generate piece {piece_index} with at least two notes "
                f"in {settings.max_attempts} attempts"
            )
    LOGGER.info(
        "Generated %s pieces; retried pieces=%s, rejected attempts=%s, "
        "decode failures=%s",
        settings.generated_pieces,
        retried_pieces,
        rejected_attempts,
        decode_failures,
    )


def prepare_dataset(settings: TrainSettings):
    """Load training rows and the explicit canonical validation split."""
    representation = "note_table" if settings.model == "ngram_note_table" else "remi"
    dataloader = RawDataloader()
    dataloader.prepare_dataset(representation=representation)
    if dataloader.train_df.empty:
        raise ValueError(f"{representation} has no training rows")
    canonical_root = dataloader.storage.materialize(dataloader.config.canonical_dir_name)
    canonical = load_canonical_dataset(canonical_root)
    validation_ids = (
        canonical.pieces.filter(pl.col("maestro_split") == "validation")
        .get_column("piece_id")
        .to_list()
    )
    validation = filter_canonical_dataset(canonical, validation_ids)
    valid_note_ids = (
        validation.notes.group_by("piece_id")
        .len()
        .filter(pl.col("len") >= 2)
        .get_column("piece_id")
        .to_list()
    )
    validation = filter_canonical_dataset(validation, valid_note_ids)
    if validation.pieces.is_empty():
        raise ValueError("Canonical validation split has no pieces with at least two notes")
    LOGGER.info(
        "Prepared %s training rows and %s validation pieces for %s",
        len(dataloader.train_df),
        validation.pieces.height,
        representation,
    )
    decoder = create_decoder(representation, dataloader.config, dataloader.storage)
    if representation == "remi" and not (
        decoder.representation_dir / "tokenizer.json"
    ).exists():
        raise FileNotFoundError(
            f"Missing REMI tokenizer: {decoder.representation_dir / 'tokenizer.json'}"
        )
    return dataloader, validation, decoder, canonical_root, representation


def train(settings: TrainSettings, dataloader: RawDataloader, decoder) -> TrainedModel:
    """Create and train the model selected by the factory."""
    trained = RunFactory.create(settings, dataloader, decoder)
    trained.train()
    return trained


def generate_dataset(settings: TrainSettings, dataloader: RawDataloader, decoder, trained: TrainedModel,
    canonical_root: Path, representation: str) -> Path:
    """Persist the generated pieces as a canonical dataset."""
    output = settings.output_dir / settings.model
    output.mkdir(parents=True, exist_ok=True)
    generated_root = output / "generated_canonical"
    LOGGER.info(
        "Generating %s pieces into %s", settings.generated_pieces, generated_root
    )
    source_fingerprint = load_canonical_manifest(canonical_root)["source_dataset_fingerprint"]
    round_trip_contract = (
        decoder.tuple_contract() if representation == "note_table" else decoder.contract
    )

    write_canonical_dataset(generated_root,
        _generated_frames(trained, decoder, settings),
        config=dataloader.config,
        atomic=True,
        overwrite=True,
        source_dataset_fingerprint=source_fingerprint,
        known_information_loss_extra=round_trip_contract.describe_loss(),
        extra={
            "generated_by": {
                "model": settings.model,
                "seed": settings.seed,
                "n_pieces": settings.generated_pieces,
            },
            "source_round_trip_contract": round_trip_contract.to_dict(),
        },
    )
    return generated_root


def evaluate(validation, generated_root: Path) -> tuple[dict[str, float], int]:
    """Compare generated music with held-out validation music."""
    generated = load_canonical_dataset(generated_root)
    LOGGER.info(
        "Evaluating %s generated pieces against %s validation pieces",
        generated.pieces.height,
        validation.pieces.height,
    )
    scores = DatasetEvaluation(
        _metrics_table(validation), _metrics_table(generated)
    ).evaluate()
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Validation produced a non-finite JSD score")
    LOGGER.info("Validation JSD scores: %s", scores)
    return scores, generated.pieces.height


def save(settings: TrainSettings, trained: TrainedModel, generated_root: Path, reference_pieces: int,
         generated_pieces: int, scores: dict[str, float]) -> dict[str, Any]:
    """Save the trained model and completed run metrics."""
    output = settings.output_dir / settings.model
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / trained.checkpoint_name
    trained.save(checkpoint)
    result = {
        "model": settings.model,
        "settings": {**asdict(settings), "output_dir": str(settings.output_dir)},
        "seed": settings.seed,
        "reference_split": "validation",
        "reference_pieces": reference_pieces,
        "generated_pieces": generated_pieces,
        "checkpoint": str(checkpoint),
        "generated_dataset": str(generated_root),
        "training_epoch_loss": trained.epoch_losses,
        "jsd": scores,
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved model to %s and metrics to %s", checkpoint, output / "metrics.json")
    return result


def _run_stage(name, action, *args):
    started = perf_counter()
    LOGGER.info("%s started", name)
    try:
        result = action(*args)
    except Exception as exc:
        LOGGER.error("%s failed after %.1fs: %s", name, perf_counter() - started, exc)
        raise
    LOGGER.info("%s complete in %.1fs", name, perf_counter() - started)
    return result


def run(settings: TrainSettings) -> dict[str, Any]:
    random.seed(settings.seed)
    np.random.seed(settings.seed % (2**32))
    torch.manual_seed(settings.seed)
    LOGGER.info(
        "Starting model=%s seed=%s output=%s",
        settings.model,
        settings.seed,
        settings.output_dir / settings.model,
    )
    dataloader, validation, decoder, canonical_root, representation = _run_stage(
        "Prepare dataset", prepare_dataset, settings
    )

    trained = _run_stage("Train", train, settings, dataloader, decoder)
    generated_root = _run_stage(
        "Generate dataset",
        generate_dataset,
        settings,
        dataloader,
        decoder,
        trained,
        canonical_root,
        representation,
    )

    scores, generated_pieces = _run_stage(
        "Evaluate", evaluate, validation, generated_root
    )

    return _run_stage(
        "Save",
        save,
        settings,
        trained,
        generated_root,
        validation.pieces.height,
        generated_pieces,
        scores,
    )

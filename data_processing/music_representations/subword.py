from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from miditok import TokenizerConfig

from .adapters import canonical_piece_to_symusic
from .base import BaseRepresentationBuilder
from .manifest import write_manifest
from .miditok_builders import TOKENIZER_TYPES
from .segmentation import build_segments


class SubwordBuilder(BaseRepresentationBuilder):
    """Train and apply a MidiTok subword model over a base tokenizer."""

    model_name: str
    representation_name = "subword"

    def _build_into(self, output_dir: Path) -> None:
        dataset = self.load_canonical()
        segments = build_segments(dataset.pieces, dataset.notes, self.config)
        base_name = self.config.subword.base_representation
        tokenizer_cls = TOKENIZER_TYPES[base_name]
        tokenizer = tokenizer_cls(TokenizerConfig())

        train_piece_ids = (
            dataset.pieces.filter(pl.col("maestro_split") == "train")
            .get_column("piece_id")
            .to_list()
        )
        train_scores = [
            canonical_piece_to_symusic(dataset, piece_id)
            for piece_id in train_piece_ids
        ]
        train_token_sequences = [
            " ".join(self._normalize_sequence(tokenizer.encode(score)).tokens)
            for score in train_scores
        ]
        tokenizer.train(
            vocab_size=self.config.subword.vocab_size,
            model=self.model_name,
            iterator=train_token_sequences,
        )

        tokens_dir = output_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for segment in segments.iter_rows(named=True):
            piece_id = str(segment["piece_id"])
            score = canonical_piece_to_symusic(dataset, piece_id)
            seq = self._normalize_sequence(tokenizer.encode(score))
            segment_id = segment["segment_id"] or f"{piece_id}_full"
            token_path = tokens_dir / f"{segment_id}.json"
            token_path.write_text(
                json.dumps({"ids": seq.ids, "tokens": seq.tokens}, indent=2),
                encoding="utf-8",
            )
            rows.append(
                {
                    "piece_id": piece_id,
                    "segment_id": segment["segment_id"],
                    "maestro_split": segment["maestro_split"],
                    "token_file": str(token_path.relative_to(output_dir)),
                    "token_count": len(seq.ids),
                }
            )
        pl.DataFrame(rows).write_parquet(output_dir / "segments.parquet")
        tokenizer.save(output_dir)
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=dataset.manifest[
                "source_canonical_schema_version"
            ],
            source_dataset_fingerprint=dataset.manifest["source_dataset_fingerprint"],
            config=self.config,
            pieces=dataset.pieces,
            segments=segments,
            extra={
                "base_representation": base_name,
                "subword_model": self.model_name,
                "subword_training_splits": ["train"],
            },
        )

    @staticmethod
    def _normalize_sequence(seq):
        if isinstance(seq, list):
            return seq[0]
        return seq


class BPEBuilder(SubwordBuilder):
    representation_name = "bpe"
    model_name = "BPE"


class UnigramBuilder(SubwordBuilder):
    representation_name = "unigram"
    model_name = "Unigram"


class WordPieceBuilder(SubwordBuilder):
    representation_name = "wordpiece"
    model_name = "WordPiece"

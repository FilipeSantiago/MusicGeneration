from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from ...core.base import BaseRepresentationBuilder
from ...helpers.adapters import canonical_piece_data_to_symusic
from ...helpers.canonical_loader import (
    iter_canonical_pieces,
    load_canonical_manifest,
    load_canonical_pieces,
)
from ...helpers.segmentation import build_piece_segments
from ...manifest import write_manifest
from .builders import build_tokenizer


class SubwordBuilder(BaseRepresentationBuilder):
    """Train and apply a MidiTok subword model over a base tokenizer."""

    model_name: str
    representation_name = "subword"

    def _build_into(self, output_dir: Path) -> None:
        pieces = load_canonical_pieces(self.canonical_dir)
        manifest = load_canonical_manifest(self.canonical_dir)
        base_name = self.config.subword.base_representation
        # Build the base tokenizer from the effective configuration, not from
        # defaults: otherwise the subword tokens describe a different REMI than the
        # `remi` representation does, and the decoder cannot know which.
        tokenizer = build_tokenizer(self.config, base_name)

        train_token_sequences = [
            " ".join(
                self._normalize_sequence(
                    tokenizer.encode(canonical_piece_data_to_symusic(piece_data))
                ).tokens
            )
            for piece_data in tqdm(
                iter_canonical_pieces(self.canonical_dir),
                total=pieces.height,
                desc=f"{self.representation_name}:train",
                unit="piece",
                leave=False,
            )
            if piece_data.piece["maestro_split"] == "train"
        ]
        tokenizer.note_table_train(
            vocab_size=self.config.subword.vocab_size,
            model=self.model_name,
            iterator=train_token_sequences,
        )

        tokens_dir = output_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for piece_data in tqdm(
            iter_canonical_pieces(self.canonical_dir),
            total=pieces.height,
            desc=self.representation_name,
            unit="piece",
            leave=False,
        ):
            seq = self._normalize_sequence(
                tokenizer.encode(canonical_piece_data_to_symusic(piece_data))
            )
            for segment in build_piece_segments(
                piece_data.piece, piece_data.notes, self.config
            ):
                piece_id = str(segment["piece_id"])
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
                        "encoder": "tokenizer",
                    }
                )
        segments_df = pl.DataFrame(rows)
        segments_df.write_parquet(output_dir / "segments.parquet")
        tokenizer.save(output_dir)
        write_manifest(
            output_dir,
            representation=self.representation_name,
            representation_schema_version=self.config.representation_schema_version,
            source_canonical_schema_version=manifest["source_canonical_schema_version"],
            source_dataset_fingerprint=manifest["source_dataset_fingerprint"],
            config=self.config,
            pieces=pieces,
            segments=segments_df,
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

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
from miditok import (
    REMI,
    TSD,
    CPWord,
    MIDILike,
    Octuple,
    PerTok,
    Structured,
    TokenizerConfig,
)

from .adapters import canonical_piece_to_symusic
from .base import BaseRepresentationBuilder
from .manifest import write_manifest
from .segmentation import build_segments

TOKENIZER_TYPES: dict[str, Callable[..., object]] = {
    "midilike": MIDILike,
    "tsd": TSD,
    "remi": REMI,
    "remi_plus": REMI,
    "structured": Structured,
    "cpword": CPWord,
    "octuple": Octuple,
    "pertok": PerTok,
}


class MidiTokRepresentationBuilder(BaseRepresentationBuilder):
    """Shared MidiTok builder."""

    representation_name = "miditok"

    def tokenizer_name(self) -> str:
        return self.representation_name

    def _tokenizer(self):
        cfg = self.config.miditok
        use_rests = (
            cfg.remi_plus_use_rests
            if self.representation_name == "remi_plus"
            else cfg.use_rests
        )
        use_programs = (
            cfg.remi_plus_use_programs
            if self.representation_name == "remi_plus"
            else cfg.use_programs
        )
        use_sustain_pedals = (
            cfg.remi_plus_use_sustain_pedals
            if self.representation_name == "remi_plus"
            else cfg.use_sustain_pedals
        )
        extra_kwargs = {}
        if self.representation_name == "pertok":
            extra_kwargs.update(
                {
                    "ticks_per_quarter": 480,
                    "use_microtiming": False,
                    "max_microtiming_shift": 0,
                    "num_microtiming_bins": 1,
                    "use_position_toks": True,
                }
            )
        tokenizer_config = TokenizerConfig(
            pitch_range=cfg.pitch_range,
            beat_res=cfg.beat_res,
            num_velocities=cfg.num_velocities,
            use_velocities=cfg.use_velocities,
            use_tempos=cfg.use_tempos,
            use_time_signatures=cfg.use_time_signatures,
            use_sustain_pedals=use_sustain_pedals,
            use_pitch_bends=cfg.use_pitch_bends,
            use_programs=use_programs,
            use_rests=use_rests,
            one_token_stream_for_programs=cfg.one_token_stream_for_programs,
            special_tokens=cfg.special_tokens,
            **extra_kwargs,
        )
        tokenizer_cls = TOKENIZER_TYPES[self.representation_name]
        return tokenizer_cls(tokenizer_config)

    def _build_into(self, output_dir: Path) -> None:
        dataset = self.load_canonical()
        segments = build_segments(dataset.pieces, dataset.notes, self.config)
        tokenizer = self._tokenizer()
        tokens_dir = output_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        mapping_rows: list[dict[str, object]] = []
        for segment in segments.iter_rows(named=True):
            piece_id = str(segment["piece_id"])
            segment_id = segment["segment_id"] or f"{piece_id}_full"
            score = canonical_piece_to_symusic(dataset, piece_id)
            seq = self._encode(tokenizer, score)
            token_path = tokens_dir / f"{segment_id}.json"
            token_path.write_text(
                json.dumps({"ids": seq.ids, "tokens": seq.tokens}, indent=2),
                encoding="utf-8",
            )
            mapping_rows.append(
                {
                    "piece_id": piece_id,
                    "segment_id": segment["segment_id"],
                    "maestro_split": segment["maestro_split"],
                    "start_tick": segment["start_tick"],
                    "end_tick": segment["end_tick"],
                    "token_file": str(token_path.relative_to(output_dir)),
                    "token_count": len(seq.ids),
                }
            )
        pl.DataFrame(mapping_rows).write_parquet(output_dir / "segments.parquet")
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
                "tokenizer_class": tokenizer.__class__.__name__,
                "special_tokens": tokenizer.config.special_tokens,
                "known_information_loss": [
                    "segment-local token exports currently encode whole pieces before mapping segments"
                ],
            },
        )

    def _encode(self, tokenizer, score):
        try:
            seq = tokenizer.encode(score)
            if isinstance(seq, list):
                return seq[0]
            return seq
        except Exception:  # noqa: BLE001
            return self._fallback_encode(score)

    def _fallback_encode(self, score):
        tokens: list[str] = []
        current_time = 0
        for track in score.tracks:
            if self.representation_name in {"structured", "pertok"}:
                tokens.append(f"Program_{track.program}")
            for note in track.notes:
                delta = note.time - current_time
                tokens.extend(
                    [
                        f"TimeShift_{delta}",
                        f"Pitch_{note.pitch}",
                        f"Velocity_{note.velocity}",
                        f"Duration_{note.duration}",
                    ]
                )
                current_time = note.time
        ids = list(range(len(tokens)))
        from miditok.classes import TokSequence

        return TokSequence(ids=ids, tokens=tokens)


class MIDILikeBuilder(MidiTokRepresentationBuilder):
    representation_name = "midilike"


class TSDBuilder(MidiTokRepresentationBuilder):
    representation_name = "tsd"


class REMIBuilder(MidiTokRepresentationBuilder):
    representation_name = "remi"


class REMIPlusBuilder(MidiTokRepresentationBuilder):
    representation_name = "remi_plus"


class StructuredBuilder(MidiTokRepresentationBuilder):
    representation_name = "structured"


class CPWordBuilder(MidiTokRepresentationBuilder):
    representation_name = "cpword"


class OctupleBuilder(MidiTokRepresentationBuilder):
    representation_name = "octuple"


class PerTokBuilder(MidiTokRepresentationBuilder):
    representation_name = "pertok"

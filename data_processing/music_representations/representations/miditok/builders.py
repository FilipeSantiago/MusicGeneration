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
from tqdm.auto import tqdm

from ...config import MusicRepresentationConfig
from ...core.base import BaseRepresentationBuilder
from ...fallback.miditok import build_miditok_fallback_sequence
from ...helpers.adapters import canonical_piece_data_to_symusic
from ...helpers.canonical_loader import (
    iter_canonical_pieces,
    load_canonical_manifest,
    load_canonical_pieces,
)
from ...helpers.segmentation import build_piece_segments
from ...manifest import write_manifest

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


def build_tokenizer(config: MusicRepresentationConfig, representation_name: str):
    """Construct the tokenizer for a representation.

    Module level on purpose: the decoder builds its tokenizer through this same
    function, so encoder and decoder configuration cannot drift apart.
    """
    cfg = config.miditok
    use_rests = (
        cfg.remi_plus_use_rests
        if representation_name == "remi_plus"
        else cfg.use_rests
    )
    use_programs = (
        cfg.remi_plus_use_programs
        if representation_name == "remi_plus"
        else cfg.use_programs
    )
    use_sustain_pedals = (
        cfg.remi_plus_use_sustain_pedals
        if representation_name == "remi_plus"
        else cfg.use_sustain_pedals
    )
    extra_kwargs = {}
    if representation_name == "pertok":
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
    return TOKENIZER_TYPES[representation_name](tokenizer_config)


class MidiTokRepresentationBuilder(BaseRepresentationBuilder):
    """Shared MidiTok builder."""

    representation_name = "miditok"

    def tokenizer_name(self) -> str:
        return self.representation_name

    def _tokenizer(self):
        return build_tokenizer(self.config, self.representation_name)

    def _build_into(self, output_dir: Path) -> None:
        pieces = load_canonical_pieces(self.canonical_dir)
        manifest = load_canonical_manifest(self.canonical_dir)
        tokenizer = self._tokenizer()
        tokens_dir = output_dir / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        mapping_rows: list[dict[str, object]] = []
        for piece_data in tqdm(
            iter_canonical_pieces(self.canonical_dir),
            total=pieces.height,
            desc=self.representation_name,
            unit="piece",
            leave=False,
        ):
            score = canonical_piece_data_to_symusic(piece_data)
            seq, used_fallback = self._encode(tokenizer, score)
            for segment in build_piece_segments(
                piece_data.piece, piece_data.notes, self.config
            ):
                piece_id = str(segment["piece_id"])
                segment_id = segment["segment_id"] or f"{piece_id}_full"
                token_path = tokens_dir / f"{segment_id}.json"
                token_path.write_text(
                    json.dumps(
                        {
                            "ids": seq.ids,
                            "tokens": seq.tokens,
                            "fallback": used_fallback,
                        },
                        indent=2,
                    ),
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
                        "encoder": "fallback" if used_fallback else "tokenizer",
                    }
                )
        segments_df = pl.DataFrame(mapping_rows)
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
                "tokenizer_class": tokenizer.__class__.__name__,
                "special_tokens": tokenizer.config.special_tokens,
                "known_information_loss_extra": [
                    "segment-local token exports currently encode whole pieces before mapping segments"
                ],
            },
        )

    def _encode(self, tokenizer, score) -> tuple[object, bool]:
        try:
            seq = tokenizer.encode(score)
            if isinstance(seq, list):
                seq = seq[0]
            return seq, False
        except Exception:
            # The fallback emits positional ids that no tokenizer can decode, so the
            # caller has to record that this segment is not round-trippable.
            self.logger.exception(
                "%s: tokenizer.encode failed, writing a fallback sequence",
                self.representation_name,
            )
            return build_miditok_fallback_sequence(score, self.representation_name), True


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

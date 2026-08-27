from __future__ import annotations

import json
from typing import Any, ClassVar

import polars as pl
from miditok import TokSequence

from ..fallback.miditok import parse_miditok_fallback_sequence
from ..helpers.canonical_frames import symusic_to_canonical_frames
from ..representations.miditok.builders import TOKENIZER_TYPES, build_tokenizer
from .base import DecodeContext, DecodeError, RepresentationDecoder


class MidiTokDecoder(RepresentationDecoder):
    """Decode a MidiTok token stream back into canonical frames.

    MidiTok decodes into a Score whose ``ticks_per_quarter`` is the tokenizer's own
    beat resolution (8 with the default config, not 480), so the score is resampled
    to the context's time division before it reaches canonical.
    """

    representation_name: ClassVar[str] = "midilike"
    #: Tokenizer class to rebuild; subword models override this with their base.
    tokenizer_name: ClassVar[str | None] = None
    ids_are_encoded: ClassVar[bool] = False

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    def decode(
        self, payload: Any, context: DecodeContext | None = None
    ) -> dict[str, pl.DataFrame]:
        context = context or DecodeContext()
        score = self._to_score(payload, context)
        if score.ticks_per_quarter != context.ticks_per_quarter:
            # MidiTok decodes on the tokenizer's own grid (8 ticks per quarter with
            # the default config), so bring it onto the caller's clock.
            score = score.resample(context.ticks_per_quarter)
        return symusic_to_canonical_frames(
            score, piece_id=context.piece_id, metadata=context.metadata
        )

    def _to_score(self, payload: Any, context: DecodeContext):
        if self._is_fallback(payload):
            return parse_miditok_fallback_sequence(
                list(payload["tokens"]), context.ticks_per_quarter
            )
        sequence = self._as_sequence(payload)
        # REMI and friends only accept a bare TokSequence when they emit one stream.
        tokens = sequence if self.tokenizer.one_token_stream else [sequence]
        return self.tokenizer.decode(tokens)

    @staticmethod
    def _is_fallback(payload: Any) -> bool:
        return isinstance(payload, dict) and bool(payload.get("fallback"))

    def load_payload(
        self, *, piece_id: str, segment_id: str | None = None
    ) -> TokSequence:
        row = self._segment_row(piece_id, segment_id)
        payload = json.loads(
            (self.representation_dir / row["token_file"]).read_text(encoding="utf-8")
        )
        if payload.get("fallback") or row.get("encoder") == "fallback":
            # Fallback ids are positional and mean nothing to the tokenizer; the
            # token strings are the only decodable part, so route them to the
            # fallback parser rather than letting the tokenizer invent music.
            if not payload.get("tokens"):
                raise DecodeError(
                    f"{self.representation_name}: segment {row['segment_id']!r} was "
                    "written by the encoder fallback and has no token strings"
                )
            return {"fallback": True, "tokens": payload["tokens"]}
        return TokSequence(
            ids=payload["ids"],
            tokens=payload["tokens"],
            are_ids_encoded=self.ids_are_encoded,
        )

    def _load_tokenizer(self):
        name = self.tokenizer_name or self.representation_name
        params = self.representation_dir / "tokenizer.json"
        if params.exists():
            # Prefer the tokenizer saved beside the tokens: it is the one that encoded
            # them, vocabulary and all.
            return TOKENIZER_TYPES[name](params=params)
        return build_tokenizer(self.config, name)

    def _as_sequence(self, payload: Any) -> TokSequence:
        if isinstance(payload, TokSequence):
            return payload
        if isinstance(payload, dict):
            return TokSequence(
                ids=payload["ids"],
                tokens=payload.get("tokens"),
                are_ids_encoded=self.ids_are_encoded,
            )
        items = list(payload)
        if items and isinstance(items[0], str):
            return TokSequence(tokens=items, are_ids_encoded=self.ids_are_encoded)
        return TokSequence(ids=items, are_ids_encoded=self.ids_are_encoded)


class MIDILikeDecoder(MidiTokDecoder):
    representation_name = "midilike"


class TSDDecoder(MidiTokDecoder):
    representation_name = "tsd"


class REMIDecoder(MidiTokDecoder):
    representation_name = "remi"


class REMIPlusDecoder(MidiTokDecoder):
    representation_name = "remi_plus"


class StructuredDecoder(MidiTokDecoder):
    representation_name = "structured"


class CPWordDecoder(MidiTokDecoder):
    representation_name = "cpword"


class OctupleDecoder(MidiTokDecoder):
    representation_name = "octuple"


class PerTokDecoder(MidiTokDecoder):
    representation_name = "pertok"

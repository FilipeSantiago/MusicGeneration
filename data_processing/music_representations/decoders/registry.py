from __future__ import annotations

from pathlib import Path

from storage import Storage

from ..config import MusicRepresentationConfig
from .base import RepresentationDecoder
from .canonical import CanonicalDecoder
from .miditok import (
    CPWordDecoder,
    MIDILikeDecoder,
    OctupleDecoder,
    PerTokDecoder,
    REMIDecoder,
    REMIPlusDecoder,
    StructuredDecoder,
    TSDDecoder,
)
from .note_table import NoteTableDecoder
from .piano_roll import PianoRollDecoder
from .subword import BPEDecoder, UnigramDecoder, WordPieceDecoder

type DecoderType = type[RepresentationDecoder]

# Mirrors REPRESENTATION_BUILDERS: every representation that can be built can be
# decoded back to canonical.
REPRESENTATION_DECODERS: dict[str, DecoderType] = {
    "canonical": CanonicalDecoder,
    "note_table": NoteTableDecoder,
    "piano_roll": PianoRollDecoder,
    "midilike": MIDILikeDecoder,
    "tsd": TSDDecoder,
    "remi": REMIDecoder,
    "remi_plus": REMIPlusDecoder,
    "structured": StructuredDecoder,
    "cpword": CPWordDecoder,
    "octuple": OctupleDecoder,
    "pertok": PerTokDecoder,
    "bpe": BPEDecoder,
    "unigram": UnigramDecoder,
    "wordpiece": WordPieceDecoder,
}

DECODER_NAMES = tuple(REPRESENTATION_DECODERS)


def create_decoder(
    name: str,
    config: MusicRepresentationConfig,
    storage: Storage | None = None,
    representation_dir: Path | None = None,
) -> RepresentationDecoder:
    try:
        decoder_cls = REPRESENTATION_DECODERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown representation: {name}") from exc
    return decoder_cls(config, storage, representation_dir)

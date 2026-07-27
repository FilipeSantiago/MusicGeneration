from __future__ import annotations

from .base import BaseRepresentationBuilder
from .builder import MusicRepresentationBuilder
from .canonical import CanonicalBuilder
from .config import MusicRepresentationConfig
from .miditok_builders import (
    CPWordBuilder,
    MIDILikeBuilder,
    OctupleBuilder,
    PerTokBuilder,
    REMIBuilder,
    REMIPlusBuilder,
    StructuredBuilder,
    TSDBuilder,
)
from .note_table import NoteTableBuilder
from .piano_roll import PianoRollBuilder
from .subword import BPEBuilder, UnigramBuilder, WordPieceBuilder

type BuilderType = type[BaseRepresentationBuilder]

REPRESENTATION_BUILDERS: dict[str, BuilderType] = {
    "canonical": CanonicalBuilder,
    "note_table": NoteTableBuilder,
    "piano_roll": PianoRollBuilder,
    "midilike": MIDILikeBuilder,
    "tsd": TSDBuilder,
    "remi": REMIBuilder,
    "remi_plus": REMIPlusBuilder,
    "structured": StructuredBuilder,
    "cpword": CPWordBuilder,
    "octuple": OctupleBuilder,
    "pertok": PerTokBuilder,
    "bpe": BPEBuilder,
    "unigram": UnigramBuilder,
    "wordpiece": WordPieceBuilder,
}

REPRESENTATION_NAMES = tuple(REPRESENTATION_BUILDERS)


def create_builder(
    name: str, config: MusicRepresentationConfig
) -> MusicRepresentationBuilder:
    try:
        builder_cls = REPRESENTATION_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown representation: {name}") from exc
    return builder_cls(config)

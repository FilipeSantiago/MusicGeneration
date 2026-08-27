from __future__ import annotations

from storage import Storage

from .config import MusicRepresentationConfig
from .core.base import BaseRepresentationBuilder
from .core.builder import MusicRepresentationBuilder
from .representations import CanonicalBuilder, NoteTableBuilder, PianoRollBuilder
from .representations.miditok import (
    BPEBuilder,
    CPWordBuilder,
    MIDILikeBuilder,
    OctupleBuilder,
    PerTokBuilder,
    REMIBuilder,
    REMIPlusBuilder,
    StructuredBuilder,
    TSDBuilder,
    UnigramBuilder,
    WordPieceBuilder,
)

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
    name: str,
    config: MusicRepresentationConfig,
    storage: Storage | None = None,
) -> MusicRepresentationBuilder:
    try:
        builder_cls = REPRESENTATION_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown representation: {name}") from exc
    return builder_cls(config, storage)

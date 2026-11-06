"""Model inputs are fixed by the model selected for a run."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    representation: str


NGRAM_REMI = ModelSpec("ngram_remi", "remi")
NGRAM_NOTE_TABLE = ModelSpec("ngram_note_table", "note_table")
RNN_REMI = ModelSpec("rnn_remi", "remi")

MODEL_SPECS = {spec.name: spec for spec in (NGRAM_REMI, NGRAM_NOTE_TABLE, RNN_REMI)}

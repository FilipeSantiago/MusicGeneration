from __future__ import annotations

from .base import DecodeError
from .miditok import MidiTokDecoder


class SubwordDecoder(MidiTokDecoder):
    """Decode BPE / Unigram / WordPiece tokens through the trained vocabulary.

    Unlike the event tokenizers, a subword model cannot be reconstructed from
    configuration alone — the vocabulary is training output — so this refuses to
    guess when the saved tokenizer is missing.
    """

    ids_are_encoded = True

    @property
    def tokenizer_name(self) -> str:  # type: ignore[override]
        return self.config.subword.base_representation

    def _load_tokenizer(self):
        params = self.representation_dir / "tokenizer.json"
        if not params.exists():
            raise DecodeError(
                f"{self.representation_name}: {params} is missing; a trained subword "
                "vocabulary cannot be rebuilt from configuration alone"
            )
        return super()._load_tokenizer()


class BPEDecoder(SubwordDecoder):
    representation_name = "bpe"


class UnigramDecoder(SubwordDecoder):
    representation_name = "unigram"


class WordPieceDecoder(SubwordDecoder):
    representation_name = "wordpiece"

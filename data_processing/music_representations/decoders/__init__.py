"""Decode any representation back into canonical frames.

Canonical is the hub in both directions: representations decode to canonical, and
only canonical writes MIDI (``helpers.adapters.canonical_frames_to_midi``).
"""

from ..contracts import RoundTripContract, contract_for, note_table_tuple_contract
from .base import DecodeContext, DecodeError, RepresentationDecoder
from .registry import DECODER_NAMES, REPRESENTATION_DECODERS, create_decoder

__all__ = [
    "DECODER_NAMES",
    "REPRESENTATION_DECODERS",
    "DecodeContext",
    "DecodeError",
    "RepresentationDecoder",
    "RoundTripContract",
    "contract_for",
    "create_decoder",
    "note_table_tuple_contract",
]

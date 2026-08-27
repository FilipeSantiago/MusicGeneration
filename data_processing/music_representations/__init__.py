"""Music representation preprocessing package."""

from .config import MusicRepresentationConfig
from .contracts import RoundTripContract, contract_for
from .core import (
    BuildResult,
    BuildStatus,
    MusicDatasetBuilder,
    MusicRepresentationBuilder,
)
from .decoders import DECODER_NAMES, DecodeContext, create_decoder
from .registry import REPRESENTATION_NAMES, create_builder

__all__ = [
    "DECODER_NAMES",
    "REPRESENTATION_NAMES",
    "BuildResult",
    "BuildStatus",
    "DecodeContext",
    "MusicDatasetBuilder",
    "MusicRepresentationBuilder",
    "MusicRepresentationConfig",
    "RoundTripContract",
    "contract_for",
    "create_builder",
    "create_decoder",
]

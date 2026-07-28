from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BuildStatus(str, Enum):
    """Representation build outcome."""

    BUILT = "built"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class BuildResult:
    """Structured representation build result."""

    representation: str
    status: BuildStatus
    output_dir: Path
    message: str = ""
    files: list[str] = field(default_factory=list)
    error: str | None = None


class MusicRepresentationBuilder(ABC):
    """Abstract builder for one music representation."""

    @abstractmethod
    def build(self, overwrite: bool = False) -> BuildResult:
        """Build the representation and return a structured result."""

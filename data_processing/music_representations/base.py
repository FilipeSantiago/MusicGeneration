from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .builder import BuildResult, BuildStatus, MusicRepresentationBuilder
from .canonical_loader import load_canonical_dataset
from .config import MusicRepresentationConfig
from .io import build_directory


class BaseRepresentationBuilder(MusicRepresentationBuilder, ABC):
    """Shared builder behavior."""

    representation_name: str

    def __init__(self, config: MusicRepresentationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.representation_name}")

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir / self.representation_name

    @property
    def canonical_dir(self) -> Path:
        return self.config.output_dir / self.config.canonical_dir_name

    def build(self, overwrite: bool = False) -> BuildResult:
        if self.output_dir.exists() and not overwrite:
            return BuildResult(
                representation=self.representation_name,
                status=BuildStatus.SKIPPED,
                output_dir=self.output_dir,
                message="target directory exists",
            )
        try:
            with build_directory(self.output_dir, overwrite) as working_dir:
                if working_dir is None:
                    return BuildResult(
                        representation=self.representation_name,
                        status=BuildStatus.SKIPPED,
                        output_dir=self.output_dir,
                        message="target directory exists",
                    )
                self._build_into(working_dir)
            return BuildResult(
                representation=self.representation_name,
                status=BuildStatus.BUILT,
                output_dir=self.output_dir,
                files=sorted(
                    str(path.relative_to(self.output_dir))
                    for path in self.output_dir.rglob("*")
                    if path.is_file()
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return BuildResult(
                representation=self.representation_name,
                status=BuildStatus.FAILED,
                output_dir=self.output_dir,
                message="build failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def load_canonical(self):
        return load_canonical_dataset(self.canonical_dir)

    @abstractmethod
    def _build_into(self, output_dir: Path) -> None:
        """Write representation files into the provided directory."""

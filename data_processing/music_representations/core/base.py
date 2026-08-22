from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import MusicRepresentationConfig
from ..helpers.canonical_loader import load_canonical_dataset
from storage import BuildState, LocalStorage, Storage
from .builder import BuildResult, BuildStatus, MusicRepresentationBuilder


class BaseRepresentationBuilder(MusicRepresentationBuilder, ABC):
    """Shared builder behavior."""

    representation_name: str

    def __init__(
        self, config: MusicRepresentationConfig, storage: Storage | None = None
    ):
        self.config = config
        self.storage = storage if storage is not None else LocalStorage(config.output_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.representation_name}")

    @property
    def output_dir(self) -> Path:
        return self.config.output_dir / self.representation_name

    @property
    def canonical_dir(self) -> Path:
        return self.storage.materialize(self.config.canonical_dir_name)

    def build(self, overwrite: bool = False) -> BuildResult:
        name = self.representation_name
        state = self.storage.state(name)
        if state is BuildState.IN_PROGRESS:
            return self._result(
                BuildStatus.LOCKED,
                f"another run is building it ({self.storage.holder(name)})",
            )
        if state is BuildState.COMPLETE and not overwrite:
            return self._result(BuildStatus.SKIPPED, "already built")
        try:
            with self.storage.staging(name, overwrite) as working_dir:
                if working_dir is None:
                    # Lost the race between the check above and claiming the lease.
                    return self._result(BuildStatus.LOCKED, "another run claimed it")
                self._build_into(working_dir)
            return self._result(BuildStatus.BUILT, files=self.storage.list_files(name))
        except Exception as exc:  # noqa: BLE001
            return self._result(
                BuildStatus.FAILED,
                "build failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _result(
        self,
        status: BuildStatus,
        message: str = "",
        *,
        files: list[str] | None = None,
        error: str | None = None,
    ) -> BuildResult:
        return BuildResult(
            representation=self.representation_name,
            status=status,
            output_dir=self.output_dir,
            location=self.storage.location(self.representation_name),
            message=message,
            files=files or [],
            error=error,
        )

    def load_canonical(self):
        return load_canonical_dataset(self.canonical_dir)

    @abstractmethod
    def _build_into(self, output_dir: Path) -> None:
        """Write representation files into the provided directory."""

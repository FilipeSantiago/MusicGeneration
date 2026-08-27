from __future__ import annotations

from dataclasses import dataclass

from storage import BuildState, Storage, build_storage

from ..config import MusicRepresentationConfig
from ..registry import REPRESENTATION_NAMES, create_builder
from .builder import BuildResult, BuildStatus

ALL_REPRESENTATIONS = list(REPRESENTATION_NAMES)
CANONICAL = "canonical"


@dataclass(slots=True)
class MusicDatasetBuilder:
    """Orchestrate canonical and derived representation builds."""

    config: MusicRepresentationConfig

    def build(
        self, representations: list[str], overwrite: bool = False
    ) -> list[BuildResult]:
        storage = build_storage(self.config)
        ordered = self._dependency_order(self._resolve(representations))
        canonical_state = storage.state(self.config.canonical_dir_name)
        if (
            any(name != CANONICAL for name in ordered)
            and canonical_state is not BuildState.COMPLETE
            and CANONICAL not in ordered
        ):
            ordered = [CANONICAL, *ordered]

        results: list[BuildResult] = []
        canonical_ready = canonical_state is BuildState.COMPLETE
        for name in ordered:
            if name != CANONICAL and not canonical_ready:
                # Every derived representation reads canonical back; without it they
                # would only fail on materialize().
                results.append(self._blocked(storage, name))
                continue
            result = create_builder(name, self.config, storage).build(
                overwrite=overwrite
            )
            results.append(result)
            if name == CANONICAL and result.status is not BuildStatus.FAILED:
                canonical_ready = result.status is not BuildStatus.LOCKED
            if result.status is BuildStatus.FAILED:
                break
        return results

    def _blocked(self, storage: Storage, name: str) -> BuildResult:
        return BuildResult(
            representation=name,
            status=BuildStatus.LOCKED,
            output_dir=self.config.output_dir / name,
            location=storage.location(name),
            message=f"{self.config.canonical_dir_name} is being built by another run",
        )

    def _resolve(self, representations: list[str]) -> list[str]:
        if not representations or "all" in representations:
            return ALL_REPRESENTATIONS
        return representations

    @staticmethod
    def _dependency_order(names: list[str]) -> list[str]:
        unique = list(dict.fromkeys(names))
        if CANONICAL in unique:
            return [CANONICAL, *[name for name in unique if name != CANONICAL]]
        return unique

from __future__ import annotations

from dataclasses import dataclass

from .builder import BuildResult, BuildStatus
from .config import MusicRepresentationConfig
from .registry import REPRESENTATION_NAMES, create_builder

ALL_REPRESENTATIONS = list(REPRESENTATION_NAMES)


@dataclass(slots=True)
class MusicDatasetBuilder:
    """Orchestrate canonical and derived representation builds."""

    config: MusicRepresentationConfig

    def build(
        self, representations: list[str], overwrite: bool = False
    ) -> list[BuildResult]:
        names = self._resolve(representations)
        ordered = self._dependency_order(names)
        results: list[BuildResult] = []
        canonical_exists = (
            self.config.output_dir / self.config.canonical_dir_name
        ).exists()
        if (
            any(name != "canonical" for name in ordered)
            and not canonical_exists
            and "canonical" not in ordered
        ):
            ordered = ["canonical", *ordered]
        for name in ordered:
            builder = create_builder(name, self.config)
            result = builder.build(overwrite=overwrite)
            results.append(result)
            if result.status is BuildStatus.FAILED:
                break
        return results

    def _resolve(self, representations: list[str]) -> list[str]:
        if not representations or "all" in representations:
            return ALL_REPRESENTATIONS
        return representations

    @staticmethod
    def _dependency_order(names: list[str]) -> list[str]:
        unique = list(dict.fromkeys(names))
        if "canonical" in unique:
            return ["canonical", *[name for name in unique if name != "canonical"]]
        return unique

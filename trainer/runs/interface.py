"""Shared contract for trained model runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TrainedModel(Protocol):
    checkpoint_name: str
    epoch_losses: list[float]

    def train(self) -> None: ...

    def generate(self, piece_index: int, attempt: int) -> list[Any] | None: ...

    def save(self, path: Path) -> None: ...

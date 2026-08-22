from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path

from data_processing.music_representations.helpers.io import build_directory, list_output_files
from .base import LOCK_PREFIX, MANIFEST_NAME, Storage
from .lease import Lease


class LocalStorage(Storage):
    """Filesystem backend rooted at a single output directory.

    Also used as the staging and cache tier of :class:`~.s3_storage.S3Storage`, so its
    leases double as the guard against two runs sharing one output directory.
    """

    def __init__(self, root: Path, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def location(self, name: str) -> str:
        return str(self.path(name))

    def list_files(self, name: str) -> list[str]:
        target = self.path(name)
        return list_output_files(target) if target.is_dir() else []

    def materialize(self, name: str) -> Path:
        return self.path(name)

    # -- backend primitives --------------------------------------------------

    def _has_manifest(self, name: str) -> bool:
        return (self.path(name) / MANIFEST_NAME).is_file()

    def _open_staging(self, name: str):
        return build_directory(self.path(name), True)

    def _commit(self, name: str) -> None:
        """No-op: ``build_directory`` already moved the build into place."""

    def _lock_path(self, name: str) -> Path:
        return self.root / LOCK_PREFIX / f"{name}.json"

    def _lease_read(self, name: str) -> Lease | None:
        try:
            return Lease.from_bytes(self._lock_path(name).read_bytes())
        except (FileNotFoundError, NotADirectoryError):
            return None

    def _lease_create(self, name: str, lease: Lease) -> bool:
        """``O_EXCL`` is the filesystem's compare-and-swap, mirroring S3's IfNoneMatch."""
        path = self._lock_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(lease.to_bytes())
        return True

    def _lease_write(self, name: str, lease: Lease) -> None:
        path = self._lock_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(lease.to_bytes())
        temporary.replace(path)

    def _lease_delete(self, name: str) -> None:
        path = self._lock_path(name)
        path.unlink(missing_ok=True)
        with suppress(OSError):
            # Leaves no trace in the output directory once the last lease is gone.
            path.parent.rmdir()

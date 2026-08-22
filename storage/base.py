from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from .lease import DEFAULT_TTL_SECONDS, Lease

LOGGER = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
LOCK_PREFIX = "_locks"


class StorageError(RuntimeError):
    """Raised when a storage backend cannot complete an operation."""


class BuildState(StrEnum):
    """What the backend knows about a representation."""

    MISSING = "missing"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class Storage(ABC):
    """Destination for built representations.

    A representation is addressed by ``name`` (``canonical``, ``remi``, ...). Builders
    always write into a real local directory handed to them by :meth:`staging`; where
    that directory ends up afterwards is the backend's business.

    Leasing lives here rather than in the subclasses: acquire, heartbeat and release are
    identical for every backend, and only the four storage primitives underneath them
    (:meth:`_lease_read` and friends) differ.
    """

    def __init__(
        self,
        *,
        lease_ttl: float = DEFAULT_TTL_SECONDS,
        leases_enabled: bool = True,
    ):
        self.lease_ttl = lease_ttl
        self.leases_enabled = leases_enabled

    # -- state ---------------------------------------------------------------

    def state(self, name: str) -> BuildState:
        """Whether ``name`` is built, being built right now, or absent.

        An expired lease reads as :attr:`BuildState.MISSING`: the run that left it
        behind is gone, so the work is up for grabs again.
        """
        if self._has_manifest(name):
            return BuildState.COMPLETE
        lease = self._read_lease(name)
        if lease is not None and not lease.is_expired():
            return BuildState.IN_PROGRESS
        return BuildState.MISSING

    def exists(self, name: str) -> bool:
        """Return whether a completed build of ``name`` is already stored."""
        return self.state(name) is BuildState.COMPLETE

    def holder(self, name: str) -> str:
        """Describe the run currently building ``name``, if any."""
        lease = self._read_lease(name)
        if lease is None or lease.is_expired():
            return ""
        return lease.describe()

    # -- building ------------------------------------------------------------

    @contextmanager
    def staging(self, name: str, overwrite: bool = False) -> Iterator[Path | None]:
        """Yield a writable local directory, or ``None`` when the build is skipped.

        ``None`` means either "already built" or "another run holds the lease". On
        clean exit the directory is committed to the backend; on error it is discarded
        and the exception propagates. The lease is always released.
        """
        if not overwrite and self._has_manifest(name):
            yield None
            return
        if not self.leases_enabled:
            with self._build(name) as working_dir:
                yield working_dir
            return
        lease = self._acquire(name)
        if lease is None:
            yield None
            return
        stop_heartbeat = self._start_heartbeat(name, lease)
        try:
            with self._build(name) as working_dir:
                yield working_dir
        finally:
            stop_heartbeat()
            self._release(name, lease)

    @contextmanager
    def _build(self, name: str) -> Iterator[Path]:
        with self._open_staging(name) as working_dir:
            yield working_dir
        self._commit(name)

    # -- leasing -------------------------------------------------------------

    def _acquire(self, name: str) -> Lease | None:
        """Claim ``name``, or return ``None`` if another live run already has it."""
        existing = self._read_lease(name)
        if existing is not None:
            if not existing.is_expired():
                return None
            LOGGER.warning(
                "Reclaiming the expired lease on %s from %s", name, existing.describe()
            )
            self._lease_delete(name)
        lease = Lease.new(self.lease_ttl)
        if not self._lease_create(name, lease):
            LOGGER.info("Another run claimed %s first", name)
            return None
        return lease

    def _release(self, name: str, lease: Lease) -> None:
        try:
            current = self._read_lease(name)
            if current is not None and current.owner != lease.owner:
                LOGGER.warning(
                    "Lease on %s is now held by %s; leaving it alone",
                    name,
                    current.describe(),
                )
                return
            self._lease_delete(name)
        except StorageError:
            LOGGER.warning("Could not release the lease on %s", name, exc_info=True)

    def _start_heartbeat(self, name: str, lease: Lease):
        """Keep the lease alive for as long as the build runs."""
        interval = max(1.0, self.lease_ttl / 3)
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(interval):
                try:
                    self._lease_write(name, lease.renewed(self.lease_ttl))
                except Exception:  # a lost heartbeat must not kill the build
                    LOGGER.warning(
                        "Could not refresh the lease on %s", name, exc_info=True
                    )

        thread = threading.Thread(target=beat, name=f"lease-{name}", daemon=True)
        thread.start()

        def cancel() -> None:
            stop.set()
            thread.join(timeout=5)

        return cancel

    def _read_lease(self, name: str) -> Lease | None:
        if not self.leases_enabled:
            return None
        return self._lease_read(name)

    # -- backend primitives --------------------------------------------------

    @abstractmethod
    def _has_manifest(self, name: str) -> bool:
        """Whether the completion marker for ``name`` is stored."""

    @abstractmethod
    def _open_staging(self, name: str):
        """Context manager yielding the local directory a build writes into."""

    @abstractmethod
    def _commit(self, name: str) -> None:
        """Publish the freshly staged build. A no-op when staging is the destination."""

    @abstractmethod
    def _lease_read(self, name: str) -> Lease | None:
        """Return the stored lease for ``name``, or ``None``."""

    @abstractmethod
    def _lease_create(self, name: str, lease: Lease) -> bool:
        """Atomically store ``lease`` only if none exists. False if one already did."""

    @abstractmethod
    def _lease_write(self, name: str, lease: Lease) -> None:
        """Overwrite the stored lease unconditionally (heartbeat)."""

    @abstractmethod
    def _lease_delete(self, name: str) -> None:
        """Remove the stored lease, tolerating its absence."""

    # -- reading -------------------------------------------------------------

    @abstractmethod
    def list_files(self, name: str) -> list[str]:
        """Return stored file paths for ``name``, relative to its root."""

    @abstractmethod
    def materialize(self, name: str) -> Path:
        """Return a local directory holding ``name``, fetching it if needed."""

    @abstractmethod
    def location(self, name: str) -> str:
        """Return a human-readable location for ``name``."""

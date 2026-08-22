from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

DEFAULT_TTL_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class Lease:
    """Claim on a representation, held by one run while it builds it.

    The expiry lives in the payload rather than in backend metadata so the same
    record means the same thing on the filesystem and in S3.
    """

    owner: str
    host: str
    pid: int
    acquired_at: str
    expires_at: str

    @classmethod
    def new(cls, ttl_seconds: float) -> Lease:
        now = datetime.now(tz=UTC)
        return cls(
            owner=uuid4().hex,
            host=socket.gethostname(),
            pid=os.getpid(),
            acquired_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat(),
        )

    def renewed(self, ttl_seconds: float) -> Lease:
        deadline = datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds)
        return replace(self, expires_at=deadline.isoformat())

    def is_expired(self) -> bool:
        """True when the holder's deadline has passed, or the record is unreadable."""
        try:
            deadline = datetime.fromisoformat(self.expires_at)
        except (TypeError, ValueError):
            return True
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        return deadline <= datetime.now(tz=UTC)

    def describe(self) -> str:
        return f"host={self.host} pid={self.pid} since={self.acquired_at}"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), indent=2, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> Lease | None:
        """Parse a stored record, treating anything unreadable as no lease at all."""
        try:
            payload = json.loads(raw.decode("utf-8"))
            return cls(
                owner=str(payload["owner"]),
                host=str(payload.get("host", "?")),
                pid=int(payload.get("pid", 0)),
                acquired_at=str(payload.get("acquired_at", "")),
                expires_at=str(payload["expires_at"]),
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError):
            return None

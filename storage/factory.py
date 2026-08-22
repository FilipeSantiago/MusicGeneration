from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from data_processing.music_representations.config import MusicRepresentationConfig
from .base import Storage
from .lease import DEFAULT_TTL_SECONDS
from .local_storage import LocalStorage
from .s3_storage import S3Storage

TRUTHY = {"1", "true", "yes", "on"}


def build_storage(
    config: MusicRepresentationConfig,
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
) -> Storage:
    """Return the S3 backend when ``AWS_BUCKET`` is set, the local one otherwise.

    This is the only place that decides which backend is in use; ``env`` and ``client``
    are injectable so the S3 path can be exercised without credentials or network.
    """
    env = os.environ if env is None else env
    leasing = {
        "lease_ttl": _lease_ttl(env),
        "leases_enabled": not _flag(env, "MUSIC_REPR_LOCK_DISABLED"),
    }
    cache = LocalStorage(config.output_dir, **leasing)
    bucket = _clean(env, "AWS_BUCKET")
    if not bucket:
        return cache
    return S3Storage(
        bucket=bucket,
        prefix=_clean(env, "MUSIC_REPR_S3_PREFIX") or config.output_dir.name,
        cache=cache,
        client=client if client is not None else _build_client(env),
        keep_local=_flag(env, "MUSIC_REPR_KEEP_LOCAL"),
        cache_exempt={config.canonical_dir_name},
        **leasing,
    )


def _clean(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _flag(env: Mapping[str, str], name: str) -> bool:
    return _clean(env, name).lower() in TRUTHY


def _lease_ttl(env: Mapping[str, str]) -> float:
    raw = _clean(env, "MUSIC_REPR_LOCK_TTL")
    if not raw:
        return DEFAULT_TTL_SECONDS
    try:
        ttl = float(raw)
    except ValueError as exc:
        raise ValueError(f"MUSIC_REPR_LOCK_TTL must be a number, got {raw!r}") from exc
    if ttl <= 0:
        raise ValueError(f"MUSIC_REPR_LOCK_TTL must be positive, got {ttl}")
    return ttl


def _build_client(env: Mapping[str, str]) -> Any:
    import boto3

    access_key = _clean(env, "AWS_ACCESS_KEY")
    secret_key = _clean(env, "AWS_SECRET_ACCESS_KEY")
    session = boto3.session.Session(
        # The project uses AWS_ACCESS_KEY, not boto3's own AWS_ACCESS_KEY_ID, so the
        # credentials have to be passed explicitly. When they are absent, fall through
        # to boto3's chain (instance role, ~/.aws/credentials, ...).
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        region_name=_clean(env, "AWS_REGION") or None,
    )
    return session.client("s3", endpoint_url=_clean(env, "AWS_ENDPOINT_URL") or None)

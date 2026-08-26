from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping
from functools import cache as memoize
from pathlib import Path
from typing import Any

from data_processing.music_representations.helpers.io import build_directory
from .base import LOCK_PREFIX, MANIFEST_NAME, Storage, StorageError
from .lease import Lease
from .local_storage import LocalStorage

LOGGER = logging.getLogger(__name__)

_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}
_TAKEN_CODES = {"412", "PreconditionFailed"}
_UNSUPPORTED_CODES = {"NotImplemented", "InvalidArgument", "InvalidRequest"}


@memoize
def _boto_errors() -> tuple[type[BaseException], ...]:
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:  # pragma: no cover - boto3 is a declared dependency
        return ()
    return (BotoCoreError, ClientError)


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return ""
    return str(response.get("Error", {}).get("Code", ""))


def _clean(env: Mapping[str, str], name: str) -> str:
    return (env.get(name) or "").strip()


def _default_cache_root(env: Mapping[str, str]) -> Path:
    raw = _clean(env, "MUSIC_REPR_OUTPUT_DIR")
    return Path(raw) if raw else Path(".music_repr_cache")


def _default_prefix(env: Mapping[str, str], cache: LocalStorage) -> str:
    configured = _clean(env, "MUSIC_REPR_S3_PREFIX").strip("/")
    if configured:
        return configured
    output_dir = _clean(env, "MUSIC_REPR_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir).name
    if cache.root.name:
        return cache.root.name
    return "music-representations"


def _build_client(env: Mapping[str, str]) -> Any:
    import boto3

    access_key = _clean(env, "AWS_ACCESS_KEY")
    secret_key = _clean(env, "AWS_SECRET_ACCESS_KEY")
    session = boto3.session.Session(
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        region_name=_clean(env, "AWS_REGION") or None,
    )
    return session.client("s3", endpoint_url=_clean(env, "AWS_ENDPOINT_URL") or None)


class S3Storage(Storage):
    """S3 backend that stages builds through a local :class:`LocalStorage` cache.

    Builders need real paths (``pq.ParquetWriter``, ``np.savez_compressed`` and
    MidiTok's ``tokenizer.save`` all write files themselves), so each build is written
    locally first and uploaded once it is complete. The local copy is then removed
    unless it is exempt (``canonical`` is read back by every other representation) or
    ``keep_local`` is set.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        cache: LocalStorage | None = None,
        client: Any | None = None,
        keep_local: bool = False,
        cache_exempt: Iterable[str] = (),
        env: Mapping[str, str] | None = None,
        **kwargs,
    ):
        env = os.environ if env is None else env
        bucket = bucket.strip() if bucket is not None else _clean(env, "AWS_BUCKET")
        if not bucket:
            raise ValueError("S3Storage requires bucket=... or AWS_BUCKET")
        cache = cache if cache is not None else LocalStorage(_default_cache_root(env), **kwargs)
        prefix = prefix.strip("/") if prefix is not None else _default_prefix(env, cache)
        client = client if client is not None else _build_client(env)
        super().__init__(**kwargs)
        self.bucket = bucket
        self.prefix = prefix
        self.keep_local = keep_local
        self._cache = cache
        self._client = client
        self._cache_exempt = set(cache_exempt)
        self._conditional_writes = True

    # -- addressing ---------------------------------------------------------

    def key(self, name: str, relative: str = "") -> str:
        parts = [part for part in (self.prefix, name, relative) if part]
        return "/".join(parts)

    def lock_key(self, name: str) -> str:
        return self.key(LOCK_PREFIX, f"{name}.json")

    def location(self, name: str) -> str:
        return f"s3://{self.bucket}/{self.key(name)}"

    # -- reading -------------------------------------------------------------

    def list_files(self, name: str) -> list[str]:
        root = self.key(name) + "/"
        return sorted(key.removeprefix(root) for key in self._iter_keys(root))

    def materialize(self, name: str) -> Path:
        """Return a local copy, downloading it once if the cache is cold."""
        local_dir = self._cache.materialize(name)
        if not local_dir.is_dir():
            self._download(name, local_dir)
        return local_dir

    def download_key(self, key: str, destination: str | Path) -> Path:
        """Download one exact S3 key to ``destination`` and return the local path."""
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._call(
            self._client.download_file,
            f"download {key}",
            self.bucket,
            key,
            str(path),
        )
        return path

    # -- backend primitives --------------------------------------------------

    def _has_manifest(self, name: str) -> bool:
        """True when the completion marker exists.

        S3 has no atomic directory swap, so ``manifest.json`` is uploaded last and
        treated as the marker: a partial upload reads as "not built" and is retried
        rather than silently accepted.
        """
        return self._head(self.key(name, MANIFEST_NAME)) is not None

    def _open_staging(self, name: str):
        # Straight to the cache's directory rather than through its own staging(): the
        # S3 lease is the one that matters, and nesting a second local lease inside it
        # would just be noise.
        return build_directory(self._cache.path(name), True)

    def _commit(self, name: str) -> None:
        local_dir = self._cache.materialize(name)
        self._replace(name, local_dir)
        if not self.keep_local and name not in self._cache_exempt:
            shutil.rmtree(local_dir, ignore_errors=True)

    # -- leasing --------------------------------------------------------------

    def _lease_read(self, name: str) -> Lease | None:
        try:
            response = self._client.get_object(
                Bucket=self.bucket, Key=self.lock_key(name)
            )
        except self._client.exceptions.ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return None
            raise StorageError(f"Cannot read the lease on {name}: {exc}") from exc
        except _boto_errors() as exc:
            raise StorageError(f"Cannot read the lease on {name}: {exc}") from exc
        return Lease.from_bytes(response["Body"].read())

    def _lease_create(self, name: str, lease: Lease) -> bool:
        """Conditional put, so two runs racing for the same name cannot both win."""
        if not self._conditional_writes:
            self._lease_write(name, lease)
            return True
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=self.lock_key(name),
                Body=lease.to_bytes(),
                IfNoneMatch="*",
            )
        except self._client.exceptions.ClientError as exc:
            code = _error_code(exc)
            if code in _TAKEN_CODES:
                return False
            if code in _UNSUPPORTED_CODES:
                # Older S3-compatible endpoints reject IfNoneMatch. Leasing without it
                # still prevents accidental duplicate runs, just not a true race.
                LOGGER.warning(
                    "%s does not support conditional writes; leases are best-effort",
                    self.bucket,
                )
                self._conditional_writes = False
                self._lease_write(name, lease)
                return True
            raise StorageError(f"Cannot claim {name}: {exc}") from exc
        except _boto_errors() as exc:
            raise StorageError(f"Cannot claim {name}: {exc}") from exc
        return True

    def _lease_write(self, name: str, lease: Lease) -> None:
        self._call(
            self._client.put_object,
            f"refresh the lease on {name}",
            Bucket=self.bucket,
            Key=self.lock_key(name),
            Body=lease.to_bytes(),
        )

    def _lease_delete(self, name: str) -> None:
        self._call(
            self._client.delete_object,
            f"release the lease on {name}",
            Bucket=self.bucket,
            Key=self.lock_key(name),
        )

    # -- transfers ----------------------------------------------------------

    def _head(self, key: str) -> dict | None:
        try:
            return self._client.head_object(Bucket=self.bucket, Key=key)
        except self._client.exceptions.ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return None
            raise StorageError(f"Cannot check s3://{self.bucket}/{key}: {exc}") from exc
        except _boto_errors() as exc:
            raise StorageError(f"Cannot check s3://{self.bucket}/{key}: {exc}") from exc

    def _replace(self, name: str, local_dir: Path) -> None:
        self._delete(name)
        for path in self._upload_order(local_dir):
            relative = path.relative_to(local_dir).as_posix()
            self._call(
                self._client.upload_file,
                f"upload {relative} to {self.location(name)}",
                str(path),
                self.bucket,
                self.key(name, relative),
            )
        LOGGER.info("Uploaded %s to %s", name, self.location(name))

    @staticmethod
    def _upload_order(local_dir: Path) -> list[Path]:
        """All files, with ``manifest.json`` last so it marks a complete upload."""
        manifest = local_dir / MANIFEST_NAME
        files = sorted(path for path in local_dir.rglob("*") if path.is_file())
        ordered = [path for path in files if path != manifest]
        if manifest.is_file():
            ordered.append(manifest)
        return ordered

    def _delete(self, name: str) -> None:
        root = self.key(name) + "/"
        keys = list(self._iter_keys(root))
        for start in range(0, len(keys), 1000):
            batch = [{"Key": key} for key in keys[start : start + 1000]]
            self._delete_batch(name, batch)

    def _delete_batch(self, name: str, batch: list[dict[str, str]]) -> None:
        self._call(
            self._client.delete_objects,
            f"delete stale objects under {self.location(name)}",
            Bucket=self.bucket,
            Delete={"Objects": batch},
        )

    def _download(self, name: str, local_dir: Path) -> None:
        root = self.key(name) + "/"
        keys = list(self._iter_keys(root))
        if not keys:
            raise StorageError(f"Nothing stored at {self.location(name)}")
        for key in keys:
            destination = local_dir / key.removeprefix(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._call(
                self._client.download_file,
                f"download {key}",
                self.bucket,
                key,
                str(destination),
            )
        LOGGER.info("Downloaded %s from %s", name, self.location(name))

    def _iter_keys(self, root: str) -> Iterator[str]:
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": root}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._call(
                self._client.list_objects_v2,
                f"list s3://{self.bucket}/{root}",
                **kwargs,
            )
            for item in response.get("Contents", []):
                yield item["Key"]
            if not response.get("IsTruncated"):
                return
            token = response.get("NextContinuationToken")
            if not token:
                return

    def _call(self, operation: Any, description: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except (self._client.exceptions.ClientError, *_boto_errors()) as exc:
            raise StorageError(f"Failed to {description}: {exc}") from exc

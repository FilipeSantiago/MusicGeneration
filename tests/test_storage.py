from __future__ import annotations

import json
import shutil
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_processing.music_representations.config import MusicRepresentationConfig
from storage import (
    BuildState,
    Lease,
    LocalStorage,
    S3Storage,
    Storage,
    StorageError,
    build_storage,
)


class FakeClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """In-memory stand-in for the subset of the boto3 S3 client we use.

    Honours ``IfNoneMatch`` so lease races behave the way real S3 does.
    """

    class exceptions:  # mirrors the boto3 client attribute
        ClientError = FakeClientError

    def __init__(self, conditional_writes: bool = True) -> None:
        self.objects: dict[str, bytes] = {}
        self.upload_log: list[str] = []
        self.fail_on: str | None = None
        self.conditional_writes = conditional_writes
        self._guard = threading.Lock()

    # -- objects ---------------------------------------------------------

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": FakeBody(self.objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs) -> dict:
        if "IfNoneMatch" in kwargs and not self.conditional_writes:
            raise FakeClientError("NotImplemented")
        with self._guard:
            if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
                raise FakeClientError("PreconditionFailed")
            self.objects[Key] = Body
        return {}

    def delete_object(self, *, Bucket: str, Key: str) -> dict:
        self.objects.pop(Key, None)
        return {}

    # -- transfers -------------------------------------------------------

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        if self.fail_on == "upload_file":
            raise FakeClientError("AccessDenied")
        self.objects[key] = Path(filename).read_bytes()
        self.upload_log.append(key)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        if key not in self.objects:
            raise FakeClientError("NoSuchKey")
        Path(filename).write_bytes(self.objects[key])

    def list_objects_v2(self, **kwargs) -> dict:
        keys = sorted(k for k in self.objects if k.startswith(kwargs["Prefix"]))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def delete_objects(self, *, Bucket: str, Delete: dict) -> dict:
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}


def _populate(directory: Path | None, *, extra: str = "data.parquet") -> None:
    assert directory is not None
    (directory / "nested").mkdir(parents=True, exist_ok=True)
    (directory / extra).write_bytes(b"payload")
    (directory / "nested" / "chunk.npz").write_bytes(b"chunk")
    (directory / "manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


@pytest.fixture()
def s3(tmp_path: Path) -> tuple[S3Storage, FakeS3Client]:
    client = FakeS3Client()
    storage = S3Storage(
        bucket="test-bucket",
        prefix="processed",
        cache=LocalStorage(tmp_path / "cache"),
        client=client,
        cache_exempt={"canonical"},
    )
    return storage, client


@pytest.fixture(params=["local", "s3"])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Storage:
    """Both backends, so the shared contract is tested against each."""
    if request.param == "local":
        return LocalStorage(tmp_path / "output")
    return S3Storage(
        bucket="test-bucket",
        prefix="processed",
        cache=LocalStorage(tmp_path / "cache"),
        client=FakeS3Client(),
    )


# --- the shared Storage contract -------------------------------------------


def test_staging_commits_and_reports_completion(storage: Storage) -> None:
    assert storage.state("remi") is BuildState.MISSING

    with storage.staging("remi") as working_dir:
        _populate(working_dir)

    assert storage.state("remi") is BuildState.COMPLETE
    assert storage.exists("remi") is True
    assert storage.list_files("remi") == [
        "data.parquet",
        "manifest.json",
        "nested/chunk.npz",
    ]
    assert (storage.materialize("remi") / "nested" / "chunk.npz").read_bytes() == b"chunk"


def test_state_is_in_progress_while_a_build_holds_the_lease(storage: Storage) -> None:
    assert storage.state("remi") is BuildState.MISSING
    with storage.staging("remi") as working_dir:
        assert storage.state("remi") is BuildState.IN_PROGRESS
        assert "pid=" in storage.holder("remi")
        _populate(working_dir)
    assert storage.state("remi") is BuildState.COMPLETE
    assert storage.holder("remi") == ""


def test_a_second_run_is_turned_away_while_the_lease_is_held(storage: Storage) -> None:
    with storage.staging("remi") as first:
        with storage.staging("remi") as second:
            assert second is None
        _populate(first)


def test_a_completed_build_is_skipped_without_overwrite(storage: Storage) -> None:
    with storage.staging("remi") as working_dir:
        _populate(working_dir)

    with storage.staging("remi", overwrite=False) as working_dir:
        assert working_dir is None

    with storage.staging("remi", overwrite=True) as working_dir:
        _populate(working_dir, extra="replacement.parquet")
    assert "replacement.parquet" in storage.list_files("remi")


def test_the_lease_is_released_after_a_failed_build(storage: Storage) -> None:
    with pytest.raises(RuntimeError), storage.staging("remi") as working_dir:
        _populate(working_dir)
        raise RuntimeError("boom")

    assert storage.state("remi") is BuildState.MISSING
    with storage.staging("remi") as working_dir:
        assert working_dir is not None
        _populate(working_dir)
    assert storage.state("remi") is BuildState.COMPLETE


def test_an_expired_lease_is_reclaimed(storage: Storage) -> None:
    stale = replace(
        Lease.new(900),
        expires_at=(datetime.now(tz=UTC) - timedelta(seconds=1)).isoformat(),
    )
    assert storage._lease_create("remi", stale) is True
    assert storage.state("remi") is BuildState.MISSING
    assert storage.holder("remi") == ""

    with storage.staging("remi") as working_dir:
        assert working_dir is not None
        _populate(working_dir)
    assert storage.state("remi") is BuildState.COMPLETE


def test_release_leaves_a_lease_that_was_taken_over(storage: Storage) -> None:
    mine = Lease.new(900)
    theirs = Lease.new(900)
    storage._lease_create("remi", theirs)

    storage._release("remi", mine)

    current = storage._lease_read("remi")
    assert current is not None and current.owner == theirs.owner


def test_an_unreadable_lease_is_treated_as_absent(storage: Storage) -> None:
    assert Lease.from_bytes(b"not json at all") is None


def test_leases_can_be_switched_off(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path, leases_enabled=False)
    with storage.staging("remi") as first:
        assert storage.state("remi") is BuildState.MISSING  # no lease to observe
        with storage.staging("remi") as second:
            assert second is not None  # nothing stops a concurrent build
        _populate(first)
    assert storage.state("remi") is BuildState.COMPLETE


# --- backend specifics ------------------------------------------------------


def test_local_storage_leaves_no_lock_directory_behind(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    with storage.staging("remi") as working_dir:
        _populate(working_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["remi"]


def test_local_storage_discards_the_staging_directory_on_error(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    with pytest.raises(RuntimeError), storage.staging("remi") as working_dir:
        _populate(working_dir)
        raise RuntimeError("boom")
    assert storage.exists("remi") is False
    assert list(tmp_path.iterdir()) == []


def test_s3_storage_uploads_every_file_with_the_manifest_last(s3) -> None:
    storage, client = s3
    with storage.staging("remi") as working_dir:
        _populate(working_dir)

    assert client.upload_log == [
        "processed/remi/data.parquet",
        "processed/remi/nested/chunk.npz",
        "processed/remi/manifest.json",
    ]
    assert storage.location("remi") == "s3://test-bucket/processed/remi"


def test_s3_storage_reports_a_manifestless_prefix_as_missing(s3) -> None:
    storage, client = s3
    client.objects["processed/remi/data.parquet"] = b"orphan"
    assert storage.state("remi") is BuildState.MISSING


def test_s3_leases_live_outside_the_representation_prefix(s3) -> None:
    storage, client = s3
    with storage.staging("remi") as working_dir:
        assert "processed/_locks/remi.json" in client.objects
        assert storage.list_files("remi") == []  # the lease is not a result file
        _populate(working_dir)

    assert "processed/_locks/remi.json" not in client.objects
    assert storage.list_files("remi") == [
        "data.parquet",
        "manifest.json",
        "nested/chunk.npz",
    ]


def test_s3_overwrite_removes_stale_objects_but_not_the_lease(s3) -> None:
    storage, client = s3
    with storage.staging("remi") as working_dir:
        _populate(working_dir)

    with storage.staging("remi", overwrite=True) as working_dir:
        assert "processed/_locks/remi.json" in client.objects
        (working_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert sorted(client.objects) == ["processed/remi/manifest.json"]


def test_s3_acquire_loses_the_race_without_raising(s3) -> None:
    storage, _ = s3
    assert storage._lease_create("remi", Lease.new(900)) is True
    assert storage._lease_create("remi", Lease.new(900)) is False


def test_s3_falls_back_when_conditional_writes_are_unsupported(tmp_path: Path) -> None:
    client = FakeS3Client(conditional_writes=False)
    storage = S3Storage(
        bucket="test-bucket",
        prefix="processed",
        cache=LocalStorage(tmp_path / "cache"),
        client=client,
    )
    assert storage._lease_create("remi", Lease.new(900)) is True
    assert storage._conditional_writes is False
    assert "processed/_locks/remi.json" in client.objects


def test_s3_storage_drops_the_local_copy_but_keeps_exempt_names(s3) -> None:
    storage, _ = s3
    with storage.staging("remi") as working_dir:
        _populate(working_dir)
    with storage.staging("canonical") as working_dir:
        _populate(working_dir)

    assert not storage._cache.path("remi").exists()
    assert storage._cache.path("canonical").is_dir()


def test_s3_storage_keep_local_retains_everything(tmp_path: Path) -> None:
    storage = S3Storage(
        bucket="test-bucket",
        prefix="processed",
        cache=LocalStorage(tmp_path / "cache"),
        client=FakeS3Client(),
        keep_local=True,
    )
    with storage.staging("remi") as working_dir:
        _populate(working_dir)
    assert (tmp_path / "cache" / "remi" / "manifest.json").is_file()


def test_s3_storage_materialize_downloads_once(s3) -> None:
    storage, client = s3
    with storage.staging("remi") as working_dir:
        _populate(working_dir)
    assert not storage._cache.path("remi").exists()

    downloaded = storage.materialize("remi")
    assert (downloaded / "nested" / "chunk.npz").read_bytes() == b"chunk"

    client.objects.clear()  # a second call must not hit the client again
    assert storage.materialize("remi") == downloaded


def test_s3_storage_materialize_rejects_an_empty_prefix(s3) -> None:
    storage, _ = s3
    with pytest.raises(StorageError, match="Nothing stored"):
        storage.materialize("remi")


def test_s3_storage_download_key_fetches_one_exact_key(s3, tmp_path: Path) -> None:
    storage, client = s3
    client.objects["processed/remi/tokens.json"] = b'{"tokens":[1,2,3]}'

    destination = tmp_path / "downloads" / "tokens.json"
    downloaded = storage.download_key("processed/remi/tokens.json", destination)

    assert downloaded == destination
    assert destination.read_bytes() == b'{"tokens":[1,2,3]}'


def test_s3_storage_download_key_wraps_missing_keys(s3, tmp_path: Path) -> None:
    storage, _ = s3

    with pytest.raises(StorageError, match="Failed to download processed/missing.json"):
        storage.download_key("processed/missing.json", tmp_path / "missing.json")


def test_s3_storage_can_default_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeS3Client()
    calls: list[dict[str, object]] = []

    class FakeSession:
        def __init__(
            self,
            *,
            aws_access_key_id: str | None,
            aws_secret_access_key: str | None,
            region_name: str | None,
        ) -> None:
            calls.append(
                {
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key,
                    "region_name": region_name,
                }
            )

        def client(self, service_name: str, endpoint_url: str | None = None):
            calls.append(
                {"service_name": service_name, "endpoint_url": endpoint_url}
            )
            return client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(session=SimpleNamespace(Session=FakeSession)))
    storage = S3Storage(
        env={
            "AWS_BUCKET": "env-bucket",
            "AWS_ACCESS_KEY": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_REGION": "us-east-1",
            "AWS_ENDPOINT_URL": "http://localhost:9000",
            "MUSIC_REPR_OUTPUT_DIR": str(tmp_path / "cache"),
        }
    )

    assert storage.bucket == "env-bucket"
    assert storage.prefix == "cache"
    assert storage._cache.root == tmp_path / "cache"
    assert storage._client is client
    assert calls == [
        {
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "region_name": "us-east-1",
        },
        {"service_name": "s3", "endpoint_url": "http://localhost:9000"},
    ]


def test_s3_storage_wraps_client_errors(s3) -> None:
    storage, client = s3
    client.fail_on = "upload_file"
    with pytest.raises(StorageError, match="Failed to upload"), storage.staging(
        "remi"
    ) as working_dir:
        _populate(working_dir)


# --- factory ----------------------------------------------------------------


def _config(tmp_path: Path) -> MusicRepresentationConfig:
    return MusicRepresentationConfig(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "maestro-v3.0.0_processed",
    )


def test_build_storage_defaults_to_local(tmp_path: Path) -> None:
    storage = build_storage(_config(tmp_path), env={})
    assert isinstance(storage, LocalStorage)
    assert storage.root == tmp_path / "maestro-v3.0.0_processed"
    assert storage.leases_enabled is True
    assert storage.lease_ttl == 900


def test_build_storage_ignores_a_blank_bucket(tmp_path: Path) -> None:
    storage = build_storage(_config(tmp_path), env={"AWS_BUCKET": "  "})
    assert isinstance(storage, LocalStorage)


def test_build_storage_selects_s3_and_derives_the_prefix(tmp_path: Path) -> None:
    storage = build_storage(
        _config(tmp_path), env={"AWS_BUCKET": "my-bucket"}, client=FakeS3Client()
    )
    assert isinstance(storage, S3Storage)
    assert storage.bucket == "my-bucket"
    assert storage.prefix == "maestro-v3.0.0_processed"
    assert storage.keep_local is False


def test_build_storage_honours_the_optional_variables(tmp_path: Path) -> None:
    storage = build_storage(
        _config(tmp_path),
        env={
            "AWS_BUCKET": "my-bucket",
            "MUSIC_REPR_S3_PREFIX": "/datasets/maestro/",
            "MUSIC_REPR_KEEP_LOCAL": "true",
            "MUSIC_REPR_LOCK_TTL": "60",
            "MUSIC_REPR_LOCK_DISABLED": "yes",
        },
        client=FakeS3Client(),
    )
    assert storage.prefix == "datasets/maestro"
    assert storage.keep_local is True
    assert storage.lease_ttl == 60
    assert storage.leases_enabled is False


def test_build_storage_rejects_a_nonsense_ttl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a number"):
        build_storage(_config(tmp_path), env={"MUSIC_REPR_LOCK_TTL": "soon"})
    with pytest.raises(ValueError, match="must be positive"):
        build_storage(_config(tmp_path), env={"MUSIC_REPR_LOCK_TTL": "0"})


# --- end to end -------------------------------------------------------------


def test_pipeline_builds_through_s3_without_leaving_local_output(
    base_config: MusicRepresentationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """canonical is built and uploaded, then read back from S3 by note_table."""
    from data_processing.music_representations.core import dataset_builder as module
    from data_processing.music_representations.core.builder import BuildStatus

    client = FakeS3Client()
    monkeypatch.setattr(
        module,
        "build_storage",
        lambda config: build_storage(
            config, env={"AWS_BUCKET": "test-bucket"}, client=client
        ),
    )

    results = module.MusicDatasetBuilder(base_config).build(["note_table"])
    assert [result.status for result in results] == [BuildStatus.BUILT] * 2
    assert [result.representation for result in results] == ["canonical", "note_table"]
    assert results[-1].location == (
        f"s3://test-bucket/{base_config.output_dir.name}/note_table"
    )

    prefix = base_config.output_dir.name
    assert f"{prefix}/canonical/manifest.json" in client.objects
    assert f"{prefix}/note_table/notes.parquet" in client.objects
    assert not [key for key in client.objects if "/_locks/" in key]
    # canonical stays on disk as the read cache; nothing else does
    assert sorted(p.name for p in base_config.output_dir.iterdir()) == ["canonical"]

    # A fresh run with a cold cache finds canonical in S3 and downloads it.
    shutil.rmtree(base_config.output_dir)
    results = module.MusicDatasetBuilder(base_config).build(["piano_roll"])
    assert [result.representation for result in results] == ["piano_roll"]
    assert results[0].status is BuildStatus.BUILT
    assert f"{prefix}/piano_roll/segments.parquet" in client.objects

    # Re-running without --overwrite skips.
    results = module.MusicDatasetBuilder(base_config).build(["piano_roll"])
    assert results[0].status is BuildStatus.SKIPPED


def test_a_locked_canonical_blocks_its_dependents(
    base_config: MusicRepresentationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_processing.music_representations.core import dataset_builder as module
    from data_processing.music_representations.core.builder import BuildStatus

    client = FakeS3Client()
    storage = build_storage(
        base_config, env={"AWS_BUCKET": "test-bucket"}, client=client
    )
    monkeypatch.setattr(module, "build_storage", lambda config: storage)

    # Another run is midway through canonical.
    storage._lease_create("canonical", Lease.new(900))

    results = module.MusicDatasetBuilder(base_config).build(["note_table"])
    assert [result.status for result in results] == [BuildStatus.LOCKED] * 2
    assert "another run" in results[0].message
    assert results[1].message.startswith("canonical is being built")
    assert not [key for key in client.objects if key.endswith("manifest.json")]


def test_two_concurrent_runs_never_build_the_same_representation_twice(
    base_config: MusicRepresentationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data_processing.music_representations.core import dataset_builder as module
    from data_processing.music_representations.core.builder import BuildStatus

    client = FakeS3Client()
    monkeypatch.setattr(
        module,
        "build_storage",
        lambda config: build_storage(
            config, env={"AWS_BUCKET": "test-bucket"}, client=client
        ),
    )

    names = ["canonical", "note_table", "piano_roll"]
    collected: list[list] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        collected.append(module.MusicDatasetBuilder(base_config).build(list(names)))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    results = [result for batch in collected for result in batch]
    assert not [result for result in results if result.status is BuildStatus.FAILED]
    for name in names:
        built = [
            result
            for result in results
            if result.representation == name and result.status is BuildStatus.BUILT
        ]
        assert len(built) <= 1, f"{name} was built more than once"
    # Between them the two runs leave every manifest in place and no lease behind.
    for name in names:
        assert f"{base_config.output_dir.name}/{name}/manifest.json" in client.objects
    assert not [key for key in client.objects if "/_locks/" in key]

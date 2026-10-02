from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import MusicRepresentationConfig
from ..manifest import write_manifest
from ..schemas import CANONICAL_FRAME_NAMES, TABLE_NAMES, TABLE_SCHEMAS
from .io import build_directory

LOGGER = logging.getLogger(__name__)

# Table name to the key the per-piece frames use for it; the inverse of
# ``canonical_loader._FRAME_TO_FIELD``. Built here rather than imported from
# ``representations.canonical`` so that ``helpers`` never points at
# ``representations``.
_FRAME_KEY_BY_TABLE = dict(zip(TABLE_NAMES, CANONICAL_FRAME_NAMES, strict=True))


def write_canonical_tables(
    output_dir: Path, frames: Iterable[Mapping[str, pl.DataFrame]]
) -> pl.DataFrame:
    """Stream per-piece canonical frames into the eight parquet tables.

    ``frames`` is consumed exactly once, one piece at a time, straight into the
    open writers: a few hundred generated pieces never all sit in RAM. Progress
    reporting belongs to the caller (wrap the iterable in ``tqdm``).

    Returns the freshly written ``pieces`` table, which the manifest needs.
    """
    output_dir = Path(output_dir)
    empty = True
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                pq.ParquetWriter(
                    output_dir / f"{name}.parquet", schema, compression="zstd"
                )
            )
            for name, schema in TABLE_SCHEMAS.items()
        }
        for index, piece_frames in enumerate(frames):
            empty = False
            # Validate up front: a missing key discovered halfway through the
            # writers would leave the tables desynchronised for this piece.
            _require_frame_keys(piece_frames, index)
            for name, writer in writers.items():
                table = piece_frames[_FRAME_KEY_BY_TABLE[name]].to_arrow()
                try:
                    # The cast is mandatory: polars widens the uint8/int16
                    # columns on ``to_arrow()``, which the writer would reject.
                    writer.write_table(table.cast(TABLE_SCHEMAS[name]))
                except pa.ArrowInvalid as error:
                    raise ValueError(
                        f"Piece at index {index} has a '{name}' frame that does not "
                        f"match the canonical schema: {error}"
                    ) from error

    if empty:
        LOGGER.warning("No canonical frames written to %s; tables are empty", output_dir)

    # The parquet footers are only flushed when the ExitStack unwinds, so the
    # read-back cannot happen inside the ``with`` block.
    pieces = pl.read_parquet(output_dir / "pieces.parquet")
    _require_unique_piece_ids(pieces)
    return pieces


def canonical_dataset_fingerprint(pieces: pl.DataFrame) -> str:
    """Hash a canonical ``pieces`` table by piece id and source checksum.

    Same algorithm as ``CanonicalBuilder._dataset_fingerprint``, so a dataset
    written through either path fingerprints identically.
    """
    payload = "|".join(
        f"{row['piece_id']}:{row['source_sha256']}"
        for row in pieces.sort("piece_id").iter_rows(named=True)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_canonical_dataset(
    output_dir: Path,
    frames: Iterable[Mapping[str, pl.DataFrame]],
    *,
    config: MusicRepresentationConfig,
    representation: str = "canonical",
    representation_schema_version: str | None = None,
    source_canonical_schema_version: str | None = None,
    source_dataset_fingerprint: str | None = None,
    known_information_loss_extra: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
    atomic: bool = False,
    overwrite: bool = False,
) -> Path | None:
    """Write per-piece canonical frames as a loadable canonical dataset.

    The result is the layout ``load_canonical_dataset`` expects: eight flat
    parquet files plus ``manifest.json`` at ``output_dir``.

    With ``atomic=True`` the tables are built in a sibling temp directory and
    swapped in only on success, so a crash mid-stream leaves any pre-existing
    dataset untouched; if the target exists and ``overwrite`` is False, nothing
    is written and ``None`` is returned. With ``atomic=False`` the files are
    written in place into ``output_dir`` (created if needed) and ``overwrite``
    is ignored -- that mode is for callers that already own staging and commit.

    ``source_dataset_fingerprint`` defaults to the fingerprint of the pieces
    just written, which is only meaningful for datasets built from files with
    real ``source_sha256`` values. Generated data should pass the fingerprint of
    the canonical corpus the model was trained on instead.
    """
    output_dir = Path(output_dir)
    if atomic:
        with build_directory(output_dir, overwrite) as staging_dir:
            if staging_dir is None:
                LOGGER.info("Skipping %s: already exists and overwrite is False", output_dir)
                return None
            _write_into(
                staging_dir,
                frames,
                config=config,
                representation=representation,
                representation_schema_version=representation_schema_version,
                source_canonical_schema_version=source_canonical_schema_version,
                source_dataset_fingerprint=source_dataset_fingerprint,
                known_information_loss_extra=known_information_loss_extra,
                extra=extra,
            )
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_into(
        output_dir,
        frames,
        config=config,
        representation=representation,
        representation_schema_version=representation_schema_version,
        source_canonical_schema_version=source_canonical_schema_version,
        source_dataset_fingerprint=source_dataset_fingerprint,
        known_information_loss_extra=known_information_loss_extra,
        extra=extra,
    )
    return output_dir


def _write_into(
    target_dir: Path,
    frames: Iterable[Mapping[str, pl.DataFrame]],
    *,
    config: MusicRepresentationConfig,
    representation: str,
    representation_schema_version: str | None,
    source_canonical_schema_version: str | None,
    source_dataset_fingerprint: str | None,
    known_information_loss_extra: Sequence[str],
    extra: Mapping[str, Any] | None,
) -> None:
    """Write the tables then the manifest into one directory.

    Private because ``target_dir`` is the staging directory under ``atomic=True``:
    ``write_manifest`` has to run last (it lists the output files) and against
    the directory the tables actually went into, which is not the caller's path.
    """
    pieces = write_canonical_tables(target_dir, frames)
    # write_manifest pops "known_information_loss_extra" out of a copy of extra,
    # so merging here cannot clobber a value the caller passed the other way.
    merged_extra = dict(extra or {})
    merged_extra["known_information_loss_extra"] = [
        *merged_extra.get("known_information_loss_extra", []),
        *known_information_loss_extra,
    ]
    write_manifest(
        target_dir,
        representation=representation,
        representation_schema_version=(
            representation_schema_version
            if representation_schema_version is not None
            else config.representation_schema_version
        ),
        source_canonical_schema_version=(
            source_canonical_schema_version
            if source_canonical_schema_version is not None
            else config.schema_version
        ),
        source_dataset_fingerprint=(
            source_dataset_fingerprint
            if source_dataset_fingerprint is not None
            else canonical_dataset_fingerprint(pieces)
        ),
        config=config,
        pieces=pieces,
        segments=None,
        extra=merged_extra,
    )


def _require_frame_keys(piece_frames: Mapping[str, pl.DataFrame], index: int) -> None:
    missing = [name for name in CANONICAL_FRAME_NAMES if name not in piece_frames]
    if missing:
        raise ValueError(
            f"Piece at index {index} is missing canonical frame(s): "
            f"{', '.join(missing)}"
        )


def _require_unique_piece_ids(pieces: pl.DataFrame) -> None:
    if pieces.is_empty():
        return
    duplicates = (
        pieces.group_by("piece_id")
        .len()
        .filter(pl.col("len") > 1)
        .sort("piece_id")
        .get_column("piece_id")
        .to_list()
    )
    if duplicates:
        raise ValueError(
            "Canonical pieces must have unique piece_id; duplicated: "
            f"{', '.join(map(str, duplicates))}"
        )

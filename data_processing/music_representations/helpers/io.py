from __future__ import annotations

import json
import logging
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

LOGGER = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def list_output_files(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


@contextmanager
def build_directory(target_dir: Path, overwrite: bool):
    """Yield a writable directory and atomically replace the target on success."""
    target_dir = target_dir.resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and not overwrite:
        yield None
        return

    tmp_dir = Path(
        mkdtemp(prefix=f".{target_dir.name}.tmp.", dir=str(target_dir.parent))
    )
    try:
        yield tmp_dir
        if target_dir.exists():
            if target_dir.parent == target_dir:
                raise ValueError(f"Invalid target directory: {target_dir}")
            shutil.rmtree(target_dir)
        tmp_dir.replace(target_dir)
    except Exception:
        LOGGER.exception("Build failed for %s", target_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def maybe_skip(target_dir: Path, overwrite: bool) -> bool:
    return target_dir.exists() and not overwrite

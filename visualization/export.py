"""High-level exports: one piece, or a labelled comparison of several.

Both functions write a single mp4 and return its path, ready for e.g.
``mlflow.log_artifact(path)``::

    from visualization import ComparisonItem, export_comparison, export_piece

    export_piece(piece, "out/sample.mp4", "portrait", max_duration_sec=30)
    export_comparison(
        [
            ComparisonItem(reference, "Reference", "MAESTRO · Chopin, Op. 10 No. 4"),
            ComparisonItem(sample, "5-gram · REMI", "early baseline"),
        ],
        "out/comparison.mp4",
        "portrait",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from data_processing.music_representations.helpers.canonical_loader import (
    CanonicalPiece,
)

from .compose import (
    AudioSynthesizer,
    Card,
    Clip,
    Frames,
    as_frames,
    default_audio_synthesizer,
    render_sequence,
)
from .config import RenderConfig
from .formats import DEFAULT_FORMAT, VideoFormat, format_config
from .timeline import Timeline

Piece = CanonicalPiece | Frames


@dataclass(frozen=True)
class ComparisonItem:
    """One labelled clip of a comparison.

    ``start_sec=None`` picks the busiest ``clip_sec`` window of the piece.
    """

    piece: Piece
    label: str
    sublabel: str | None = None
    start_sec: float | None = None


def _window(
    piece: Piece,
    start_sec: float | None,
    end_sec: float | None,
    max_duration_sec: float | None,
) -> tuple[float, float | None]:
    if start_sec is None and max_duration_sec is not None:
        start_sec = Timeline.from_frames(as_frames(piece)).busiest_window(
            max_duration_sec
        )
    start = start_sec or 0.0
    if max_duration_sec is not None:
        end_sec = (
            min(end_sec, start + max_duration_sec)
            if end_sec
            else start + max_duration_sec
        )
    return start, end_sec


def export_piece(
    piece: Piece,
    out_path: Path | str,
    video_format: str | VideoFormat = DEFAULT_FORMAT,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_duration_sec: float | None = None,
    caption: str | None = None,
    subcaption: str | None = None,
    config: RenderConfig | None = None,
    audio_synthesizer: AudioSynthesizer = default_audio_synthesizer,
    faststart: bool = True,
    progress: bool = False,
) -> Path:
    """Export one piece as a piano-roll video in ``video_format``.

    - ``start_sec``/``end_sec`` cut the piece; with only ``max_duration_sec`` the
      busiest window of that length is used.
    - ``caption``/``subcaption`` replace the title bar with a large text panel.
    - ``config`` supplies any other :class:`RenderConfig` settings; the format
      sets the frame size and layout on top of it.
    """
    render_config = format_config(
        video_format,
        config,
        caption=caption,
        subcaption=subcaption,
    )
    start, end = _window(piece, start_sec, end_sec, max_duration_sec)
    return render_sequence(
        [Clip(piece, start, end)],
        out_path,
        render_config,
        audio_synthesizer=audio_synthesizer,
        faststart=faststart,
        progress=progress,
    )


def export_comparison(
    items: Sequence[ComparisonItem],
    out_path: Path | str,
    video_format: str | VideoFormat = DEFAULT_FORMAT,
    *,
    clip_sec: float = 15.0,
    intro: Card | None = None,
    outro: Card | None = None,
    config: RenderConfig | None = None,
    audio_synthesizer: AudioSynthesizer = default_audio_synthesizer,
    faststart: bool = True,
    progress: bool = False,
) -> Path:
    """Export ``items`` back to back, each ``clip_sec`` long and labelled.

    Each clip shows its ``label``/``sublabel`` in the caption panel and fades
    in and out; optional ``intro``/``outro`` cards frame the sequence.
    """
    if not items:
        raise ValueError("export_comparison needs at least one item")
    render_config = format_config(video_format, config)
    sequence: list[Clip | Card] = [intro] if intro else []
    for item in items:
        start, end = _window(item.piece, item.start_sec, None, clip_sec)
        sequence.append(
            Clip(item.piece, start, end, caption=item.label, subcaption=item.sublabel)
        )
    if outro:
        sequence.append(outro)
    return render_sequence(
        sequence,
        out_path,
        render_config,
        audio_synthesizer=audio_synthesizer,
        faststart=faststart,
        progress=progress,
    )

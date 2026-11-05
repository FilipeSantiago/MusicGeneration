"""Top-level entry point: canonical piece -> piano-roll mp4."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from symusic import Score

from data_processing.music_representations.helpers.canonical_frames import (
    symusic_to_canonical_frames,
)
from data_processing.music_representations.helpers.canonical_loader import (
    CanonicalPiece,
)

from .compose import (
    AudioSynthesizer,
    Clip,
    Frames,
    as_frames,
    default_audio_synthesizer,
    render_sequence,
)
from .config import RenderConfig

__all__ = [
    "AudioSynthesizer",
    "Frames",
    "as_frames",
    "default_audio_synthesizer",
    "load_midi_frames",
    "render_piano_roll_video",
]


def load_midi_frames(
    path: Path | str, *, title: str | None = None
) -> dict[str, pl.DataFrame]:
    """Parse a MIDI file into canonical frames (piece id and title from the file name)."""
    path = Path(path)
    score = Score(str(path))
    return symusic_to_canonical_frames(
        score, piece_id=path.stem, metadata={"title": title or path.stem}
    )


def render_piano_roll_video(
    piece: CanonicalPiece | Frames,
    out_path: Path | str,
    config: RenderConfig | None = None,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    audio_synthesizer: AudioSynthesizer = default_audio_synthesizer,
    progress: bool = False,
    faststart: bool = False,
) -> Path:
    """Render ``piece`` (optionally only ``[start_sec, end_sec)``) as an mp4.

    Audio comes from ``audio_synthesizer`` (FluidSynth by default) when
    ``config.audio`` is true; raises ``AudioUnavailableError`` if it can't run.
    """
    return render_sequence(
        [Clip(piece, start_sec, end_sec)],
        out_path,
        config,
        audio_synthesizer=audio_synthesizer,
        progress=progress,
        faststart=faststart,
    )

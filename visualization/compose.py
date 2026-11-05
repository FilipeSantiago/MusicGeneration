"""Compose several clips and text cards into one video with one soundtrack."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from data_processing.music_representations.helpers.adapters import (
    piece_frames_from_canonical_piece,
)
from data_processing.music_representations.helpers.canonical_loader import (
    CanonicalPiece,
)

from .audio import cut_samples, decode_wav, encode_wav, synthesize_wav
from .config import RenderConfig
from .encoder import FFmpegError, FFmpegWriter, remux_faststart
from .renderer import PianoRollRenderer, TextLine, draw_card, pygame
from .timeline import Timeline

Frames = Mapping[str, pl.DataFrame]
AudioSynthesizer = Callable[[Frames, RenderConfig], bytes]


def default_audio_synthesizer(frames: Frames, config: RenderConfig) -> bytes:
    return synthesize_wav(
        frames,
        soundfont=config.resolve_soundfont(),
        sample_rate=config.sample_rate,
        fluidsynth_bin=config.fluidsynth_bin,
    )


def as_frames(piece: CanonicalPiece | Frames) -> dict[str, pl.DataFrame]:
    if isinstance(piece, CanonicalPiece):
        return piece_frames_from_canonical_piece(piece)
    return dict(piece)


@dataclass(frozen=True)
class Clip:
    """``[start_sec, end_sec)`` of a piece; ``end_sec=None`` runs to the end (+tail).

    ``caption``/``subcaption`` override the config's for this clip only.
    """

    piece: CanonicalPiece | Frames
    start_sec: float = 0.0
    end_sec: float | None = None
    caption: str | None = None
    subcaption: str | None = None


@dataclass(frozen=True)
class Card:
    """A silent full-frame text card."""

    lines: tuple[TextLine, ...]
    duration_sec: float = 3.0


@dataclass
class _Segment:
    n_frames: int
    draw: Callable[[pygame.Surface, int], None]
    audio: np.ndarray | None  # int16 (samples, channels), None = silence


def _frame_count(seconds: float, fps: int) -> int:
    return max(math.ceil(seconds * fps - 1e-9), 1)


def _fade_factor(index: int, n_frames: int, fade_frames: int) -> float:
    if fade_frames <= 0:
        return 1.0
    return min(1.0, (index + 1) / fade_frames, (n_frames - index) / fade_frames)


def render_sequence(
    items: Sequence[Clip | Card],
    out_path: Path | str,
    config: RenderConfig | None = None,
    *,
    audio_synthesizer: AudioSynthesizer = default_audio_synthesizer,
    progress: bool = False,
    faststart: bool = False,
) -> Path:
    """Render clips and cards back to back into one mp4 at ``out_path``.

    Each clip fades in/out over ``config.fade_sec`` (video and audio together);
    cards are silent. With ``faststart`` the result is remuxed to a regular mp4
    (falling back, with a warning, to the streamed file if ffmpeg can't write
    ``out_path`` -- e.g. a sandboxed ffmpeg and a path outside ``$HOME``).
    """
    if not items:
        raise ValueError("render_sequence needs at least one clip or card")
    config = config or RenderConfig()
    fps = config.fps
    rate = config.sample_rate
    segments: list[_Segment] = []
    channels = 2

    for item in items:
        if isinstance(item, Card):
            n = _frame_count(item.duration_sec, fps)
            card = pygame.Surface((config.width, config.height))
            draw_card(card, config, item.lines)

            def show_card(s: pygame.Surface, _i: int, c: pygame.Surface = card) -> None:
                s.blit(c, (0, 0))

            segments.append(_Segment(n, show_card, None))
            continue

        frames = as_frames(item.piece)
        timeline = Timeline.from_frames(frames)
        clip_config = replace(
            config,
            caption=item.caption if item.caption is not None else config.caption,
            subcaption=(
                item.subcaption if item.subcaption is not None else config.subcaption
            ),
        )
        renderer = PianoRollRenderer(timeline, clip_config)
        end = item.end_sec
        if end is None:
            end = timeline.duration_sec + config.tail_sec
        if end <= item.start_sec:
            raise ValueError(f"clip end ({end}) must be after start ({item.start_sec})")
        n = _frame_count(end - item.start_sec, fps)

        audio = None
        if config.audio:
            samples, rate = decode_wav(audio_synthesizer(frames, clip_config))
            channels = samples.shape[1]
            audio = cut_samples(
                samples,
                rate,
                start_sec=item.start_sec,
                n_samples=round(n * rate / fps),
                fade_sec=config.fade_sec,
            )

        def draw(s: pygame.Surface, i: int, r=renderer, t0=item.start_sec) -> None:
            r.draw(s, t0 + i / fps)

        segments.append(_Segment(n, draw, audio))

    soundtrack = None
    if config.audio:
        parts = []
        for segment in segments:
            n_samples = round(segment.n_frames * rate / fps)
            if segment.audio is None:
                parts.append(np.zeros((n_samples, channels), dtype=np.int16))
            elif segment.audio.shape[1] != channels:
                raise ValueError("all clips must synthesize the same channel count")
            else:
                parts.append(segment.audio)
        soundtrack = encode_wav(np.concatenate(parts), rate)

    out_path = Path(out_path)
    stream_path = (
        out_path.with_name(out_path.stem + ".partial.mp4") if faststart else out_path
    )
    total = sum(segment.n_frames for segment in segments)
    fade_frames = round(config.fade_sec * fps)
    surface = pygame.Surface((config.width, config.height))
    with (
        FFmpegWriter(
            stream_path,
            width=config.width,
            height=config.height,
            fps=fps,
            audio_wav=soundtrack,
            duration_sec=total / fps,
            ffmpeg_bin=config.ffmpeg_bin,
            crf=config.crf,
        ) as writer,
        tqdm(total=total, desc="frames", disable=not progress, unit="f") as bar,
    ):
        for segment in segments:
            for index in range(segment.n_frames):
                segment.draw(surface, index)
                level = _fade_factor(index, segment.n_frames, fade_frames)
                if level < 1.0:
                    v = round(255 * level)
                    surface.fill((v, v, v), special_flags=pygame.BLEND_RGB_MULT)
                writer.write(pygame.image.tobytes(surface, "RGB"))
                bar.update()

    if faststart:
        try:
            remux_faststart(stream_path, out_path, ffmpeg_bin=config.ffmpeg_bin)
        except FFmpegError as exc:
            # Keep the streamed (fragmented) mp4: it plays everywhere modern,
            # only some upload pipelines prefer the regular layout.
            warnings.warn(f"{exc}; keeping the fragmented mp4", stacklevel=2)
            stream_path.replace(out_path)
        finally:
            stream_path.unlink(missing_ok=True)
    return out_path

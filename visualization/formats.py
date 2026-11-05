"""Layout presets (aspect ratio + matching scroll speed) for exported videos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .config import RenderConfig


@dataclass(frozen=True, slots=True)
class VideoFormat:
    """Frame size plus the layout values that should change with it.

    ``lookahead_sec`` is chosen per format so notes scroll at a similar speed
    relative to the frame height (about a fifth of the height per second).
    ``fps=None`` keeps the base config's frame rate.

    Post formats run at 60 fps: at 30 fps a fast scroll moves bars ~13 px per
    frame, and on a phone the eye tracking them sees faint "phantom" copies at
    each step position (stroboscopic effect). 60 fps plus a slightly slower
    scroll brings the step down to ~5 px.
    """

    name: str
    width: int
    height: int
    lookahead_sec: float
    keyboard_height_ratio: float
    fps: int | None = None


FORMATS: dict[str, VideoFormat] = {
    # LinkedIn / Instagram feed: takes the most room on a phone screen.
    "portrait": VideoFormat("portrait", 1080, 1350, 3.6, 0.12, fps=60),
    "square": VideoFormat("square", 1080, 1080, 3.2, 0.13, fps=60),
    # Stories / Reels / Shorts.
    "story": VideoFormat("story", 1080, 1920, 4.8, 0.10, fps=60),
    "landscape": VideoFormat("landscape", 1920, 1080, 3.2, 0.16, fps=60),
}
DEFAULT_FORMAT = "portrait"


def get_format(video_format: str | VideoFormat) -> VideoFormat:
    if isinstance(video_format, VideoFormat):
        return video_format
    try:
        return FORMATS[video_format]
    except KeyError:
        names = ", ".join(sorted(FORMATS))
        raise ValueError(
            f"unknown format {video_format!r}; expected one of {names}"
        ) from None


def format_config(
    video_format: str | VideoFormat = DEFAULT_FORMAT,
    base: RenderConfig | None = None,
    **overrides: Any,
) -> RenderConfig:
    """A :class:`RenderConfig` laid out for ``video_format``.

    Post-oriented defaults: notes coloured by hand, no clock, short fades.
    ``base`` supplies any other settings; ``overrides`` win over everything.
    """
    fmt = get_format(video_format)
    base = base or RenderConfig()
    config = replace(
        base,
        fps=fmt.fps or base.fps,
        width=fmt.width,
        height=fmt.height,
        lookahead_sec=fmt.lookahead_sec,
        keyboard_height_ratio=fmt.keyboard_height_ratio,
        color_mode="hand",
        show_clock=False,
        fade_sec=0.3,
    )
    return replace(config, **overrides)

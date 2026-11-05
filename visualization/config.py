"""Rendering options for piano-roll videos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .colors import DEFAULT_HAND_COLORS, DEFAULT_TRACK_PALETTE, RGB

SOUNDFONT_ENV_VAR = "MUSICGEN_SOUNDFONT"
COLOR_MODES = ("track", "hand")
DEFAULT_SOUNDFONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
    Path("/usr/share/soundfonts/default.sf2"),
)


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Everything that controls how a canonical piece becomes a video.

    ``pitch_low``/``pitch_high`` default to the 88 keys of a piano (A0-C8);
    notes outside that range are not drawn. ``lookahead_sec`` is how many
    seconds of upcoming notes are visible above the keyboard.
    """

    width: int = 1280
    height: int = 720
    fps: int = 30
    lookahead_sec: float = 3.0
    pitch_low: int = 21
    pitch_high: int = 108
    keyboard_height_ratio: float = 0.16
    background: RGB = (18, 18, 24)
    track_palette: tuple[RGB, ...] = DEFAULT_TRACK_PALETTE
    # "track": one hue per track; "hand": left/right split at ``hand_split``
    # (solo piano is a single track, so this is what gives it two colours).
    color_mode: str = "track"
    hand_split: int = 60
    hand_colors: tuple[RGB, RGB] = DEFAULT_HAND_COLORS
    show_sustain: bool = True
    show_overlay: bool = True
    particles: bool = True
    particle_density: float = 1.0
    title: str | None = None
    show_clock: bool = True
    # A caption replaces the title bar with a large centred text panel.
    caption: str | None = None
    subcaption: str | None = None
    fade_sec: float = 0.0
    tail_sec: float = 1.0
    audio: bool = True
    soundfont: Path | None = None
    sample_rate: int = 44100
    fluidsynth_bin: str = "fluidsynth"
    ffmpeg_bin: str = "ffmpeg"
    crf: int = 20

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 2 or self.height % 2:
            # libx264 with yuv420p needs even dimensions.
            raise ValueError("width and height must be even")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.lookahead_sec <= 0:
            raise ValueError("lookahead_sec must be positive")
        if not 0 <= self.pitch_low < self.pitch_high <= 127:
            raise ValueError("expected 0 <= pitch_low < pitch_high <= 127")
        if not 0.05 <= self.keyboard_height_ratio <= 0.5:
            raise ValueError("keyboard_height_ratio must be within [0.05, 0.5]")
        if not self.track_palette:
            raise ValueError("track_palette must not be empty")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}")
        if self.fade_sec < 0:
            raise ValueError("fade_sec must be >= 0")
        if self.particle_density < 0:
            raise ValueError("particle_density must be >= 0")
        if self.tail_sec < 0:
            raise ValueError("tail_sec must be >= 0")

    def resolve_soundfont(self) -> Path | None:
        """Explicit ``soundfont``, then ``$MUSICGEN_SOUNDFONT``, then system defaults."""
        if self.soundfont is not None:
            return Path(self.soundfont)
        env = os.environ.get(SOUNDFONT_ENV_VAR)
        if env:
            return Path(env)
        for candidate in DEFAULT_SOUNDFONT_CANDIDATES:
            if candidate.is_file():
                return candidate
        return None

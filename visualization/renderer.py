"""Draws one video frame of the piano roll onto a pygame surface."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

# Headless: never open a window or an audio device.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from .colors import (
    RGB,
    lighten,
    scale,
    track_color,
    velocity_brightness,
    velocity_shade,
)
from .config import RenderConfig
from .keyboard import KeyboardLayout
from .particles import ParticleBatch, ParticleField
from .timeline import Timeline

OVERLAY_HEIGHT = 44
CAPTION_PANEL_RATIO = 0.15
SUSTAIN_STRIP_HEIGHT = 6
BLACK_KEY_HEIGHT_RATIO = 0.62
GLOW_HEIGHT = 28
NOTE_GAP = 2  # px between consecutive notes on one key
BLACK_NOTE_SHADE = 0.82  # black-key notes a touch darker, like the keys
SPRITE_CACHE_LIMIT = 8192
COLOR_STEP = 8  # particle colours are quantized so sprites can be cached

WHITE_KEY: RGB = (236, 236, 240)
BLACK_KEY: RGB = (28, 28, 34)
KEY_BORDER: RGB = (70, 70, 80)
TEXT: RGB = (225, 225, 232)
TEXT_DIM: RGB = (130, 130, 145)
SUSTAIN_ON: RGB = (255, 196, 61)


@dataclass(frozen=True, slots=True)
class TextLine:
    """One line (wrapped if needed) of a caption panel or title card.

    ``size`` is the font height as a fraction of the frame height.
    """

    text: str
    size: float = 0.04
    color: RGB = TEXT


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and font.size(candidate)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    """``text`` shortened with an ellipsis so it renders within ``max_width``."""
    if font.size(text)[0] <= max_width:
        return text
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def draw_text_block(
    surface: pygame.Surface,
    rect: pygame.Rect,
    lines: Sequence[TextLine],
    *,
    frame_height: int,
    gap: float = 0.35,
) -> None:
    """Draw ``lines`` centred (horizontally and vertically) inside ``rect``."""
    rendered: list[pygame.Surface | int] = []
    for index, line in enumerate(lines):
        font = pygame.font.Font(None, max(round(frame_height * line.size), 8))
        if index:
            rendered.append(round(font.get_height() * gap))
        for text in wrap_text(font, line.text, int(rect.width * 0.9)):
            rendered.append(font.render(text, True, line.color))
    total = sum(r if isinstance(r, int) else r.get_height() for r in rendered)
    y = rect.top + (rect.height - total) // 2
    for item in rendered:
        if isinstance(item, int):
            y += item
            continue
        surface.blit(item, item.get_rect(midtop=(rect.centerx, y)))
        y += item.get_height()


def draw_card(
    surface: pygame.Surface, config: RenderConfig, lines: Sequence[TextLine]
) -> None:
    """A full-frame text card (intro, reveal, credits)."""
    surface.fill(config.background)
    draw_text_block(
        surface, surface.get_rect(), lines, frame_height=config.height, gap=0.6
    )


def format_clock(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


class PianoRollRenderer:
    """Falling notes above an 88-key keyboard, with pedal strip and overlay."""

    def __init__(self, timeline: Timeline, config: RenderConfig) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self.timeline = timeline
        self.config = config
        width, height = config.width, config.height
        # Title-bar text and spacing were designed for 720p; scale with the frame.
        self.ui_scale = min(width, height) / 720

        self.caption_mode = bool(config.caption or config.subcaption)
        if self.caption_mode:
            self.overlay_height = round(height * CAPTION_PANEL_RATIO)
        else:
            self.overlay_height = (
                round(OVERLAY_HEIGHT * self.ui_scale) if config.show_overlay else 0
            )
        self.keyboard_height = round(height * config.keyboard_height_ratio)
        self.keyboard_top = height - self.keyboard_height
        strip = SUSTAIN_STRIP_HEIGHT if config.show_sustain else 0
        self.roll_top = self.overlay_height
        self.roll_bottom = self.keyboard_top - strip
        self.pixels_per_second = (
            self.roll_bottom - self.roll_top
        ) / config.lookahead_sec
        self.layout = KeyboardLayout.build(config.pitch_low, config.pitch_high, width)

        self._font = pygame.font.Font(None, round(26 * self.ui_scale))
        self._small_font = pygame.font.Font(None, round(22 * self.ui_scale))
        self._background = self._build_background()
        self._glow_cache: dict[tuple[int, RGB], pygame.Surface] = {}
        self._sprite_cache: dict[tuple[int, RGB], pygame.Surface] = {}
        self.particles = (
            ParticleField(
                timeline,
                self.layout,
                origin_y=self.roll_bottom,
                frame_height=height,
                density=config.particle_density,
            )
            if config.particles
            else None
        )
        self._title_surface = self._font.render(
            fit_text(self._font, self._title_text(), round(width * 0.62)), True, TEXT
        )
        self._caption_panel = self._build_caption_panel() if self.caption_mode else None

    def new_surface(self) -> pygame.Surface:
        return pygame.Surface((self.config.width, self.config.height))

    def draw(self, surface: pygame.Surface, t: float) -> None:
        surface.blit(self._background, (0, 0))
        self._draw_notes(surface, t)
        lit = self._lit_keys(t)
        self._draw_glow(surface, lit)
        self._draw_keyboard(surface, lit)
        if self.config.show_sustain:
            self._draw_sustain(surface, t)
        if self.particles is not None:
            self._draw_particles(surface, t)
        if self._caption_panel is not None:
            surface.blit(self._caption_panel, (0, 0))
        elif self.config.show_overlay:
            self._draw_overlay(surface, t)

    # -- colours -----------------------------------------------------------

    def base_color(self, track_id: int, pitch: int) -> RGB:
        config = self.config
        if config.color_mode == "hand":
            return config.hand_colors[0 if pitch < config.hand_split else 1]
        return track_color(track_id, config.track_palette)

    def note_color(self, track_id: int, pitch: int, velocity: int) -> RGB:
        return velocity_shade(self.base_color(track_id, pitch), velocity)

    def _base_colors(self, notes: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`base_color` for timeline note indices -> (n, 3)."""
        config = self.config
        if config.color_mode == "hand":
            hands = np.array(config.hand_colors, dtype=np.float64)
            right = self.timeline.pitch[notes] >= config.hand_split
            return hands[right.astype(np.int64)]
        palette = np.array(config.track_palette, dtype=np.float64)
        return palette[self.timeline.track[notes] % len(palette)]

    # -- static layers -----------------------------------------------------

    def _build_background(self) -> pygame.Surface:
        surface = self.new_surface()
        surface.fill(self.config.background)
        guide = lighten(self.config.background, 0.06)
        for key in self.layout.white_keys:
            if key.pitch % 12 in (0, 5):  # C and F boundaries
                x = round(key.x)
                width = 2 if key.pitch % 12 == 0 else 1
                pygame.draw.line(
                    surface, guide, (x, self.roll_top), (x, self.roll_bottom), width
                )
        return surface

    def _build_caption_panel(self) -> pygame.Surface:
        width, height = self.config.width, self.overlay_height
        panel = pygame.Surface((width, height))
        panel.fill(scale(self.config.background, 0.6))
        lines = []
        if self.config.caption:
            lines.append(TextLine(self.config.caption, 0.05, TEXT))
        if self.config.subcaption:
            lines.append(TextLine(self.config.subcaption, 0.027, TEXT_DIM))
        draw_text_block(panel, panel.get_rect(), lines, frame_height=self.config.height)
        pygame.draw.line(
            panel, scale(TEXT_DIM, 0.5), (0, height - 1), (width, height - 1)
        )
        return panel

    def _title_text(self) -> str:
        if self.config.title:
            return self.config.title
        timeline = self.timeline
        title = timeline.title or timeline.piece_id or "Untitled"
        return f"{title} — {timeline.composer}" if timeline.composer else title

    # -- dynamic layers ----------------------------------------------------

    def _draw_notes(self, surface: pygame.Surface, t: float) -> None:
        timeline = self.timeline
        indices = timeline.visible(t, self.config.lookahead_sec)
        if indices.size == 0:
            return
        # White-key notes first so black-key notes stay visible on top.
        ordered = sorted(
            (int(i) for i in indices if self.layout.contains(int(timeline.pitch[i]))),
            key=lambda i: self.layout.key(int(timeline.pitch[i])).is_black,
        )
        pps = self.pixels_per_second
        background = self.config.background
        for i in ordered:
            key = self.layout.key(int(timeline.pitch[i]))
            bottom = min(
                self.roll_bottom - (timeline.onset[i] - t) * pps, self.roll_bottom
            )
            top = self.roll_bottom - (timeline.end[i] - t) * pps + NOTE_GAP
            top = max(top, self.roll_top)
            if bottom - top < 2.0:
                top = bottom - 2.0
            color = self.note_color(
                int(timeline.track[i]),
                int(timeline.pitch[i]),
                int(timeline.velocity[i]),
            )
            if key.is_black:
                color = scale(color, BLACK_NOTE_SHADE)
            if timeline.onset[i] <= t:
                color = lighten(color, 0.25)
            rect = pygame.Rect(
                round(key.x) + 1,
                round(top),
                max(round(key.width) - 2, 1),
                max(round(bottom - top), 2),
            )
            if key.is_black:
                # A background-coloured rim keeps a black-key note readable as
                # its own bar where it overlaps a neighbouring white-key note.
                pygame.draw.rect(
                    surface, background, rect.inflate(4, 0), border_radius=4
                )
            pygame.draw.rect(surface, color, rect, border_radius=3)
            pygame.draw.rect(surface, scale(color, 0.6), rect, width=1, border_radius=3)

    def _lit_keys(self, t: float) -> dict[int, RGB]:
        timeline = self.timeline
        lit: dict[int, tuple[int, RGB]] = {}
        for i in timeline.active(t):
            pitch = int(timeline.pitch[i])
            velocity = int(timeline.velocity[i])
            if not self.layout.contains(pitch):
                continue
            # When tracks overlap on a key, the loudest note sets the colour.
            if pitch not in lit or velocity > lit[pitch][0]:
                lit[pitch] = (
                    velocity,
                    self.note_color(int(timeline.track[i]), pitch, velocity),
                )
        return {pitch: color for pitch, (_, color) in lit.items()}

    def _glow_surface(self, width: int, color: RGB) -> pygame.Surface:
        cache_key = (width, color)
        glow = self._glow_cache.get(cache_key)
        if glow is None:
            glow = pygame.Surface((width, GLOW_HEIGHT), pygame.SRCALPHA)
            for row in range(GLOW_HEIGHT):
                alpha = round(150 * (row / GLOW_HEIGHT) ** 2)
                pygame.draw.line(glow, (*color, alpha), (0, row), (width - 1, row))
            self._glow_cache[cache_key] = glow
        return glow

    def _draw_glow(self, surface: pygame.Surface, lit: dict[int, RGB]) -> None:
        for pitch, color in lit.items():
            key = self.layout.key(pitch)
            width = max(round(key.width), 1)
            surface.blit(
                self._glow_surface(width, color),
                (round(key.x), self.roll_bottom - GLOW_HEIGHT),
            )

    def _draw_keyboard(self, surface: pygame.Surface, lit: dict[int, RGB]) -> None:
        top, height = self.keyboard_top, self.keyboard_height
        for key in self.layout.white_keys:
            rect = pygame.Rect(
                round(key.x),
                top,
                max(round(key.x + key.width) - round(key.x), 1),
                height,
            )
            pygame.draw.rect(surface, lit.get(key.pitch, WHITE_KEY), rect)
            pygame.draw.rect(surface, KEY_BORDER, rect, width=1)
        black_height = round(height * BLACK_KEY_HEIGHT_RATIO)
        for key in self.layout.black_keys:
            rect = pygame.Rect(
                round(key.x), top, max(round(key.width), 1), black_height
            )
            color = lit.get(key.pitch)
            pygame.draw.rect(
                surface,
                color if color else BLACK_KEY,
                rect,
                border_bottom_left_radius=2,
                border_bottom_right_radius=2,
            )
            if color:
                pygame.draw.rect(surface, BLACK_KEY, rect, width=1)
        pygame.draw.line(
            surface, scale(SUSTAIN_ON, 0.5), (0, top), (self.config.width, top), 2
        )

    def _draw_sustain(self, surface: pygame.Surface, t: float) -> None:
        on = self.timeline.sustain_at(t)
        color = SUSTAIN_ON if on else lighten(self.config.background, 0.08)
        rect = pygame.Rect(
            0, self.roll_bottom, self.config.width, self.keyboard_top - self.roll_bottom
        )
        pygame.draw.rect(surface, color, rect)

    def _draw_particles(self, surface: pygame.Surface, t: float) -> None:
        assert self.particles is not None
        for batch in self.particles.at(t):
            if len(batch):
                self._blit_batch(surface, batch)

    def _blit_batch(self, surface: pygame.Surface, batch: ParticleBatch) -> None:
        timeline = self.timeline
        base = self._base_colors(batch.note)
        loudness = timeline.velocity[batch.note] / 127.0
        shade = velocity_brightness(0) + (1.0 - velocity_brightness(0)) * loudness
        color = base * shade[:, None]
        color += (255.0 - color) * batch.whiteness[:, None]
        color *= batch.intensity[:, None]
        quantized = (np.round(color / COLOR_STEP) * COLOR_STEP).clip(0, 255)
        quantized = quantized.astype(np.int64)
        radii = np.maximum(np.round(batch.radius), 1).astype(np.int64)
        add = pygame.BLEND_RGB_ADD
        for x, y, radius, (r, g, b) in zip(
            batch.x.tolist(),
            batch.y.tolist(),
            radii.tolist(),
            quantized.tolist(),
            strict=True,
        ):
            if r == g == b == 0:
                continue
            sprite = self._sprite(radius, (r, g, b))
            half = sprite.get_width() // 2
            surface.blit(sprite, (round(x) - half, round(y) - half), special_flags=add)

    def _sprite(self, radius: int, color: RGB) -> pygame.Surface:
        """A soft round dot (bright core, halo twice the radius) for additive blits."""
        cache_key = (radius, color)
        sprite = self._sprite_cache.get(cache_key)
        if sprite is None:
            if len(self._sprite_cache) >= SPRITE_CACHE_LIMIT:
                self._sprite_cache.clear()
            halo = radius * 2
            grid = np.arange(-halo, halo + 1, dtype=np.float64)
            distance = np.hypot(grid[:, None], grid[None, :])
            core = np.clip(1.0 - distance / max(radius, 1), 0.0, 1.0)
            glow = np.clip(1.0 - distance / halo, 0.0, 1.0) ** 2 * 0.45
            falloff = np.maximum(core, glow)
            pixels = falloff[:, :, None] * np.array(color, dtype=np.float64)
            sprite = pygame.surfarray.make_surface(pixels.astype(np.uint8))
            self._sprite_cache[cache_key] = sprite
        return sprite

    def _draw_overlay(self, surface: pygame.Surface, t: float) -> None:
        width, height = self.config.width, self.overlay_height
        pygame.draw.rect(
            surface, scale(self.config.background, 0.6), (0, 0, width, height)
        )

        pad = round(14 * self.ui_scale)
        title = self._title_surface
        surface.blit(title, (pad, (height - title.get_height()) // 2))

        duration = self.timeline.duration_sec
        x = width - pad
        if self.config.show_clock:
            bar, beat = self.timeline.bar_beat_at(t)
            clock = (
                f"Bar {bar} · Beat {beat}    "
                f"{format_clock(t)} / {format_clock(duration)}"
            )
            clock_surface = self._small_font.render(clock, True, TEXT)
            x -= clock_surface.get_width()
            surface.blit(clock_surface, (x, (height - clock_surface.get_height()) // 2))

        if self.config.show_sustain:
            on = self.timeline.sustain_at(t)
            label = self._small_font.render(
                "PED", True, (20, 20, 20) if on else TEXT_DIM
            )
            pill = pygame.Rect(
                0, 0, label.get_width() + pad, label.get_height() + pad // 2
            )
            pill.midright = (x - pad, height // 2)
            if on:
                pygame.draw.rect(
                    surface, SUSTAIN_ON, pill, border_radius=pill.height // 2
                )
            else:
                pygame.draw.rect(
                    surface, TEXT_DIM, pill, width=1, border_radius=pill.height // 2
                )
            surface.blit(label, label.get_rect(center=pill.center))

        if duration > 0:
            progress = min(max(t / duration, 0.0), 1.0)
            thick = max(round(2 * self.ui_scale), 2)
            bar_y = height - thick
            pygame.draw.rect(surface, scale(TEXT_DIM, 0.5), (0, bar_y, width, thick))
            pygame.draw.rect(surface, TEXT, (0, bar_y, round(width * progress), thick))

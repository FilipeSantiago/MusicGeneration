"""Colour helpers: one hue per track, brightness from velocity."""

from __future__ import annotations

from collections.abc import Sequence

RGB = tuple[int, int, int]

DEFAULT_TRACK_PALETTE: tuple[RGB, ...] = (
    (76, 201, 240),  # cyan
    (247, 127, 0),  # orange
    (128, 237, 153),  # green
    (247, 37, 133),  # pink
    (255, 209, 102),  # yellow
    (181, 131, 255),  # violet
    (6, 214, 160),  # teal
    (239, 71, 111),  # red
)

# (left hand, right hand)
DEFAULT_HAND_COLORS: tuple[RGB, RGB] = ((181, 131, 255), (76, 201, 240))

# Quiet notes are dimmed only down to this: lower reads as see-through on the
# dark background.
MIN_VELOCITY_BRIGHTNESS = 0.65


def track_color(track_id: int, palette: Sequence[RGB] = DEFAULT_TRACK_PALETTE) -> RGB:
    return palette[int(track_id) % len(palette)]


def velocity_shade(color: RGB, velocity: int) -> RGB:
    """Scale ``color`` from 100% at velocity 127 down to the brightness floor at 0."""
    return scale(color, velocity_brightness(velocity))


def velocity_brightness(velocity: float) -> float:
    velocity = min(max(float(velocity), 0.0), 127.0)
    return MIN_VELOCITY_BRIGHTNESS + (1.0 - MIN_VELOCITY_BRIGHTNESS) * velocity / 127


def scale(color: RGB, factor: float) -> RGB:
    r, g, b = (min(255, max(0, round(c * factor))) for c in color)
    return (r, g, b)


def lighten(color: RGB, amount: float) -> RGB:
    """Blend ``color`` towards white by ``amount`` (0..1)."""
    r, g, b = (round(c + (255 - c) * amount) for c in color)
    return (r, g, b)

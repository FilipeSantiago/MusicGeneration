"""Stateless particle effects for key presses.

Every particle is a closed-form function of its note's onset and a hash of
(note index, particle index), so a frame at time ``t`` can be drawn on its own
-- no simulation state, identical output on every run, any frame testable.

Three effects:
- burst: sparks thrown upward on the press, falling back under gravity;
- embers: a slow trickle rising for as long as the key is held;
- flash: a short glow at the top of the key on the press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .keyboard import KeyboardLayout
from .timeline import Timeline

# Tuned for a 720 px tall frame; distances scale with the frame height.
REFERENCE_HEIGHT = 720.0

BURST_COUNT = 22
BURST_LIFE = (0.4, 1.0)  # seconds (min, max)
BURST_SPEED = (130.0, 430.0)  # px/s
BURST_SPREAD = 1.75  # radians around straight up
BURST_GRAVITY = 600.0  # px/s^2, pulls down
BURST_BRIGHTNESS = 0.8

EMBER_RATE = 36.0  # particles per second while a key is held
EMBER_LIFE = (0.4, 0.9)
EMBER_SPEED = (60.0, 170.0)
EMBER_SPREAD = 0.9
EMBER_GRAVITY = -40.0  # embers keep rising slightly faster
EMBER_BRIGHTNESS = 0.8

FLASH_LIFE = 0.2
FLASH_BRIGHTNESS = 0.6

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _hash01(a: np.ndarray, b: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic uniform [0, 1) values from integer arrays (splitmix64)."""
    with np.errstate(over="ignore"):
        x = (
            a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
            ^ b.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
            ^ np.uint64((salt * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
        ) & _MASK64
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


@dataclass(frozen=True, slots=True)
class ParticleBatch:
    """Particles alive at one instant; ``note`` indexes the timeline's notes."""

    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    intensity: np.ndarray  # 0..1 brightness multiplier
    whiteness: np.ndarray  # 0..1 blend of the note colour towards white
    note: np.ndarray

    @classmethod
    def empty(cls) -> ParticleBatch:
        f, i = np.zeros(0), np.zeros(0, dtype=np.int64)
        return cls(f, f, f, f, f, i)

    def __len__(self) -> int:
        return int(self.x.size)


class ParticleField:
    """Computes the burst, ember and flash particles alive at a given time."""

    def __init__(
        self,
        timeline: Timeline,
        layout: KeyboardLayout,
        *,
        origin_y: float,
        frame_height: int,
        density: float = 1.0,
    ) -> None:
        self.timeline = timeline
        self.origin_y = float(origin_y)
        self.scale = frame_height / REFERENCE_HEIGHT
        self.density = max(density, 0.0)

        key_x = np.full(128, np.nan)
        key_width = np.full(128, np.nan)
        for key in layout.keys:
            key_x[key.pitch] = key.x
            key_width[key.pitch] = key.width
        self._key_x = key_x
        self._key_width = key_width
        self._on_keyboard = ~np.isnan(key_x[timeline.pitch])

    def at(self, t: float) -> list[ParticleBatch]:
        if self.density == 0:
            return []
        return [self.flashes(t), self.embers(t), self.bursts(t)]

    # -- effects ------------------------------------------------------------

    def bursts(self, t: float) -> ParticleBatch:
        notes = self._started_within(t, BURST_LIFE[1])
        if notes.size == 0:
            return ParticleBatch.empty()
        n_max = max(round(BURST_COUNT * self.density), 1)
        note = np.repeat(notes, n_max)
        part = np.tile(np.arange(n_max), notes.size)
        loudness = self.timeline.velocity[note] / 127.0
        keep = part < np.maximum(np.round(n_max * (0.4 + 0.6 * loudness)), 1)
        note, part, loudness = note[keep], part[keep], loudness[keep]

        rand = lambda salt: _hash01(note, part, salt)
        age = t - self.timeline.onset[note]
        life = BURST_LIFE[0] + (BURST_LIFE[1] - BURST_LIFE[0]) * rand(1)
        angle = -math.pi / 2 + (rand(2) - 0.5) * BURST_SPREAD
        speed = (
            (BURST_SPEED[0] + (BURST_SPEED[1] - BURST_SPEED[0]) * rand(3))
            * (0.45 + 0.55 * loudness)
            * self.scale
        )
        x0 = self._key_x[self.timeline.pitch[note]] + self._key_width[
            self.timeline.pitch[note]
        ] * rand(4)
        progress = age / life
        return self._batch(
            note=note,
            x=x0 + np.cos(angle) * speed * age,
            y=self.origin_y
            + np.sin(angle) * speed * age
            + 0.5 * BURST_GRAVITY * self.scale * age**2,
            radius=(1.8 + 2.4 * rand(5)) * (1.0 - 0.5 * progress) * self.scale,
            intensity=BURST_BRIGHTNESS * (1.0 - np.minimum(progress, 1.0)) ** 1.5,
            whiteness=0.1 + 0.3 * rand(6),
            alive=progress < 1.0,
        )

    def embers(self, t: float) -> ParticleBatch:
        timeline = self.timeline
        horizon = timeline.max_note_duration + EMBER_LIFE[1]
        lo = int(np.searchsorted(timeline.onset, t - horizon, side="left"))
        hi = int(np.searchsorted(timeline.onset, t, side="right"))
        candidates = np.arange(lo, hi)
        candidates = candidates[
            self._on_keyboard[lo:hi] & (timeline.end[lo:hi] > t - EMBER_LIFE[1])
        ]
        rate = EMBER_RATE * self.density
        if candidates.size == 0 or rate <= 0:
            return ParticleBatch.empty()

        onset = timeline.onset[candidates]
        # Spawn k happens at onset + k / rate (k >= 1; the press itself is the burst).
        first = np.maximum(np.ceil((t - EMBER_LIFE[1] - onset) * rate), 1).astype(int)
        last = np.floor((np.minimum(t, timeline.end[candidates]) - onset) * rate)
        counts = np.maximum(last.astype(int) - first + 1, 0)
        if counts.sum() == 0:
            return ParticleBatch.empty()
        note = np.repeat(candidates, counts)
        offsets = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        spawn = np.repeat(first, counts) + offsets

        rand = lambda salt: _hash01(note, spawn, salt + 100)
        age = t - (timeline.onset[note] + spawn / rate)
        life = EMBER_LIFE[0] + (EMBER_LIFE[1] - EMBER_LIFE[0]) * rand(1)
        angle = -math.pi / 2 + (rand(2) - 0.5) * EMBER_SPREAD
        speed = (
            EMBER_SPEED[0] + (EMBER_SPEED[1] - EMBER_SPEED[0]) * rand(3)
        ) * self.scale
        pitch = timeline.pitch[note]
        x0 = self._key_x[pitch] + self._key_width[pitch] * (0.2 + 0.6 * rand(4))
        progress = age / life
        loudness = timeline.velocity[note] / 127.0
        return self._batch(
            note=note,
            x=x0 + np.cos(angle) * speed * age,
            y=self.origin_y
            + np.sin(angle) * speed * age
            + 0.5 * EMBER_GRAVITY * self.scale * age**2,
            radius=(1.5 + 2.0 * rand(5)) * self.scale,
            intensity=EMBER_BRIGHTNESS * (0.5 + 0.5 * loudness) * (1.0 - progress) ** 2,
            whiteness=0.1 * rand(6),
            alive=(age >= 0) & (progress < 1.0),
        )

    def flashes(self, t: float) -> ParticleBatch:
        notes = self._started_within(t, FLASH_LIFE)
        if notes.size == 0:
            return ParticleBatch.empty()
        pitch = self.timeline.pitch[notes]
        age = t - self.timeline.onset[notes]
        loudness = self.timeline.velocity[notes] / 127.0
        return self._batch(
            note=notes,
            x=self._key_x[pitch] + self._key_width[pitch] / 2,
            y=np.full(notes.size, self.origin_y),
            radius=self._key_width[pitch] * 1.1,
            intensity=FLASH_BRIGHTNESS
            * (1.0 - age / FLASH_LIFE) ** 2
            * (0.5 + 0.5 * loudness),
            whiteness=np.full(notes.size, 0.3),
            alive=np.ones(notes.size, dtype=bool),
        )

    # -- helpers ------------------------------------------------------------

    def _started_within(self, t: float, window: float) -> np.ndarray:
        """Indices of on-keyboard notes with ``t - window < onset <= t``."""
        lo = int(np.searchsorted(self.timeline.onset, t - window, side="right"))
        hi = int(np.searchsorted(self.timeline.onset, t, side="right"))
        notes = np.arange(lo, hi)
        return notes[self._on_keyboard[lo:hi]]

    @staticmethod
    def _batch(
        *,
        note: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        radius: np.ndarray,
        intensity: np.ndarray,
        whiteness: np.ndarray,
        alive: np.ndarray,
    ) -> ParticleBatch:
        return ParticleBatch(
            x=x[alive],
            y=y[alive],
            radius=radius[alive],
            intensity=np.clip(intensity[alive], 0.0, 1.0),
            whiteness=whiteness[alive],
            note=note[alive],
        )

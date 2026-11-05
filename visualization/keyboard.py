"""Piano keyboard geometry (pure functions, no drawing)."""

from __future__ import annotations

from dataclasses import dataclass

BLACK_PITCH_CLASSES = frozenset({1, 3, 6, 8, 10})
BLACK_KEY_WIDTH_RATIO = 0.6


def is_black(pitch: int) -> bool:
    return pitch % 12 in BLACK_PITCH_CLASSES


@dataclass(frozen=True, slots=True)
class KeyGeometry:
    pitch: int
    x: float
    width: float
    is_black: bool


@dataclass(frozen=True, slots=True)
class KeyboardLayout:
    """Horizontal position of every key in ``[low, high]`` over ``width`` pixels.

    White keys tile the width evenly; each black key is centred on the boundary
    between its neighbouring white keys.
    """

    low: int
    high: int
    width: float
    white_width: float
    keys: tuple[KeyGeometry, ...]

    @classmethod
    def build(cls, low: int, high: int, width: float) -> KeyboardLayout:
        if not 0 <= low <= high <= 127:
            raise ValueError("expected 0 <= low <= high <= 127")
        n_white = sum(1 for pitch in range(low, high + 1) if not is_black(pitch))
        white_width = width / max(n_white, 1)
        black_width = white_width * BLACK_KEY_WIDTH_RATIO

        keys: list[KeyGeometry] = []
        white_index = 0
        for pitch in range(low, high + 1):
            if is_black(pitch):
                x = white_index * white_width - black_width / 2
                x = min(max(x, 0.0), width - black_width)
                keys.append(KeyGeometry(pitch, x, black_width, True))
            else:
                keys.append(
                    KeyGeometry(pitch, white_index * white_width, white_width, False)
                )
                white_index += 1
        return cls(
            low=low, high=high, width=width, white_width=white_width, keys=tuple(keys)
        )

    def contains(self, pitch: int) -> bool:
        return self.low <= pitch <= self.high

    def key(self, pitch: int) -> KeyGeometry:
        if not self.contains(pitch):
            raise KeyError(f"pitch {pitch} outside keyboard [{self.low}, {self.high}]")
        return self.keys[pitch - self.low]

    @property
    def white_keys(self) -> tuple[KeyGeometry, ...]:
        return tuple(key for key in self.keys if not key.is_black)

    @property
    def black_keys(self) -> tuple[KeyGeometry, ...]:
        return tuple(key for key in self.keys if key.is_black)

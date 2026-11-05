"""Thin CLI over :func:`visualization.export_piece` for quick manual renders.

    python -m visualization INPUT -o out.mp4 [--format portrait] [--max-duration 30]

INPUT is a ``.mid``/``.midi`` file, or a canonical dataset directory together
with ``--piece-id``. Comparisons are library-only: see ``export_comparison``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl

from data_processing.music_representations.helpers.canonical_loader import (
    CanonicalPiece,
    load_canonical_piece,
)

from .audio import AudioUnavailableError
from .config import RenderConfig
from .encoder import FFmpegError
from .export import export_piece
from .formats import DEFAULT_FORMAT, FORMATS
from .video import load_midi_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization",
        description="Export a piece as a falling-notes piano-roll video.",
    )
    add = parser.add_argument
    add("input", type=Path, help="MIDI file or canonical dataset directory")
    add("-o", "--output", type=Path, required=True, help="output .mp4 path")
    add("--piece-id", help="piece to render when INPUT is a canonical dataset")
    add("--format", choices=sorted(FORMATS), default=DEFAULT_FORMAT)
    add("--start", type=float, help="clip start in seconds")
    add("--end", type=float, help="clip end in seconds")
    add(
        "--max-duration",
        type=float,
        help="cap the length; without --start, the busiest window is used",
    )
    add("--title", help="override the title bar text")
    add("--caption", help="large caption panel (replaces the title bar)")
    add("--subcaption", help="smaller line under the caption")
    add("--soundfont", type=Path, help="SoundFont (.sf2) for FluidSynth")
    add("--particle-density", type=float, default=1.0)
    add("--no-audio", action="store_true", help="render a silent video")
    add("--no-particles", action="store_true", help="disable key-press particles")
    add("--no-faststart", action="store_true", help="skip the upload-friendly remux")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    piece: CanonicalPiece | dict[str, pl.DataFrame]
    if args.input.is_dir():
        if not args.piece_id:
            print(
                "error: --piece-id is required for a canonical dataset", file=sys.stderr
            )
            return 2
        piece = load_canonical_piece(args.input, args.piece_id)
    elif args.input.suffix.lower() in {".mid", ".midi"}:
        piece = load_midi_frames(args.input, title=args.title)
    else:
        print(f"error: unsupported input {args.input}", file=sys.stderr)
        return 2

    try:
        base = replace(
            RenderConfig(),
            title=args.title,
            audio=not args.no_audio,
            soundfont=args.soundfont,
            particles=not args.no_particles,
            particle_density=args.particle_density,
        )
        out = export_piece(
            piece,
            args.output,
            args.format,
            start_sec=args.start,
            end_sec=args.end,
            max_duration_sec=args.max_duration,
            caption=args.caption,
            subcaption=args.subcaption,
            config=base,
            faststart=not args.no_faststart,
            progress=True,
        )
    except (AudioUnavailableError, FFmpegError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

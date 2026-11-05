"""Render canonical pieces as falling-notes piano-roll videos (mp4).

Library entry points (each writes one mp4 and returns its path):

- :func:`export_piece` -- one piece, optionally cut to a window.
- :func:`export_comparison` -- labelled clips of several pieces, back to back.
- :func:`render_sequence` -- lower level: any mix of :class:`Clip` and :class:`Card`.

Layouts come from :data:`FORMATS` (``portrait`` 4:5 by default, ``square``,
``story``, ``landscape``). Needs ``ffmpeg``; audio also needs FluidSynth and a
SoundFont (``sudo apt install fluidsynth fluid-soundfont-gm``), otherwise pass
``config=RenderConfig(audio=False)``.
"""

from .audio import AudioUnavailableError, synthesize_wav
from .compose import Card, Clip, render_sequence
from .config import RenderConfig
from .encoder import FFmpegError, FFmpegWriter
from .export import ComparisonItem, export_comparison, export_piece
from .formats import FORMATS, VideoFormat, format_config
from .keyboard import KeyboardLayout
from .renderer import PianoRollRenderer, TextLine
from .timeline import Timeline
from .video import load_midi_frames, render_piano_roll_video

__all__ = [
    "FORMATS",
    "AudioUnavailableError",
    "Card",
    "Clip",
    "ComparisonItem",
    "FFmpegError",
    "FFmpegWriter",
    "KeyboardLayout",
    "PianoRollRenderer",
    "RenderConfig",
    "TextLine",
    "Timeline",
    "VideoFormat",
    "export_comparison",
    "export_piece",
    "format_config",
    "load_midi_frames",
    "render_piano_roll_video",
    "render_sequence",
    "synthesize_wav",
]

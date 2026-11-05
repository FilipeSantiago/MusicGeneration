"""Synthesizes a canonical piece to WAV with FluidSynth and a SoundFont."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from data_processing.music_representations.helpers.adapters import (
    canonical_frames_to_midi,
)

DEFAULT_PEAK = 0.89  # about -1 dBFS


class AudioUnavailableError(RuntimeError):
    """FluidSynth or a SoundFont is missing; render with ``audio=False`` instead."""


def synthesize_wav(
    frames: Mapping[str, pl.DataFrame],
    *,
    soundfont: Path | None,
    sample_rate: int = 44100,
    fluidsynth_bin: str = "fluidsynth",
    normalize_peak: float | None = DEFAULT_PEAK,
) -> bytes:
    """Return the WAV bytes for ``frames`` rendered offline by FluidSynth.

    FluidSynth's default gain is quiet, so the result is peak-normalized to
    ``normalize_peak`` (fraction of full scale); pass ``None`` to keep it raw.
    """
    executable = shutil.which(fluidsynth_bin)
    if executable is None:
        raise AudioUnavailableError(
            f"FluidSynth executable {fluidsynth_bin!r} not found; install it with "
            "`sudo apt install fluidsynth fluid-soundfont-gm` or render with audio=False"
        )
    if soundfont is None or not Path(soundfont).is_file():
        raise AudioUnavailableError(
            f"SoundFont not found ({soundfont}); pass RenderConfig.soundfont, set "
            "$MUSICGEN_SOUNDFONT, or install `fluid-soundfont-gm`"
        )

    with tempfile.TemporaryDirectory(prefix="musicgen-audio-") as tmp:
        midi_path = canonical_frames_to_midi(dict(frames), Path(tmp) / "piece.mid")
        wav_path = Path(tmp) / "piece.wav"
        cmd = [
            executable, "-ni", "-q",
            "-F", str(wav_path), "-T", "wav", "-r", str(sample_rate),
            str(soundfont), str(midi_path),
        ]  # fmt: skip
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not wav_path.is_file():
            detail = (result.stderr or result.stdout).decode(errors="replace").strip()
            raise AudioUnavailableError(
                f"FluidSynth failed (exit {result.returncode}): {detail}"
            )
        data = wav_path.read_bytes()
    return normalize_wav(data, normalize_peak) if normalize_peak else data


def normalize_wav(data: bytes, peak: float = DEFAULT_PEAK) -> bytes:
    """Scale 16-bit PCM WAV so its loudest sample is ``peak`` of full scale."""
    with wave.open(io.BytesIO(data)) as reader:
        params = reader.getparams()
        if params.sampwidth != 2:
            return data
        samples = np.frombuffer(reader.readframes(params.nframes), dtype="<i2")
    loudest = int(np.abs(samples.astype(np.int32)).max()) if samples.size else 0
    if loudest == 0:
        return data
    scaled = samples.astype(np.float64) * (peak * 32767 / loudest)
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(np.clip(scaled, -32768, 32767).astype("<i2").tobytes())
    return out.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes -> (int16 samples shaped (frames, channels), rate)."""
    with wave.open(io.BytesIO(data)) as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        channels, rate = reader.getnchannels(), reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels), rate


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(samples.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
    return out.getvalue()


def cut_samples(
    samples: np.ndarray,
    rate: int,
    *,
    start_sec: float,
    n_samples: int,
    fade_sec: float = 0.0,
) -> np.ndarray:
    """``n_samples`` from ``start_sec`` (zero-padded), with linear fade in/out."""
    begin = max(round(start_sec * rate), 0)
    out = np.zeros((n_samples, samples.shape[1]), dtype=np.float64)
    chunk = samples[begin : begin + n_samples]
    out[: len(chunk)] = chunk
    fade = min(round(fade_sec * rate), n_samples // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)[:, None]
        out[:fade] *= ramp
        out[n_samples - fade :] *= ramp[::-1]
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)

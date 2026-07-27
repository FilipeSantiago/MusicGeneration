from __future__ import annotations

import csv
import sys
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.music_representations.config import (
    MusicRepresentationConfig,
    SegmentationConfig,
    SubwordConfig,
)


def _write_midi(
    path: Path, *, note_start: int, duration: int, velocity: int, split_name: str
) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=f"{split_name}_piano", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("key_signature", key="C", time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    track.append(mido.Message("control_change", control=64, value=127, time=0))
    track.append(mido.Message("pitchwheel", pitch=200, time=0))
    track.append(mido.Message("note_on", note=60, velocity=velocity, time=note_start))
    track.append(mido.Message("note_off", note=60, velocity=0, time=duration))
    track.append(mido.Message("control_change", control=66, value=64, time=5))
    track.append(mido.Message("control_change", control=64, value=0, time=7))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


@pytest.fixture()
def maestro_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rows = []
    specs = [
        ("train", "train_piece.mid", 123, 457, 93, "Composer Train", "Train Piece"),
        (
            "validation",
            "validation_piece.mid",
            131,
            211,
            77,
            "Composer Validation",
            "Validation Piece",
        ),
        ("test", "test_piece.mid", 145, 333, 88, "Composer Test", "Test Piece"),
    ]
    for split, filename, start, duration, velocity, composer, title in specs:
        split_dir = input_dir / split
        split_dir.mkdir(exist_ok=True)
        midi_path = split_dir / filename
        _write_midi(
            midi_path,
            note_start=start,
            duration=duration,
            velocity=velocity,
            split_name=split,
        )
        rows.append(
            {
                "midi_filename": f"{split}/{filename}",
                "canonical_composer": composer,
                "title": title,
                "split": split,
            }
        )
    metadata_path = input_dir / "maestro-v3.0.0.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["midi_filename", "canonical_composer", "title", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return input_dir, output_dir


@pytest.fixture()
def base_config(maestro_fixture: tuple[Path, Path]) -> MusicRepresentationConfig:
    input_dir, output_dir = maestro_fixture
    return MusicRepresentationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_csv=input_dir / "maestro-v3.0.0.csv",
        segmentation=SegmentationConfig(
            enabled=True,
            segment_ticks=480,
            hop_ticks=240,
            min_notes=1,
            allow_incomplete_final_segment=True,
        ),
        subword=SubwordConfig(base_representation="remi", vocab_size=64),
    )

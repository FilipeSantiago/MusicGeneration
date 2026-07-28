from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from symusic import Score
from tqdm.auto import tqdm

EVENT_SCHEMA = pa.schema(
    [
        ("track_id", pa.int16()),
        ("track_name", pa.string()),
        ("program", pa.int16()),
        ("is_drum", pa.bool_()),
        ("event_type", pa.string()),
        ("time_tick", pa.int64()),
        ("duration_tick", pa.int64()),
        ("time_quarter", pa.float64()),
        ("duration_quarter", pa.float64()),
        ("time_sec", pa.float64()),
        ("duration_sec", pa.float64()),
        ("pitch", pa.uint8()),
        ("velocity", pa.uint8()),
        ("number", pa.uint8()),
        ("value", pa.int32()),
        ("qpm", pa.float64()),
        ("numerator", pa.uint8()),
        ("denominator", pa.uint8()),
        ("key", pa.int8()),
        ("tonality", pa.int8()),
        ("text", pa.string()),
    ]
)


class MidiToCanonical:
    MIDI_SUFFIXES = {".mid", ".midi"}

    def __init__(self, midi_root: str | Path):
        self.midi_root = Path(midi_root).expanduser().resolve()
        if not self.midi_root.exists():
            raise FileNotFoundError(f"MIDI root does not exist: {self.midi_root}")
        if not self.midi_root.is_dir():
            raise NotADirectoryError(f"MIDI root is not a directory: {self.midi_root}")

    @staticmethod
    def _pair_events(tick_events, second_events, label: str):
        if len(tick_events) != len(second_events):
            raise ValueError(
                f"{label}: different event counts after conversion: "
                f"{len(tick_events)} != {len(second_events)}"
            )
        return zip(tick_events, second_events, strict=True)

    def _parse_piece(self, midi_path: Path) -> tuple[list[dict], dict]:
        score_tick = Score(midi_path)
        score_sec = score_tick.to("second")

        tpq = score_tick.ticks_per_quarter
        events: list[dict] = []

        for track_id, (track_tick, track_sec) in enumerate(
            zip(score_tick.tracks, score_sec.tracks, strict=True)
        ):
            track_fields = {
                "track_id": track_id,
                "track_name": track_tick.name,
                "program": track_tick.program,
                "is_drum": track_tick.is_drum,
            }

            for tick, sec in self._pair_events(
                track_tick.notes, track_sec.notes, "notes"
            ):
                events.append(
                    track_fields
                    | {
                        "event_type": "note",
                        "time_tick": tick.time,
                        "duration_tick": tick.duration,
                        "time_quarter": tick.time / tpq,
                        "duration_quarter": tick.duration / tpq,
                        "time_sec": sec.time,
                        "duration_sec": sec.duration,
                        "pitch": tick.pitch,
                        "velocity": tick.velocity,
                    }
                )

            for tick, sec in self._pair_events(
                track_tick.controls,
                track_sec.controls,
                "control changes",
            ):
                events.append(
                    track_fields
                    | {
                        "event_type": "control_change",
                        "time_tick": tick.time,
                        "time_quarter": tick.time / tpq,
                        "time_sec": sec.time,
                        "number": tick.number,
                        "value": tick.value,
                    }
                )

            for tick, sec in self._pair_events(
                track_tick.pitch_bends,
                track_sec.pitch_bends,
                "pitch bends",
            ):
                events.append(
                    track_fields
                    | {
                        "event_type": "pitch_bend",
                        "time_tick": tick.time,
                        "time_quarter": tick.time / tpq,
                        "time_sec": sec.time,
                        "value": tick.value,
                    }
                )

        global_fields = {
            "track_id": None,
            "track_name": None,
            "program": None,
            "is_drum": None,
        }

        for tick, sec in self._pair_events(
            score_tick.tempos, score_sec.tempos, "tempos"
        ):
            events.append(
                global_fields
                | {
                    "event_type": "tempo",
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "qpm": tick.qpm,
                }
            )

        for tick, sec in self._pair_events(
            score_tick.time_signatures,
            score_sec.time_signatures,
            "time signatures",
        ):
            events.append(
                global_fields
                | {
                    "event_type": "time_signature",
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "numerator": tick.numerator,
                    "denominator": tick.denominator,
                }
            )

        for tick, sec in self._pair_events(
            score_tick.key_signatures,
            score_sec.key_signatures,
            "key signatures",
        ):
            events.append(
                global_fields
                | {
                    "event_type": "key_signature",
                    "time_tick": tick.time,
                    "time_quarter": tick.time / tpq,
                    "time_sec": sec.time,
                    "key": tick.key,
                    "tonality": tick.tonality,
                }
            )

        for event_type, attribute in (("marker", "markers"), ("lyric", "lyrics")):
            if not hasattr(score_tick, attribute):
                continue

            tick_events = getattr(score_tick, attribute)
            second_events = getattr(score_sec, attribute)

            for tick, sec in self._pair_events(tick_events, second_events, event_type):
                events.append(
                    global_fields
                    | {
                        "event_type": event_type,
                        "time_tick": tick.time,
                        "time_quarter": tick.time / tpq,
                        "time_sec": sec.time,
                        "text": tick.text,
                    }
                )

        events.sort(
            key=lambda event: (
                event["time_tick"],
                -1 if event["track_id"] is None else event["track_id"],
                event["event_type"],
            )
        )

        statistics = {
            "ticks_per_quarter": tpq,
            "end_tick": score_tick.end(),
            "duration_sec": score_sec.end(),
            "num_tracks": len(score_tick.tracks),
            "num_notes": score_tick.note_num(),
            "num_events": len(events),
        }

        return events, statistics

    def _iter_midi_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.midi_root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.MIDI_SUFFIXES
        )

    def export(self, output_root: str | Path) -> pd.DataFrame:
        output_root = Path(output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        midi_files = self._iter_midi_files()
        manifest: list[dict] = []

        for index, midi_path in enumerate(
            tqdm(midi_files, desc="midi_to_canonical", unit="midi", leave=False),
            start=1,
        ):
            relative_midi = midi_path.relative_to(self.midi_root)
            output_path = (output_root / relative_midi).with_suffix(".parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            events, statistics = self._parse_piece(midi_path)
            table = pa.Table.from_pylist(events, schema=EVENT_SCHEMA)
            pq.write_table(
                table,
                output_path,
                compression="zstd",
                use_dictionary=["event_type", "track_name"],
            )

            manifest.append(
                {
                    "piece_id": hashlib.sha256(
                        str(relative_midi).encode("utf-8")
                    ).hexdigest()[:16],
                    "source_midi": str(relative_midi),
                    "canonical_events": str(output_path.relative_to(output_root)),
                    "source_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
                    **statistics,
                }
            )

            print(f"[{index:04d}/{len(midi_files)}] {relative_midi}")

        manifest_df = pd.DataFrame(manifest)
        manifest_df.to_parquet(output_root / "manifest.parquet", index=False)
        return manifest_df


if __name__ == "__main__":
    source_root = Path("/home/skynet/research/data/maestro/maestro-v3.0.0/")
    exporter = MidiToCanonical(source_root)
    exporter.export(
        Path("/home/skynet/research/data/maestro/maestro-v3.0.0_cononical/")
    )

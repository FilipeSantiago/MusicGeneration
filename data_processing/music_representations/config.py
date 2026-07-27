from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SegmentationConfig:
    enabled: bool = False
    segment_ticks: int | None = None
    hop_ticks: int | None = None
    min_notes: int = 1
    allow_incomplete_final_segment: bool = True


@dataclass(slots=True)
class NoteTableConfig:
    time_unit: str = "tick"


@dataclass(slots=True)
class PianoRollConfig:
    resolution: int = 24
    onset_window_ticks: int = 1
    velocity_dtype: str = "uint8"


@dataclass(slots=True)
class MidiTokRepresentationConfig:
    pitch_range: tuple[int, int] = (21, 109)
    beat_res: dict[tuple[int, int], int] = field(
        default_factory=lambda: {(0, 4): 8, (4, 12): 4}
    )
    num_velocities: int = 32
    use_velocities: bool = True
    use_tempos: bool = True
    use_time_signatures: bool = True
    use_sustain_pedals: bool = True
    use_pitch_bends: bool = True
    use_programs: bool = True
    use_rests: bool = False
    one_token_stream_for_programs: bool = True
    special_tokens: list[str] = field(
        default_factory=lambda: ["PAD", "BOS", "EOS", "MASK"]
    )
    remi_plus_use_programs: bool = True
    remi_plus_use_rests: bool = True
    remi_plus_use_sustain_pedals: bool = True


@dataclass(slots=True)
class SubwordConfig:
    base_representation: str = "remi"
    vocab_size: int = 512
    unk_token: str = "[UNK]"


@dataclass(slots=True)
class MusicRepresentationConfig:
    input_dir: Path
    output_dir: Path
    canonical_dir_name: str = "canonical"
    metadata_csv: Path | None = None
    schema_version: str = "1.0.0"
    representation_schema_version: str = "1.0.0"
    limit: int | None = None
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    note_table: NoteTableConfig = field(default_factory=NoteTableConfig)
    piano_roll: PianoRollConfig = field(default_factory=PianoRollConfig)
    miditok: MidiTokRepresentationConfig = field(
        default_factory=MidiTokRepresentationConfig
    )
    subword: SubwordConfig = field(default_factory=SubwordConfig)
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        payload = _normalize(asdict(self))
        payload["input_dir"] = str(self.input_dir)
        payload["output_dir"] = str(self.output_dir)
        if self.metadata_csv is not None:
            payload["metadata_csv"] = str(self.metadata_csv)
        return payload

    def config_hash(self) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MusicRepresentationConfig:
        segmentation = SegmentationConfig(**payload.pop("segmentation", {}))
        note_table = NoteTableConfig(**payload.pop("note_table", {}))
        piano_roll = PianoRollConfig(**payload.pop("piano_roll", {}))
        miditok = MidiTokRepresentationConfig(**payload.pop("miditok", {}))
        subword = SubwordConfig(**payload.pop("subword", {}))
        return cls(
            input_dir=Path(payload["input_dir"]),
            output_dir=Path(payload["output_dir"]),
            metadata_csv=Path(payload["metadata_csv"])
            if payload.get("metadata_csv")
            else None,
            canonical_dir_name=payload.get("canonical_dir_name", "canonical"),
            schema_version=payload.get("schema_version", "1.0.0"),
            representation_schema_version=payload.get(
                "representation_schema_version", "1.0.0"
            ),
            limit=payload.get("limit"),
            segmentation=segmentation,
            note_table=note_table,
            piano_roll=piano_roll,
            miditok=miditok,
            subword=subword,
            log_level=payload.get("log_level", "INFO"),
        )


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .config import MusicRepresentationConfig

NoteMatch = Literal["index", "pitch_onset"]
NoteCount = Literal["exact", "subset"]


@dataclass(frozen=True, slots=True)
class RoundTripContract:
    """What a representation promises to preserve across ``canonical -> X -> canonical``.

    Timing tolerances are expressed in **quarters**, never ticks: the MidiTok
    tokenizers decode into a Score whose ``ticks_per_quarter`` is their own beat
    resolution, so a tick-denominated tolerance would mean different things on
    either side of the round trip.
    """

    representation: str
    note_match: NoteMatch = "index"
    note_count: NoteCount = "exact"
    onset_tol_quarters: float = 0.0
    duration_tol_quarters: float = 0.0
    velocity_tol: int | None = 0
    preserves_tempos: bool = True
    preserves_time_signatures: bool = True
    preserves_key_signatures: bool = True
    preserved_control_numbers: tuple[int, ...] | None = None
    preserves_pitch_bends: bool = True
    preserves_tracks: bool = True
    preserves_ticks_per_quarter: bool = True
    preserves_absolute_onset: bool = True
    extra_loss: tuple[str, ...] = field(default_factory=tuple)

    def describe_loss(self) -> list[str]:
        lines: list[str] = []
        if self.onset_tol_quarters:
            lines.append(
                f"note onsets are quantized to within {self.onset_tol_quarters:g} quarters"
            )
        if self.duration_tol_quarters:
            lines.append(
                f"note durations are quantized to within {self.duration_tol_quarters:g} quarters"
            )
        if self.velocity_tol is None:
            lines.append("note velocities are not recoverable")
        elif self.velocity_tol:
            lines.append(f"note velocities are binned to within +/-{self.velocity_tol}")
        if self.note_count == "subset":
            lines.append("notes outside the configured pitch range are dropped")
        if not self.preserves_tempos:
            lines.append("tempo changes are stored only in the canonical source tables")
        if not self.preserves_time_signatures:
            lines.append("time signatures are stored only in the canonical source tables")
        if not self.preserves_key_signatures:
            lines.append("key signatures are stored only in the canonical source tables")
        if self.preserved_control_numbers is not None:
            if self.preserved_control_numbers:
                kept = ", ".join(str(number) for number in self.preserved_control_numbers)
                lines.append(f"only control changes {kept} survive; the rest are dropped")
            else:
                lines.append("control changes are stored only in the canonical source tables")
        if not self.preserves_pitch_bends:
            lines.append("pitch bends are stored only in the canonical source tables")
        if not self.preserves_tracks:
            lines.append("track identity (program, drum flag, name) is not recoverable")
        if not self.preserves_ticks_per_quarter:
            lines.append("the decoded time division is the tokenizer's, not the source's")
        if not self.preserves_absolute_onset:
            lines.append(
                "only relative timing survives; the decoded piece always starts at tick 0"
            )
        lines.append("note_id is regenerated on decode and carries no source meaning")
        lines.extend(self.extra_loss)
        return lines

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extra_loss"] = list(self.extra_loss)
        if self.preserved_control_numbers is not None:
            payload["preserved_control_numbers"] = list(self.preserved_control_numbers)
        return payload


def _lossless(name: str) -> RoundTripContract:
    return RoundTripContract(representation=name)


def _note_table_contract(config: MusicRepresentationConfig) -> RoundTripContract:
    bin_size = config.note_table.bin_size
    return RoundTripContract(
        representation="note_table",
        # The stored table keeps an exact float onset and an exact velocity;
        # only durations pass through the bin grid.
        onset_tol_quarters=1 / 480,
        duration_tol_quarters=bin_size / 2 + 1 / 480,
        velocity_tol=0,
        preserves_tempos=False,
        preserves_time_signatures=False,
        preserves_key_signatures=False,
        preserved_control_numbers=(),
        preserves_pitch_bends=False,
    )


def note_table_tuple_contract(config: MusicRepresentationConfig) -> RoundTripContract:
    """Contract for the in-memory `(pitch, velocity_bin, delta_onset_bin, duration_bin)`
    form the n-gram model emits, which is strictly lossier than the stored table."""
    bin_size = config.note_table.bin_size
    half_bin = config.note_table.velocity_bin_size // 2
    return RoundTripContract(
        representation="note_table",
        note_match="pitch_onset",
        onset_tol_quarters=bin_size / 2,
        duration_tol_quarters=bin_size / 2,
        velocity_tol=half_bin,
        preserves_tempos=False,
        preserves_time_signatures=False,
        preserves_key_signatures=False,
        preserved_control_numbers=(),
        preserves_pitch_bends=False,
        preserves_tracks=False,
        preserves_absolute_onset=False,
        extra_loss=(
            (
                "onsets are rebuilt from independently rounded deltas, so timing "
                "error accumulates along the sequence"
            ),
        ),
    )


def _piano_roll_contract(config: MusicRepresentationConfig) -> RoundTripContract:
    frame = 1 / config.piano_roll.resolution
    return RoundTripContract(
        representation="piano_roll",
        note_match="pitch_onset",
        note_count="subset",
        onset_tol_quarters=frame,
        # The builder floors the start frame and ceils the end frame, so a note's
        # frame span can be two frames wider than the note actually is.
        duration_tol_quarters=2 * frame,
        velocity_tol=0,
        preserves_tempos=False,
        preserves_time_signatures=False,
        preserves_key_signatures=False,
        preserved_control_numbers=(64,),
        preserves_pitch_bends=False,
        preserves_tracks=False,
        extra_loss=(
            "all tracks are flattened into one 128-pitch roll",
            "sustain is binarized at CC64 >= 64, so decoded values are only 0 or 127",
            "same-pitch notes landing on one frame merge irrecoverably",
        ),
    )


def _miditok_contract(name: str, config: MusicRepresentationConfig) -> RoundTripContract:
    # MidiTok silently disables attributes a tokenization cannot express (Structured
    # drops tempos and time signatures; the multi-vocabulary ones drop pitch bends and
    # sustain). Reading the flags back off the constructed tokenizer is the only way
    # the contract describes what actually happens.
    from ..music_representations.representations.miditok.builders import build_tokenizer

    cfg = build_tokenizer(config, name).config
    # beat_res maps a beat range to its resolution, e.g. {(0, 4): 8, (4, 12): 4}.
    # Positions land on the finest grid, but a duration long enough to reach the
    # coarse range is quantized there, so the two tolerances are not the same.
    finest_grid = 1 / max(cfg.beat_res.values())
    coarsest_grid = 1 / min(cfg.beat_res.values())
    return RoundTripContract(
        representation=name,
        note_match="pitch_onset",
        note_count="subset",
        onset_tol_quarters=finest_grid,
        duration_tol_quarters=coarsest_grid,
        velocity_tol=(128 // cfg.num_velocities if cfg.use_velocities else None),
        preserves_tempos=cfg.use_tempos,
        preserves_time_signatures=cfg.use_time_signatures,
        preserves_key_signatures=False,
        preserved_control_numbers=(64,) if cfg.use_sustain_pedals else (),
        preserves_pitch_bends=cfg.use_pitch_bends,
        preserves_tracks=cfg.use_programs,
        preserves_ticks_per_quarter=False,
        extra_loss=(
            ("rests are re-derived from the token grid",) if cfg.use_rests else ()
        ),
    )


MIDITOK_NAMES = (
    "midilike",
    "tsd",
    "remi",
    "remi_plus",
    "structured",
    "cpword",
    "octuple",
    "pertok",
)
SUBWORD_NAMES = ("bpe", "unigram", "wordpiece")


def contract_for(name: str, config: MusicRepresentationConfig) -> RoundTripContract:
    """Materialize a representation's contract against the effective configuration.

    Several tolerances are config-derived (`bin_size`, `resolution`, `beat_res`,
    `num_velocities`), so the contract cannot be a plain constant.
    """
    if name == "canonical":
        return _lossless("canonical")
    if name == "note_table":
        return _note_table_contract(config)
    if name == "piano_roll":
        return _piano_roll_contract(config)
    if name in MIDITOK_NAMES:
        return _miditok_contract(name, config)
    if name in SUBWORD_NAMES:
        base = _miditok_contract(config.subword.base_representation, config)
        return replace(
            base,
            representation=name,
            extra_loss=(
                *base.extra_loss,
                f"tokens are re-encoded through the trained {name} vocabulary",
            ),
        )
    raise KeyError(f"No round-trip contract for representation: {name}")

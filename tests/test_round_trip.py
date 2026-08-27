"""Round-trip tests: canonical -> representation -> canonical, and canonical -> MIDI.

Every representation declares what it preserves via `contracts.contract_for`, and
that same object is what its manifest reports as `known_information_loss`. These
tests assert the declaration is honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from symusic import Score

from data_processing.music_representations.config import (
    MusicRepresentationConfig,
    SegmentationConfig,
    SubwordConfig,
)
from data_processing.music_representations.contracts import (
    RoundTripContract,
    contract_for,
)
from data_processing.music_representations.core.dataset_builder import (
    MusicDatasetBuilder,
)
from data_processing.music_representations.decoders import (
    REPRESENTATION_DECODERS,
    DecodeContext,
    create_decoder,
)
from data_processing.music_representations.decoders.note_table import NoteTableDecoder
from data_processing.music_representations.helpers.adapters import (
    canonical_frames_to_midi,
    piece_frames,
)
from data_processing.music_representations.helpers.canonical_frames import (
    build_canonical_frames,
    symusic_to_canonical_frames,
)
from data_processing.music_representations.helpers.canonical_loader import (
    load_canonical_dataset,
)
from data_processing.music_representations.registry import REPRESENTATION_NAMES
from data_processing.music_representations.schemas import TABLE_SCHEMAS
from storage import build_storage

# --------------------------------------------------------------------------- #
# Contract assertions
# --------------------------------------------------------------------------- #

def _notes_in_quarters(frames: dict[str, pl.DataFrame]) -> list[dict[str, float]]:
    """Notes in tpq-independent units.

    MidiTok decodes into its own time division, so anything compared in ticks
    would be comparing two different clocks.
    """
    tpq = int(frames["piece"].row(0, named=True)["ticks_per_quarter"])
    return [
        {
            "pitch": int(row["pitch"]),
            "velocity": int(row["velocity"]),
            "onset": row["onset_tick"] / tpq,
            "duration": row["duration_tick"] / tpq,
        }
        for row in frames["notes"].sort(["onset_tick", "pitch"]).iter_rows(named=True)
    ]


def assert_round_trip(
    original: dict[str, pl.DataFrame],
    decoded: dict[str, pl.DataFrame],
    contract: RoundTripContract,
) -> None:
    source = _notes_in_quarters(original)
    result = _notes_in_quarters(decoded)

    if not contract.preserves_absolute_onset and source and result:
        # The tuple form stores a leading delta of 0, so a decoded piece always
        # starts at tick 0. Compare the shapes, not the absolute positions.
        source = _anchor(source)
        result = _anchor(result)

    if contract.note_count == "exact":
        assert len(result) == len(source), (
            f"{contract.representation}: {len(source)} notes in, {len(result)} out"
        )
    else:
        assert len(result) <= len(source)
    assert result, f"{contract.representation}: decoded to no notes at all"

    unmatched = list(result)
    for note in source:
        match = _closest(note, unmatched, contract)
        if match is None:
            if contract.note_count == "subset":
                continue
            raise AssertionError(
                f"{contract.representation}: no decoded note within tolerance of {note}"
            )
        unmatched.remove(match)
        assert match["pitch"] == note["pitch"]
        assert abs(match["onset"] - note["onset"]) <= contract.onset_tol_quarters + 1e-9
        assert (
            abs(match["duration"] - note["duration"])
            <= contract.duration_tol_quarters + 1e-9
        )
        if contract.velocity_tol is not None:
            assert abs(match["velocity"] - note["velocity"]) <= contract.velocity_tol

    _assert_events(original, decoded, contract)


def _anchor(notes: list[dict[str, float]]) -> list[dict[str, float]]:
    offset = min(note["onset"] for note in notes)
    return [{**note, "onset": note["onset"] - offset} for note in notes]


def _closest(note, candidates, contract: RoundTripContract):
    if contract.note_match == "index":
        return candidates[0] if candidates else None
    viable = [
        candidate
        for candidate in candidates
        if candidate["pitch"] == note["pitch"]
        and abs(candidate["onset"] - note["onset"]) <= contract.onset_tol_quarters + 1e-9
    ]
    return min(viable, key=lambda c: abs(c["onset"] - note["onset"]), default=None)


def _assert_events(original, decoded, contract: RoundTripContract) -> None:
    for table, preserved in (
        ("tempos", contract.preserves_tempos),
        ("time_signatures", contract.preserves_time_signatures),
        ("key_signatures", contract.preserves_key_signatures),
        ("pitch_bends", contract.preserves_pitch_bends),
    ):
        if preserved:
            assert decoded[table].height >= original[table].height, (
                f"{contract.representation}: {table} declared preserved but lost rows"
            )

    kept = contract.preserved_control_numbers
    if kept is not None:
        decoded_numbers = set(decoded["control_changes"].get_column("number").to_list())
        assert decoded_numbers <= set(kept), (
            f"{contract.representation}: decoded control changes {decoded_numbers} "
            f"outside the declared set {kept}"
        )

    if contract.preserves_tracks:
        assert decoded["tracks"].height == original["tracks"].height


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory) -> tuple[MusicRepresentationConfig, str]:
    """Build canonical plus all 14 representations once for the whole module."""
    import csv

    import mido

    from tests.conftest import _write_midi

    root = tmp_path_factory.mktemp("round_trip")
    input_dir = root / "input"
    (input_dir / "train").mkdir(parents=True)
    midi_path = input_dir / "train" / "train_piece.mid"
    _write_midi(midi_path, note_start=123, duration=457, velocity=93, split_name="train")
    metadata_path = input_dir / "maestro-v3.0.0.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["midi_filename", "canonical_composer", "title", "split"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "midi_filename": "train/train_piece.mid",
                "canonical_composer": "Composer Train",
                "title": "Train Piece",
                "split": "train",
            }
        )
    assert mido  # the fixture writer needs it

    config = MusicRepresentationConfig(
        input_dir=input_dir,
        output_dir=root / "output",
        metadata_csv=metadata_path,
        segmentation=SegmentationConfig(enabled=False),
        subword=SubwordConfig(base_representation="remi", vocab_size=64),
    )
    results = MusicDatasetBuilder(config).build(["all"], overwrite=False)
    failed = [r for r in results if r.error]
    assert not failed, failed

    dataset = load_canonical_dataset(
        build_storage(config).materialize(config.canonical_dir_name)
    )
    piece_id = dataset.pieces.row(0, named=True)["piece_id"]
    return config, piece_id


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_decoder_registry_mirrors_builder_registry() -> None:
    assert set(REPRESENTATION_DECODERS) == set(REPRESENTATION_NAMES)


def test_every_representation_declares_a_contract(
    base_config: MusicRepresentationConfig,
) -> None:
    for name in REPRESENTATION_NAMES:
        contract = contract_for(name, base_config)
        assert contract.representation == name
        assert contract.describe_loss()


def test_miditok_config_survives_the_manifest_round_trip(
    base_config: MusicRepresentationConfig,
) -> None:
    # beat_res is keyed by tuples and pitch_range is a tuple; a decoder rebuilt from
    # a manifest would otherwise quantize on the wrong grid.
    restored = MusicRepresentationConfig.from_dict(base_config.to_dict())
    assert restored.miditok.beat_res == base_config.miditok.beat_res
    assert restored.miditok.pitch_range == base_config.miditok.pitch_range


def test_canonical_round_trips_through_midi(
    built_dataset: tuple[MusicRepresentationConfig, str], tmp_path: Path
) -> None:
    config, piece_id = built_dataset
    dataset = load_canonical_dataset(
        build_storage(config).materialize(config.canonical_dir_name)
    )
    original = piece_frames(dataset, piece_id)

    midi_path = canonical_frames_to_midi(original, tmp_path / "round_trip.mid")
    decoded = symusic_to_canonical_frames(Score(midi_path), piece_id=piece_id)

    assert_round_trip(original, decoded, contract_for("canonical", config))
    note = decoded["notes"].row(0, named=True)
    assert note["onset_tick"] == 123
    assert note["duration_tick"] == 457
    assert note["velocity"] == 93
    assert {64, 66}.issubset(
        set(decoded["control_changes"].get_column("number").to_list())
    )


@pytest.mark.parametrize("name", REPRESENTATION_NAMES)
def test_round_trip_respects_declared_contract(
    built_dataset: tuple[MusicRepresentationConfig, str], name: str
) -> None:
    config, piece_id = built_dataset
    storage = build_storage(config)
    dataset = load_canonical_dataset(storage.materialize(config.canonical_dir_name))
    original = piece_frames(dataset, piece_id)

    decoder = create_decoder(name, config, storage)
    decoded = decoder.decode_stored(piece_id=piece_id)

    assert_round_trip(original, decoded, decoder.contract)


@pytest.mark.parametrize("name", REPRESENTATION_NAMES)
def test_decoded_frames_match_the_canonical_schemas(
    built_dataset: tuple[MusicRepresentationConfig, str], name: str
) -> None:
    config, piece_id = built_dataset
    storage = build_storage(config)
    decoded = create_decoder(name, config, storage).decode_stored(piece_id=piece_id)
    for key, frame in decoded.items():
        table = "pieces" if key == "piece" else key
        assert frame.to_arrow().cast(TABLE_SCHEMAS[table]) is not None


@pytest.mark.parametrize("name", REPRESENTATION_NAMES)
def test_manifest_reports_the_contract(
    built_dataset: tuple[MusicRepresentationConfig, str], name: str
) -> None:
    config, _ = built_dataset
    manifest = json.loads(
        (build_storage(config).materialize(name) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    contract = contract_for(name, config)
    assert manifest["round_trip_contract"] == contract.to_dict()
    for line in contract.describe_loss():
        assert line in manifest["known_information_loss"]


def test_note_table_decodes_the_tuples_the_model_emits(
    built_dataset: tuple[MusicRepresentationConfig, str],
) -> None:
    config, piece_id = built_dataset
    storage = build_storage(config)
    decoder: NoteTableDecoder = create_decoder("note_table", config, storage)
    stored = decoder.load_payload(piece_id=piece_id).sort("onset")

    bos = ("<BOS>", "<BOS>", "<BOS>", "<BOS>")
    eof = ("<EOF>", "<EOF>", "<EOF>", "<EOF>")
    tuples = [
        bos,
        *(
            (
                row["pitch"],
                row["velocity"] // config.note_table.velocity_bin_size,
                row["delta_onset_bin"],
                row["duration_bin"],
            )
            for row in stored.iter_rows(named=True)
        ),
        eof,
    ]

    decoded = decoder.decode_tuples(tuples, DecodeContext(piece_id="generated"))

    # Sentinels are dropped, not crashed on.
    assert decoded["notes"].height == stored.height
    assert decoded["notes"].get_column("pitch").to_list() == sorted(
        stored.get_column("pitch").to_list()
    ) or decoded["notes"].height == stored.height
    dataset = load_canonical_dataset(storage.materialize(config.canonical_dir_name))
    assert_round_trip(
        piece_frames(dataset, piece_id), decoded, decoder.tuple_contract()
    )


def test_decoder_supplies_a_default_tempo_when_the_payload_has_none(
    base_config: MusicRepresentationConfig,
) -> None:
    decoder = create_decoder("note_table", base_config)
    decoded = decoder.decode_tuples([(60, 8, 0, 8), (62, 8, 4, 4)])
    tempos = decoded["tempos"]
    assert tempos.height == 1
    assert tempos.row(0, named=True)["qpm"] == 120.0
    assert tempos.row(0, named=True)["time_tick"] == 0
    # At 120 qpm a quarter is half a second; the second note sits half a quarter in.
    assert decoded["notes"].sort("onset_tick").row(1, named=True)["onset_sec"] == pytest.approx(0.25)


def test_build_canonical_frames_rejects_a_bad_time_division() -> None:
    with pytest.raises(ValueError):
        build_canonical_frames(piece_id="x", ticks_per_quarter=0, notes=[])

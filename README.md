# MusicGeneration

This project builds machine-learning-ready music datasets from symbolic piano performance data, with a focus on MAESTRO-style MIDI corpora.

At a high level, it does two jobs:

1. Parse raw MIDI files into a canonical, lossless, library-neutral dataset.
2. Derive multiple research representations from that canonical source without reparsing the original MIDI files.

The code is designed so that the canonical dataset is the source of truth and every downstream representation is reproducible from it.

## What This Project Does

This is a symbolic music preprocessing pipeline.

It is for situations where you want to train or evaluate models on music, but the raw input is MIDI and the model needs something more structured, such as:

- note tables;
- piano rolls;
- event-token sequences;
- subword tokenizations over event sequences.

The pipeline currently supports these target representations:

- `canonical`
- `note_table`
- `piano_roll`
- `midilike`
- `tsd`
- `remi`
- `remi_plus`
- `structured`
- `cpword`
- `octuple`
- `pertok`
- `bpe`
- `unigram`
- `wordpiece`

It does **not** implement raw MIDI-byte modeling.

## Why Symbolic Music Needs Multiple Representations

Music is not one thing computationally.

The same performed piano passage can be described in several valid ways depending on what a model needs to learn.

### Performance vs score

A performed MIDI file is not a printed score.

A score says what notes should exist and how they are grouped metrically. A performance adds timing deviations, articulation, rubato, pedal use, dynamics, and other expressive details. MAESTRO is especially useful because it captures expressive piano performance rather than only score-like note grids.

That means the preprocessing layer has to preserve:

- exact onset and duration in ticks;
- timing in seconds;
- tempo changes;
- meter changes;
- velocity;
- pedal and other control changes;
- pitch bends when present;
- stable train/validation/test split metadata.

### Notes, rhythm, meter, and harmony

From a music-theory perspective:

- A **note** has pitch, start time, duration, and loudness.
- **Rhythm** is the pattern of durations and onsets.
- **Meter** organizes time into beats and measures, usually through time signatures like `4/4` or `3/4`.
- **Tempo** determines how quickly beats unfold in real time.
- **Harmony** emerges when notes sound together or in sequence, forming chords and tonal motion.
- **Articulation and expression** appear in performed timing, note lengths, dynamics, and pedal usage.

Different model families emphasize different subsets of that information.

### Why several representations exist

- A **note table** is convenient for tabular models, statistics, and explicit event analysis.
- A **piano roll** is convenient for convolutional or image-like models because time and pitch form a grid.
- A **token sequence** is convenient for language-model-style architectures because music becomes a stream of discrete symbols.
- A **subword model** is useful when frequently recurring token patterns should become higher-level units.

No single representation is best for all research questions.

## Canonical Dataset

The canonical dataset is the foundation of the project.

Its purpose is to preserve the musical content of the input MIDI files without committing to any one downstream modeling representation.

### Design goals

The canonical layer is:

- unquantized;
- unsegmented by default;
- independent of model-specific tokenization;
- independent of MusPy, MidiTok, and Pypianoroll storage formats;
- stored in project-owned Parquet tables.

### What it stores

The canonical dataset preserves, when present:

- piece identity;
- MAESTRO split;
- source MIDI path;
- source hash;
- composer;
- title;
- ticks per quarter note;
- note onsets and durations in ticks;
- note onsets and durations in quarter-note units;
- note onsets and durations in seconds;
- pitch;
- velocity;
- track metadata;
- tempo changes;
- time-signature changes;
- key-signature changes;
- control changes, including sustain pedal;
- pitch bends.

### Canonical files

The canonical output directory contains Parquet tables such as:

- `pieces.parquet`
- `tracks.parquet`
- `notes.parquet`
- `tempos.parquet`
- `time_signatures.parquet`
- `key_signatures.parquet`
- `control_changes.parquet`
- `pitch_bends.parquet`

It also contains a manifest describing schema version, dependency versions, configuration, and dataset fingerprinting.

## Derived Representations

All derived representations consume the canonical dataset.

That is an important architectural decision. It means:

- parsing is centralized;
- splits stay consistent;
- failures in one representation do not require reparsing the corpus;
- experimental representations remain comparable.

### Note table

The note-table export is the closest derived format to canonical note events.

It provides a flat, model-friendly table with fields such as:

- piece id;
- segment id;
- split;
- pitch;
- velocity;
- onset;
- duration;
- track id;
- program;
- drum flag;
- track name.

This is useful when the model or analysis pipeline wants explicit note events rather than a dense grid or token stream.

### Piano roll

A piano roll represents music on a time-by-pitch grid.

In piano pedagogy and composition, a piano roll is analogous to a perforated player-piano roll or a DAW piano-roll editor: horizontal position means time, vertical position means pitch, and note length becomes a horizontal span.

This project exports piano-roll-aligned arrays for:

- active notes;
- note onsets;
- note velocities;
- sustain state.

That preserves more expressive information than a bare binary roll.

### Event token representations

The token representations are based on MidiTok.

These convert symbolic music into discrete event vocabularies, similar to turning text into tokens for a language model.

From a music-theory angle, this is useful because musical structure can be expressed as ordered symbolic decisions:

- where we are in time;
- what pitch occurs;
- how long it lasts;
- how loud it is;
- what meter or tempo context applies.

Supported token families include:

- **MIDILike**: close to MIDI-style event ordering.
- **TSD**: time-shift and duration event modeling.
- **REMI**: meter-aware event representation with bar and position concepts.
- **REMI+**: configured REMI variant with additional enabled features.
- **Structured**: stricter symbolic event ordering.
- **CPWord**: compound token strategy.
- **Octuple**: bundled multi-attribute token structure.
- **PerTok**: performance-oriented tokenization path.

### Subword representations

The subword builders train over a base token representation, defaulting to `remi`.

This is similar in spirit to subword tokenization in NLP:

- **BPE** merges common token sequences.
- **Unigram** learns a probabilistic subtoken inventory.
- **WordPiece** builds a vocabulary from frequent compositional pieces.

In music, these can capture recurring symbolic fragments such as:

- stereotyped rhythmic cells;
- common melodic gestures;
- repeated accompaniment patterns;
- recurring event combinations in a local style.

The vocabulary is trained on the training split only, then applied unchanged to validation and test.

## Project Structure

Important paths:

- [`main.py`](./main.py): root runner for PyCharm or command-line use.
- [`data_processing/music_representations/`](./data_processing/music_representations): preprocessing package.
- [`.env`](./.env): default runtime configuration for `main.py`.
- [`tests/`](./tests): synthetic-fixture tests for the preprocessing layer.

## Running the Project

The intended entrypoint is the root runner:

```bash
uv run python main.py
```

That runner can also still accept explicit CLI flags, but the normal PyCharm workflow is:

1. edit `.env`;
2. run `main.py`.

## Environment Variables

The root runner reads `.env` from the repository root and loads variables with the prefix `MUSIC_REPR_`.

If CLI arguments are passed, they override `.env`. If no CLI arguments are passed, `main.py` uses `.env`.

### Required variables

#### `MUSIC_REPR_INPUT_DIR`

Path to the source MIDI dataset root.

For a MAESTRO-style dataset, this should point at the directory containing:

- the split-organized MIDI files;
- optionally `maestro-v3.0.0.csv` metadata.

Example:

```env
MUSIC_REPR_INPUT_DIR=/home/you/data/maestro/maestro-v3.0.0
```

#### `MUSIC_REPR_OUTPUT_DIR`

Path where the built representations will be written.

Each representation gets its own subdirectory under this root.

Example:

```env
MUSIC_REPR_OUTPUT_DIR=/home/you/data/maestro_preprocessed
```

### Optional variables

#### `MUSIC_REPR_REPRESENTATIONS`

Comma-separated list of representations to build.

If omitted, the runner defaults to:

```env
MUSIC_REPR_REPRESENTATIONS=all
```

Examples:

```env
MUSIC_REPR_REPRESENTATIONS=canonical
```

```env
MUSIC_REPR_REPRESENTATIONS=canonical,remi,piano_roll
```

Supported names:

- `canonical`
- `note_table`
- `piano_roll`
- `midilike`
- `tsd`
- `remi`
- `remi_plus`
- `structured`
- `cpword`
- `octuple`
- `pertok`
- `bpe`
- `unigram`
- `wordpiece`
- `all`

#### `MUSIC_REPR_OVERWRITE`

Whether to rebuild a representation directory if it already exists.

Accepted truthy forms:

- `1`
- `true`
- `yes`
- `on`

Anything else is treated as false.

Example:

```env
MUSIC_REPR_OVERWRITE=false
```

#### `MUSIC_REPR_LIMIT`

Optional deterministic limit on how many pieces to process.

This is mainly for smoke tests and local debugging. It does not reshuffle splits; it simply truncates the sorted file list.

Example:

```env
MUSIC_REPR_LIMIT=16
```

#### `MUSIC_REPR_CONFIG`

Optional path to a JSON configuration file.

This allows you to override deeper settings such as segmentation, tokenizer configuration, or piano-roll resolution without editing code.

Example:

```env
MUSIC_REPR_CONFIG=/home/you/research/MusicGeneration/configs/remi_experiment.json
```

## Example `.env`

```env
MUSIC_REPR_INPUT_DIR=/home/you/data/maestro/maestro-v3.0.0
MUSIC_REPR_OUTPUT_DIR=/home/you/data/maestro_preprocessed
MUSIC_REPR_REPRESENTATIONS=all
MUSIC_REPR_OVERWRITE=false
MUSIC_REPR_LIMIT=
MUSIC_REPR_CONFIG=
```

## Reproducibility and Safety

The builders are designed to be conservative about filesystem state.

### Skip behavior

If a representation output directory already exists, it is skipped by default.

### Overwrite behavior

If overwrite is enabled:

- the builder writes into a temporary sibling directory;
- the existing target is replaced only after a successful build;
- a failed build does not replace a previously valid directory.

### Why this matters

Preprocessing large music corpora is expensive. If one derived representation fails, the others should remain usable. This is especially important in research workflows where tokenization experiments may change frequently but the canonical source should remain stable.

## Music-Theory Interpretation of the Main Formats

### Tick time

MIDI tick time is relative symbolic time. It does not directly mean seconds; it means subdivisions of a beat according to the file’s ticks-per-quarter-note setting.

This matters because:

- metric structure lives naturally in beat-relative units;
- expressive timing lives naturally in seconds as well;
- both need to be preserved for performance modeling.

### Velocity

Velocity is not identical to acoustic loudness, but in piano MIDI it acts as a practical proxy for attack intensity and dynamic shaping.

Musically, it supports distinctions such as:

- melody vs accompaniment emphasis;
- legato shaping through attack contour;
- phrase peaks;
- accent patterns.

### Sustain pedal

For piano, CC64 is structurally important.

Without sustain information, the symbolic note list misses a large part of real piano sonority. Harmonic overlap, resonance, and legato behavior can all depend on pedal use, even when the raw note durations do not show it.

### Tempo and rubato

Tempo changes are not only technical metadata. They are part of phrasing and expressive timing.

A performance model that ignores tempo context can lose:

- ritardando and accelerando behavior;
- phrase-boundary timing;
- beat-level expressive stretch.

### Time signature and bar position

Representations like REMI make explicit use of barlines and positions inside the bar.

That mirrors a central fact of tonal and metrical music: listeners and performers do not hear time as an undifferentiated stream. They hear strong and weak beats, measure boundaries, pickup gestures, syncopation, and phrase alignment to meter.

### Harmony and texture

Even though this project does not yet build explicit harmonic analyses, several representations indirectly support harmonic learning because they preserve:

- simultaneity;
- duration overlap;
- metric placement;
- voice-leading-compatible note structure.

That allows downstream models to infer chordal behavior, accompaniment patterns, arpeggiation, and contrapuntal density from the symbolic data.

## Current Limitations

Some limits come from the current implementation and some from upstream library behavior.

- Lyrics and marker text are not currently persisted in the canonical tables.
- Piano-roll exports keep key expressive channels, but tempo changes, non-sustain CCs, and pitch bends remain canonical-side information rather than roll-native channels.
- The installed `miditok` version emits warnings for some attribute-control combinations and disables them internally in some tokenizers.
- Some tokenizers in the installed `miditok` surface require fallback handling for robust export in this project.

## Development Notes

Basic local verification commands:

```bash
uv run pytest -q
```

```bash
uv run ruff check .
```

```bash
uv run mypy data_processing tests main.py
```

## Summary

This project is a symbolic music representation factory.

Its core research idea is simple:

- parse once;
- preserve the musical truth in canonical form;
- derive many model-facing views from the same source;
- keep splits, metadata, and reproducibility stable across experiments.

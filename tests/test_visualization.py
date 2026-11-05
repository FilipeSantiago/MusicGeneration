"""Piano-roll video rendering: keyboard geometry, timeline queries, frame drawing,
ffmpeg encoding (with and without audio) and the CLI."""

from __future__ import annotations

import io
import math
import re
import shutil
import struct
import subprocess
import warnings
import wave
from pathlib import Path

import mido
import numpy as np
import pytest

from data_processing.music_representations.helpers.canonical_frames import (
    build_canonical_frames,
)
from visualization import (
    Card,
    Clip,
    ComparisonItem,
    KeyboardLayout,
    PianoRollRenderer,
    RenderConfig,
    TextLine,
    Timeline,
    VideoFormat,
    export_comparison,
    export_piece,
    format_config,
    render_piano_roll_video,
    render_sequence,
)
from visualization.__main__ import main as cli_main
from visualization.audio import AudioUnavailableError, cut_samples, synthesize_wav
from visualization.colors import track_color, velocity_shade
from visualization.renderer import WHITE_KEY, fit_text, pygame, wrap_text

TPQ = 480
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _frames(
    notes: list[dict],
    *,
    tempos: list[dict] | None = None,
    time_signatures: list[dict] | None = None,
    control_changes: list[dict] | None = None,
):
    return build_canonical_frames(
        piece_id="viz",
        ticks_per_quarter=TPQ,
        notes=[{"track_id": 0, "velocity": 100, **note} for note in notes],
        tempos=tempos or [{"time_tick": 0, "qpm": 120.0}],
        time_signatures=time_signatures or [],
        control_changes=control_changes or [],
        metadata={"title": "Viz Test", "composer": "Tester"},
    )


def _probe(path: Path) -> tuple[str, float, int]:
    """(ffmpeg stderr, decoded duration in seconds, decoded video frames).

    Fed through stdin so sandboxed ffmpeg builds can read pytest's tmp dir.
    """
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", "pipe:0", "-f", "null", "-"],
        input=path.read_bytes(),
        capture_output=True,
        check=True,
    )
    log = result.stderr.decode(errors="replace")
    frames = int(re.findall(r"frame=\s*(\d+)", log)[-1])
    h, m, s = re.findall(r"time=(\d+):(\d+):([\d.]+)", log)[-1]
    return log, int(h) * 3600 + int(m) * 60 + float(s), frames


def _sine_wav(seconds: float, sample_rate: int = 22050) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        samples = (
            struct.pack(
                "<h", round(6000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            )
            for i in range(round(seconds * sample_rate))
        )
        handle.writeframes(b"".join(samples))
    return buffer.getvalue()


# -- keyboard ---------------------------------------------------------------


def test_piano_keyboard_has_52_white_and_36_black_keys():
    layout = KeyboardLayout.build(21, 108, 1040)
    assert len(layout.keys) == 88
    assert len(layout.white_keys) == 52
    assert len(layout.black_keys) == 36
    assert layout.white_width == pytest.approx(20.0)


def test_black_keys_sit_on_white_key_boundaries():
    layout = KeyboardLayout.build(21, 108, 1040)
    c4, c_sharp4, d4 = layout.key(60), layout.key(61), layout.key(62)
    assert not c4.is_black and c_sharp4.is_black and not d4.is_black
    assert c_sharp4.x + c_sharp4.width / 2 == pytest.approx(d4.x)
    whites = [key.x for key in layout.white_keys]
    assert whites == sorted(whites)
    assert layout.white_keys[-1].x + layout.white_width == pytest.approx(1040)


def test_keyboard_rejects_pitches_outside_range():
    layout = KeyboardLayout.build(21, 108, 1040)
    assert not layout.contains(20)
    with pytest.raises(KeyError):
        layout.key(109)


# -- timeline ---------------------------------------------------------------


def test_active_and_visible_notes():
    # 120 qpm: one quarter (480 ticks) = 0.5 s.
    timeline = Timeline.from_frames(
        _frames(
            [
                {"onset_tick": 0, "duration_tick": 480, "pitch": 60},  # 0.0-0.5 s
                {"onset_tick": 240, "duration_tick": 1920, "pitch": 64},  # 0.25-2.25 s
                {"onset_tick": 1920, "duration_tick": 480, "pitch": 67},  # 2.0-2.5 s
            ]
        )
    )
    pitches = lambda idx: sorted(int(timeline.pitch[i]) for i in idx)
    assert pitches(timeline.active(0.1)) == [60]
    assert pitches(timeline.active(0.3)) == [60, 64]
    assert pitches(timeline.active(0.5)) == [64]  # end is exclusive
    assert pitches(timeline.active(2.1)) == [64, 67]
    assert pitches(timeline.visible(0.6, 1.0)) == [64]
    assert pitches(timeline.visible(0.6, 1.5)) == [64, 67]
    assert timeline.duration_sec == pytest.approx(2.5)
    assert (timeline.title, timeline.composer) == ("Viz Test", "Tester")


def test_sustain_follows_cc64():
    timeline = Timeline.from_frames(
        _frames(
            [{"onset_tick": 0, "duration_tick": 480, "pitch": 60}],
            control_changes=[
                {"track_id": 0, "time_tick": 480, "number": 64, "value": 127},
                {"track_id": 0, "time_tick": 960, "number": 67, "value": 127},
                {"track_id": 0, "time_tick": 1440, "number": 64, "value": 0},
            ],
        )
    )
    assert not timeline.sustain_at(0.2)
    assert timeline.sustain_at(0.5)
    assert timeline.sustain_at(0.6, track=0)
    assert not timeline.sustain_at(0.6, track=3)
    assert timeline.sustain_at(1.2)  # CC67 (soft pedal) does not release it
    assert not timeline.sustain_at(1.5)
    assert not timeline.sustain_at(3.0)


def test_bar_and_beat_across_tempo_and_metre_changes():
    # 4 quarters at 120 qpm (2 s), then 60 qpm; 3/4 from quarter 8 (t = 6 s).
    timeline = Timeline.from_frames(
        _frames(
            [{"onset_tick": 0, "duration_tick": 480 * 16, "pitch": 60}],
            tempos=[
                {"time_tick": 0, "qpm": 120.0},
                {"time_tick": 4 * TPQ, "qpm": 60.0},
            ],
            time_signatures=[
                {"time_tick": 0, "numerator": 4, "denominator": 4},
                {"time_tick": 8 * TPQ, "numerator": 3, "denominator": 4},
            ],
        )
    )
    assert timeline.quarter_at(1.0) == pytest.approx(2.0)
    assert timeline.quarter_at(3.0) == pytest.approx(5.0)
    assert timeline.bar_beat_at(0.0) == (1, 1)
    assert timeline.bar_beat_at(1.6) == (1, 4)
    assert timeline.bar_beat_at(2.25) == (2, 1)
    assert timeline.bar_beat_at(6.0) == (3, 1)
    assert timeline.bar_beat_at(7.0) == (3, 2)
    assert timeline.bar_beat_at(9.0) == (4, 1)


def test_metre_change_mid_bar_starts_a_new_bar():
    timeline = Timeline.from_frames(
        _frames(
            [{"onset_tick": 0, "duration_tick": 480 * 12, "pitch": 60}],
            time_signatures=[
                {"time_tick": 0, "numerator": 4, "denominator": 4},
                {"time_tick": 6 * TPQ, "numerator": 6, "denominator": 8},
            ],
        )
    )
    # Quarter 6 (t = 3 s) is halfway through bar 2, so 6/8 begins bar 3.
    assert timeline.bar_beat_at(2.9) == (2, 2)
    assert timeline.bar_beat_at(3.0) == (3, 1)
    # 6/8 bars are 3 quarters; eighth-note beats are 0.25 s at 120 qpm.
    assert timeline.bar_beat_at(3.25) == (3, 2)
    assert timeline.bar_beat_at(4.5) == (4, 1)


# -- rendering --------------------------------------------------------------


def test_pressed_key_takes_track_colour():
    frames = _frames(
        [
            {
                "onset_tick": 0,
                "duration_tick": 960,
                "pitch": 60,
                "velocity": 127,
                "track_id": 1,
            },
            {
                "onset_tick": 0,
                "duration_tick": 960,
                "pitch": 61,
                "velocity": 40,
                "track_id": 0,
            },
        ]
    )
    config = RenderConfig(width=640, height=360)
    renderer = PianoRollRenderer(Timeline.from_frames(frames), config)
    surface = renderer.new_surface()
    renderer.draw(surface, 0.5)

    bottom = renderer.keyboard_top + renderer.keyboard_height - 3
    c4 = renderer.layout.key(60)
    assert surface.get_at((round(c4.x + c4.width / 2), bottom))[:3] == track_color(1)
    e4 = renderer.layout.key(64)
    assert surface.get_at((round(e4.x + e4.width / 2), bottom))[:3] == WHITE_KEY
    c_sharp = renderer.layout.key(61)
    black_y = renderer.keyboard_top + 5
    expected = velocity_shade(track_color(0), 40)
    assert (
        surface.get_at((round(c_sharp.x + c_sharp.width / 2), black_y))[:3] == expected
    )


def test_render_without_overlay_or_sustain():
    frames = _frames([{"onset_tick": 0, "duration_tick": 480, "pitch": 30}])
    config = RenderConfig(width=320, height=180, show_overlay=False, show_sustain=False)
    renderer = PianoRollRenderer(Timeline.from_frames(frames), config)
    assert renderer.roll_top == 0
    assert renderer.roll_bottom == renderer.keyboard_top
    renderer.draw(renderer.new_surface(), 0.0)


def test_config_validation():
    with pytest.raises(ValueError):
        RenderConfig(width=641)
    with pytest.raises(ValueError):
        RenderConfig(pitch_low=90, pitch_high=80)


# -- encoding ---------------------------------------------------------------


@needs_ffmpeg
def test_silent_video_has_expected_size_and_length(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 960, "pitch": 60}])  # 1 s
    config = RenderConfig(width=320, height=180, fps=10, audio=False, tail_sec=0.5)
    out = render_piano_roll_video(frames, tmp_path / "silent.mp4", config)

    log, duration, n_frames = _probe(out)
    assert "320x180" in log
    assert "Audio:" not in log
    assert n_frames == 15
    assert duration == pytest.approx(1.5, abs=0.15)


@needs_ffmpeg
def test_audio_is_muxed_and_trimmed_to_video(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 960, "pitch": 60}])
    config = RenderConfig(width=320, height=180, fps=10, tail_sec=0.0)
    calls = []

    def fake_synth(frames_in, config_in):
        calls.append(config_in)
        return _sine_wav(3.0)  # longer than the video: must be cut

    out = render_piano_roll_video(
        frames, tmp_path / "audio.mp4", config, audio_synthesizer=fake_synth
    )
    log, duration, n_frames = _probe(out)
    assert calls == [config]
    assert "Audio:" in log
    assert n_frames == 10
    # 3 s of audio is cut to the 1 s video (AAC frame padding adds a little).
    assert duration == pytest.approx(1.0, abs=0.3)


def test_missing_fluidsynth_raises_clear_error(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 480, "pitch": 60}])
    with pytest.raises(AudioUnavailableError, match="not found"):
        synthesize_wav(
            frames, soundfont=tmp_path / "x.sf2", fluidsynth_bin="no-such-fluidsynth"
        )


@pytest.mark.skipif(
    shutil.which("fluidsynth") is None or RenderConfig().resolve_soundfont() is None,
    reason="fluidsynth or SoundFont not installed",
)
def test_fluidsynth_renders_wav():
    frames = _frames([{"onset_tick": 0, "duration_tick": 960, "pitch": 60}])
    data = synthesize_wav(frames, soundfont=RenderConfig().resolve_soundfont())
    with wave.open(io.BytesIO(data)) as handle:
        seconds = handle.getnframes() / handle.getframerate()
    assert seconds >= 1.0


@needs_ffmpeg
def test_cli_renders_midi_file(tmp_path: Path):
    midi = mido.MidiFile(ticks_per_beat=TPQ)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=TPQ))
    midi_path = tmp_path / "tiny.mid"
    midi.save(midi_path)

    out = tmp_path / "tiny.mp4"
    args = [str(midi_path), "-o", str(out), "--no-audio", "--format", "landscape"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # sandboxed ffmpeg: no remux
        assert cli_main(args) == 0
    assert "1920x1080" in _probe(out)[0]


def test_cli_requires_piece_id_for_dataset_dir(tmp_path: Path):
    assert cli_main([str(tmp_path), "-o", str(tmp_path / "x.mp4")]) == 2


def test_normalize_wav_scales_peak():
    from visualization.audio import normalize_wav

    louder = normalize_wav(_sine_wav(0.1), peak=0.5)
    with wave.open(io.BytesIO(louder)) as handle:
        samples = struct.unpack(f"<{handle.getnframes()}h", handle.readframes(-1))
    assert max(abs(s) for s in samples) == pytest.approx(0.5 * 32767, abs=2)


# -- particles --------------------------------------------------------------


def _particle_renderer(**overrides):
    frames = _frames(
        [{"onset_tick": 480, "duration_tick": 480, "pitch": 60, "velocity": 120}]
    )  # 0.5-1.0 s
    config = RenderConfig(width=640, height=360, **overrides)
    return PianoRollRenderer(Timeline.from_frames(frames), config)


def test_particles_follow_the_press_and_fade_out():
    field = _particle_renderer().particles
    assert field is not None
    count = lambda t: sum(len(batch) for batch in field.at(t))
    assert count(0.49) == 0  # nothing before the press
    bursts = field.bursts(0.6)
    assert len(bursts) > 10
    assert (bursts.y < field.origin_y).any()  # thrown upwards
    key = _particle_renderer().layout.key(60)
    assert abs(float(bursts.x.mean()) - (key.x + key.width / 2)) < 3 * key.width
    assert len(field.embers(0.9)) > 0  # trickle while held
    assert len(field.flashes(0.55)) == 1
    assert count(3.0) == 0  # everything has died out


def test_particles_are_deterministic_per_frame():
    a, b = _particle_renderer(), _particle_renderer()
    surface_a, surface_b = a.new_surface(), b.new_surface()
    a.draw(surface_a, 0.2)  # draw another time first: no hidden state
    a.draw(surface_a, 0.7)
    b.draw(surface_b, 0.7)
    assert pygame.image.tobytes(surface_a, "RGB") == pygame.image.tobytes(
        surface_b, "RGB"
    )


def test_particles_brighten_the_roll_above_the_key():
    with_particles, without = _particle_renderer(), _particle_renderer(particles=False)
    assert without.particles is None
    surfaces = []
    for renderer in (with_particles, without):
        surface = renderer.new_surface()
        renderer.draw(surface, 0.65)
        surfaces.append(surface)
    key = with_particles.layout.key(60)
    region = pygame.Rect(
        round(key.x - 40), with_particles.roll_bottom - 80, round(key.width + 80), 75
    )
    brightness = [
        sum(
            sum(s.get_at((x, y))[:3])
            for x in range(region.left, region.right)
            for y in range(region.top, region.bottom)
        )
        for s in surfaces
    ]
    assert brightness[0] > brightness[1]


def test_zero_density_disables_particles():
    field = _particle_renderer(particle_density=0.0).particles
    assert field is not None and field.at(0.6) == []


# -- clips, presets and composition ------------------------------------------


def test_busiest_window_finds_the_dense_passage():
    sparse = [
        {"onset_tick": i * 960, "duration_tick": 240, "pitch": 60} for i in range(10)
    ]
    dense = [
        {"onset_tick": 9600 + i * 60, "duration_tick": 60, "pitch": 64}
        for i in range(40)
    ]  # 10.0-12.5 s
    timeline = Timeline.from_frames(_frames(sparse + dense))
    start = timeline.busiest_window(1.0)
    assert 10.0 <= start <= 10.3
    assert Timeline.from_frames(_frames(sparse)).busiest_window(100.0) == 0.0


def test_hand_colour_mode_splits_at_middle_c():
    frames = _frames(
        [
            {"onset_tick": 0, "duration_tick": 960, "pitch": 48, "velocity": 127},
            {"onset_tick": 0, "duration_tick": 960, "pitch": 72, "velocity": 127},
        ]
    )
    config = RenderConfig(
        width=640, height=360, color_mode="hand", particles=False, show_overlay=False
    )
    renderer = PianoRollRenderer(Timeline.from_frames(frames), config)
    surface = renderer.new_surface()
    renderer.draw(surface, 0.5)
    y = renderer.keyboard_top + renderer.keyboard_height - 3
    for pitch, color in ((48, config.hand_colors[0]), (72, config.hand_colors[1])):
        key = renderer.layout.key(pitch)
        assert surface.get_at((round(key.x + key.width / 2), y))[:3] == color


def test_caption_panel_replaces_title_bar():
    frames = _frames([{"onset_tick": 0, "duration_tick": 480, "pitch": 60}])
    config = format_config(
        VideoFormat("tiny", 540, 676, 3.0, 0.12), caption="Reference", subcaption="A"
    )
    renderer = PianoRollRenderer(Timeline.from_frames(frames), config)
    assert renderer.caption_mode
    assert renderer.roll_top == round(676 * 0.15)
    renderer.draw(renderer.new_surface(), 0.1)


def test_wrap_text_respects_width():
    font = pygame.font.Font(None, 30)
    lines = wrap_text(font, "one two three four five six seven eight nine ten", 120)
    assert len(lines) > 1
    assert all(font.size(line)[0] <= 120 for line in lines if " " in line)


def test_cut_samples_pads_and_fades():
    samples = np.full((100, 2), 1000, dtype=np.int16)
    cut = cut_samples(samples, 10, start_sec=8.0, n_samples=40, fade_sec=1.0)
    assert cut.shape == (40, 2)
    assert cut[0, 0] == 0  # fade-in starts silent
    assert cut[15, 0] == 1000  # untouched middle
    assert (cut[20:] == 0).all()  # past the end of the source: zero padding


@needs_ffmpeg
def test_render_sequence_concatenates_clips_and_cards(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 1920, "pitch": 60}])  # 2 s
    config = RenderConfig(width=320, height=180, fps=10, fade_sec=0.2)
    lengths = []

    def fake_synth(frames_in, config_in):
        return _sine_wav(2.0)

    out = render_sequence(
        [
            Clip(frames, 0.5, 1.5, caption="A"),
            Clip(frames, 0.0, 0.8),
            Card((TextLine("Reveal", 0.1),), duration_sec=1.0),
        ],
        tmp_path / "sequence.mp4",
        config,
        audio_synthesizer=lambda f, c: lengths.append(c.caption) or fake_synth(f, c),
    )
    log, duration, n_frames = _probe(out)
    assert lengths == ["A", None]  # per-clip caption reaches the clip's config
    assert n_frames == 10 + 8 + 10
    assert "Audio:" in log
    assert duration == pytest.approx(2.8, abs=0.3)


def test_render_sequence_rejects_empty_and_inverted_clips(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 480, "pitch": 60}])
    with pytest.raises(ValueError):
        render_sequence([], tmp_path / "x.mp4")
    with pytest.raises(ValueError):
        render_sequence(
            [Clip(frames, 2.0, 1.0)], tmp_path / "x.mp4", RenderConfig(audio=False)
        )


@needs_ffmpeg
@pytest.mark.filterwarnings("ignore::UserWarning")  # sandboxed ffmpeg: no remux
def test_cli_format_and_busiest_window(tmp_path: Path):
    midi = mido.MidiFile(ticks_per_beat=TPQ)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    for i in range(16):
        track.append(mido.Message("note_on", note=60 + i % 12, velocity=90, time=0))
        track.append(mido.Message("note_off", note=60 + i % 12, velocity=0, time=TPQ))
    path = tmp_path / "scale.mid"
    midi.save(path)
    out = tmp_path / "square.mp4"
    args = [str(path), "-o", str(out), "--format", "square", "--max-duration", "1"]
    with pytest.raises(SystemExit):
        cli_main([*args, "--format", "nope"])
    assert cli_main([*args, "--caption", "Hi", "--no-audio"]) == 0
    log, duration, _ = _probe(out)
    assert "1080x1080" in log
    assert duration == pytest.approx(1.0, abs=0.15)


# -- phantom-bar fixes ----------------------------------------------------------


def test_restruck_key_cuts_the_previous_note():
    timeline = Timeline.from_frames(
        _frames(
            [
                {"onset_tick": 0, "duration_tick": 1920, "pitch": 60},  # 0-2 s
                {"onset_tick": 480, "duration_tick": 480, "pitch": 60},  # 0.5-1 s
                {"onset_tick": 0, "duration_tick": 1920, "pitch": 64},  # other key
            ]
        )
    )
    c4 = sorted(
        (float(o), float(e))
        for o, e, p in zip(timeline.onset, timeline.end, timeline.pitch, strict=True)
        if p == 60
    )
    assert c4 == [(0.0, 0.5), (0.5, 1.0)]
    assert timeline.end[timeline.pitch == 64][0] == pytest.approx(2.0)


def test_duplicate_simultaneous_notes_are_dropped():
    timeline = Timeline.from_frames(
        _frames(
            [
                {"onset_tick": 0, "duration_tick": 480, "pitch": 60},
                {"onset_tick": 0, "duration_tick": 960, "pitch": 60},
            ]
        )
    )
    assert timeline.note_count == 1
    assert float(timeline.end[0]) == pytest.approx(1.0)


def test_black_key_note_has_background_rim():
    frames = _frames(
        [
            {"onset_tick": 0, "duration_tick": 1920, "pitch": 60, "velocity": 127},
            {"onset_tick": 0, "duration_tick": 1920, "pitch": 61, "velocity": 127},
        ]
    )
    config = RenderConfig(width=640, height=360, particles=False, show_overlay=False)
    renderer = PianoRollRenderer(Timeline.from_frames(frames), config)
    surface = renderer.new_surface()
    renderer.draw(surface, -1.0)  # both notes falling, not yet pressed
    c_sharp = renderer.layout.key(61)
    y = renderer.roll_bottom - 60  # inside both bars
    left_edge = round(c_sharp.x) - 1  # where C's bar would show through
    assert surface.get_at((left_edge, y))[:3] == config.background


# -- library exports ------------------------------------------------------------

TINY = VideoFormat("tiny", 320, 400, 2.0, 0.12)


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown format"):
        format_config("cinema")


def test_fit_text_adds_ellipsis():
    font = pygame.font.Font(None, 30)
    text = fit_text(font, "a very long piece title that will not fit", 120)
    assert text.endswith("…")
    assert font.size(text)[0] <= 120


@needs_ffmpeg
def test_export_piece_uses_format_and_busiest_window(tmp_path: Path):
    notes = [
        {"onset_tick": i * 960, "duration_tick": 240, "pitch": 60} for i in range(6)
    ]
    notes += [
        {"onset_tick": 5760 + i * 60, "duration_tick": 60, "pitch": 67}
        for i in range(16)
    ]
    windows = []

    def synth(frames, config):
        windows.append((config.width, config.height))
        return _sine_wav(8.0)

    out = export_piece(
        _frames(notes),
        tmp_path / "piece.mp4",
        TINY,
        max_duration_sec=1.0,
        config=RenderConfig(fps=10),
        audio_synthesizer=synth,
        faststart=False,
    )
    log, duration, n_frames = _probe(out)
    assert windows == [(320, 400)]
    assert "320x400" in log and "Audio:" in log
    assert n_frames == 10
    assert duration == pytest.approx(1.0, abs=0.3)


@needs_ffmpeg
def test_export_comparison_labels_each_clip(tmp_path: Path):
    frames = _frames([{"onset_tick": 0, "duration_tick": 3840, "pitch": 60}])  # 4 s
    seen = []

    def synth(frames_in, config):
        seen.append((config.caption, config.subcaption))
        return _sine_wav(4.0)

    out = export_comparison(
        [
            ComparisonItem(frames, "Reference", "MAESTRO"),
            ComparisonItem(frames, "5-gram", start_sec=1.0),
        ],
        tmp_path / "cmp.mp4",
        TINY,
        clip_sec=1.5,
        outro=Card((TextLine("Thanks"),), duration_sec=0.5),
        config=RenderConfig(fps=10),
        audio_synthesizer=synth,
        faststart=False,
    )
    _, _, n_frames = _probe(out)
    assert seen == [("Reference", "MAESTRO"), ("5-gram", None)]
    assert n_frames == 15 + 15 + 5
    with pytest.raises(ValueError):
        export_comparison([], tmp_path / "x.mp4")


@needs_ffmpeg
def test_faststart_falls_back_when_remux_fails(tmp_path: Path, monkeypatch):
    from visualization import compose

    def broken_remux(src, dst, **kwargs):
        raise compose.FFmpegError("cannot write there")

    monkeypatch.setattr(compose, "remux_faststart", broken_remux)
    frames = _frames([{"onset_tick": 0, "duration_tick": 480, "pitch": 60}])
    with pytest.warns(UserWarning, match="keeping the fragmented mp4"):
        out = export_piece(
            frames,
            tmp_path / "frag.mp4",
            TINY,
            config=RenderConfig(fps=10, audio=False),
        )
    assert out.is_file()
    assert not (tmp_path / "frag.partial.mp4").exists()
    assert _probe(out)[2] > 0

from __future__ import annotations

from miditok.classes import TokSequence
from symusic import Note, Score, Track


def build_miditok_fallback_sequence(score, representation_name: str) -> TokSequence:
    tokens: list[str] = []
    current_time = 0
    for track in score.tracks:
        if representation_name in {"structured", "pertok"}:
            tokens.append(f"Program_{track.program}")
        for note in track.notes:
            delta = note.time - current_time
            tokens.extend(
                [
                    f"TimeShift_{delta}",
                    f"Pitch_{note.pitch}",
                    f"Velocity_{note.velocity}",
                    f"Duration_{note.duration}",
                ]
            )
            current_time = note.time
    ids = list(range(len(tokens)))
    return TokSequence(ids=ids, tokens=tokens)


def parse_miditok_fallback_sequence(tokens: list[str], ticks_per_quarter: int) -> Score:
    """Inverse of :func:`build_miditok_fallback_sequence`.

    The fallback stores absolute ticks, velocities and durations verbatim, so notes
    survive exactly; only the tempo map, controls and pitch bends are gone. Its ids
    are positional, so decoding has to work off the token strings.
    """
    score = Score(ticks_per_quarter)
    track = Track(program=0, is_drum=False)
    score.tracks.append(track)

    current_time = 0
    pending: dict[str, int] = {}
    for token in tokens:
        name, _, value = token.partition("_")
        if name == "Program":
            track.program = int(value)
        elif name == "TimeShift":
            current_time += int(value)
        elif name in {"Pitch", "Velocity", "Duration"}:
            pending[name] = int(value)
            if len(pending) == 3:
                track.notes.append(
                    Note(
                        current_time,
                        pending["Duration"],
                        pending["Pitch"],
                        pending["Velocity"],
                    )
                )
                pending = {}
    return score

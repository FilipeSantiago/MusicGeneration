from __future__ import annotations

from miditok.classes import TokSequence


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

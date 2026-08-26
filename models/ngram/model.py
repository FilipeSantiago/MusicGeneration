import copy
from collections import Counter, defaultdict
import random


class Model:

    def __init__(self, loader, neighbors=None, n=4):
        self.loader = loader
        self.neighbors = neighbors
        self.n = n
        self.models = []

    def train(self):
        x = n = self.n

        BOS = ("<BOS>", "<BOS>", "<BOS>", "<BOS>")
        EOF = ("<EOF>", "<EOF>", "<EOF>", "<EOF>")

        while 2 <= x <= n:
            model = defaultdict(Counter)  # noqa: F821
            for batch in self.loader:
                for song in batch:
                    note_tokens = list(
                        zip(song["pitch"], song["velocity_bin"], song["delta_onset_bin"], song["duration_bin"],)
                    )
                    note_tokens.append(EOF)
                    note_tokens = [BOS] * (x - 1) + note_tokens
                    for i in range(len(note_tokens) - x + 1):
                        context = tuple(note_tokens[i : i + x - 1])
                        target = note_tokens[i + x - 1]
                        model[context][target] += 1

            self.models.append(model)
            x -= 1

    def predict(self, tokens):
        next_token = None
        x = self.n

        while x > 1:
            use_tokens = tokens[(x - 1) * -1 :]
            model = self.models[self.n - x]
            next_token_candidates = model[use_tokens]
            sequences = set(use_tokens)
            i = 0

            while sum(next_token_candidates.values()) < 10 and i < len(use_tokens):
                options = [
                    neighbor
                    for neighbor, _distance in self.neighbors.get(
                        use_tokens[i], ()
                    )
                ]
                if not options:
                    i += 1
                    continue

                sequences = [
                    sequence[:i] + (option,) + sequence[i + 1 :]
                    for sequence in sequences
                    for option in options
                ]

                # print(x, i, len(sequences))
                for sequence in sequences:
                    next_token_candidates += Counter(model[sequence[0],])

                i += 1
            x -= 1
            if sum(next_token_candidates.values()) > 10 or x == 1:
                next_token = random.choices(
                    list(next_token_candidates.keys()), next_token_candidates.values()
                )
                break

        return next_token
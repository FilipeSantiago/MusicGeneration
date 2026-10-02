import random
from collections import Counter, defaultdict
from heapq import heappop, heappush


class Model:

    def __init__(self, loader, neighbors=None, n=4):
        self.loader = loader
        self.neighbors = neighbors
        self.n = n
        self.models = []

    def remi_train(self):
        x = n = self.n

        BOS = -1
        EOF = -1000

        while 2 <= x <= n:
            model = defaultdict(Counter)
            for batch in self.loader:
                for song in batch:
                    note_tokens = song['tokens']
                    note_tokens = [BOS] * (x - 1) + note_tokens + [EOF]
                    for i in range(len(note_tokens) - x + 1):
                        context = tuple(note_tokens[i : i + x - 1])
                        target = note_tokens[i + x - 1]
                        model[context][target] += 1

            self.models.append(model)
            x -= 1
    
    def remi_predict(self, tokens):
        x = self.n

        while x > 1:
            use_tokens = tokens[(x - 1) * -1 :]
            model = self.models[self.n - x]
            next_token_candidates = model.get(use_tokens)

            x -= 1
            if next_token_candidates and (
                next_token_candidates.total() >= 10 or x == 1
            ):
                return random.choices(list(next_token_candidates.keys()), next_token_candidates.values())

        return None

    
    def note_table_train(self):
        x = n = self.n

        BOS = ("<BOS>", "<BOS>", "<BOS>", "<BOS>")
        EOF = ("<EOF>", "<EOF>", "<EOF>", "<EOF>")

        while 2 <= x <= n:
            model = defaultdict(Counter)
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

    def note_table_predict(self, tokens):
        x = self.n

        while x > 1:
            use_tokens = tokens[(x - 1) * -1 :]
            model = self.models[self.n - x]
            next_token_candidates = Counter(model.get(use_tokens, ()))
            candidate_count = next_token_candidates.total()

            if candidate_count < 10 and self.neighbors:
                options_by_position = [
                    [(token, 0.0), *self.neighbors.get(token, ())]
                    for token in use_tokens
                ]
                exact_indices = (0,) * len(use_tokens)
                pending = [(0.0, exact_indices)]
                visited = {exact_indices}

                while pending and candidate_count < 10:
                    distance, indices = heappop(pending)
                    if indices != exact_indices:
                        sequence = tuple(
                            options_by_position[position][option_index][0]
                            for position, option_index in enumerate(indices)
                        )
                        counts = model.get(sequence)
                        if counts:
                            next_token_candidates.update(counts)
                            candidate_count += counts.total()

                    for position, option_index in enumerate(indices):
                        next_index = option_index + 1
                        if next_index >= len(options_by_position[position]):
                            continue
                        next_indices = (
                            indices[:position]
                            + (next_index,)
                            + indices[position + 1 :]
                        )
                        if next_indices in visited:
                            continue
                        visited.add(next_indices)
                        next_distance = (
                            distance
                            - options_by_position[position][option_index][1]
                            + options_by_position[position][next_index][1]
                        )
                        heappush(pending, (next_distance, next_indices))

            x -= 1
            if candidate_count >= 10 or x == 1:
                return random.choices(
                    list(next_token_candidates.keys()), next_token_candidates.values()
                )

        return None

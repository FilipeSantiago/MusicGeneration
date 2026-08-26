import numpy as np


class KNN:

    def __init__(self, n=5, min_count=10):
        self.n_neighbors = n
        self.min_count = min_count

        velocity_scale = 15
        data_scale = 16
        duration_scale = 32

        self.scales = np.array([velocity_scale, data_scale, duration_scale,])

    def get_nearest_neighbors(self, token_counts_by_pitch):

        nearest_neighbors = {}

        for pitch, counts in token_counts_by_pitch.items():
            candidates = self._candidates(counts)
            if not candidates:
                for token in counts:
                    nearest_neighbors[token] = []
                continue

            # Built once per pitch instead of once per token.
            candidate_features = self._features(candidates)
            candidate_index = {token: i for i, token in enumerate(candidates)}

            for token in sorted(counts):
                diff = candidate_features - self._features([token])
                distances = np.sqrt(np.sum(diff ** 2, axis=1))

                # Don't return the token itself
                self_index = candidate_index.get(token)
                if self_index is not None:
                    distances[self_index] = np.inf

                k = min(self.n_neighbors, len(candidates) - (self_index is not None))
                if k <= 0:
                    nearest_neighbors[token] = []
                    continue

                indices = np.argpartition(distances, k - 1,)[:k]

                # argpartition doesn't sort the selected values
                indices = indices[np.argsort(distances[indices])]
                nearest_neighbors[token] = [(candidates[j], float(distances[j]),) for j in indices]

        return nearest_neighbors

    def _candidates(self, counts):
        frequent = sorted(
            token for token, count in counts.items() if count >= self.min_count
        )
        if frequent:
            return frequent

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return sorted(token for token, _count in ranked[: self.n_neighbors + 1])

    def _features(self, tokens):
        return np.array(tokens, dtype=float)[:, 1:] / self.scales

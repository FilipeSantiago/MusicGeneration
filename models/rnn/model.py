"""Recurrent token model used by the REMI trainer."""

from __future__ import annotations

import torch
from torch import nn


class MusicRNN(nn.Module):
    """The embedding, vanilla RNN, and projection from rnn_REMI.ipynb."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        values, hidden = self.rnn(self.embedding(x), hidden)
        return self.output(values), hidden

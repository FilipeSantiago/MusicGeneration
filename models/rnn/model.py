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
        self.rnn2 = nn.RNN(hidden_size, hidden_size*2, batch_first=True)
        self.output = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x, hidden=None):
        first, second = hidden if hidden is not None else (None, None)
        values, first = self.rnn(self.embedding(x), first)
        values, second = self.rnn2(values, second)
        return self.output(values), (first, second)

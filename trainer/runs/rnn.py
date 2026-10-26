"""REMI RNN training, sampling, and checkpoint behavior."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from data_loader.raw_dataloader import RawDataloader
from models.rnn.model import MusicRNN
from trainer.settings import TrainSettings

LOGGER = logging.getLogger(__name__)


class RNNRun:
    checkpoint_name = "model.pt"

    def __init__(self, dataloader: RawDataloader, decoder, settings: TrainSettings):
        self.dataloader = dataloader
        self.decoder = decoder
        self.settings = settings
        self.model: MusicRNN | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed_tokens: list[int] = []
        self.eos_id: int | None = None
        self.forbidden_ids: set[int] = set()
        self.epoch_losses: list[float] = []

    @staticmethod
    def _iter_training_batches(
        song_loader, sequence_length=256, stride=256, batch_size=32, buffer_size=2048
    ):
        """Shuffle notebook next-token windows without retaining all songs."""
        buffer = []
        for song_batch in song_loader:
            for song in song_batch:
                ids = song["tokens"]
                for start in range(0, len(ids) - sequence_length, stride):
                    chunk = ids[start : start + sequence_length + 1]
                    buffer.append((chunk[:-1], chunk[1:]))
                    if len(buffer) >= buffer_size:
                        random.shuffle(buffer)
                        while len(buffer) >= batch_size:
                            yield [buffer.pop() for _ in range(batch_size)]
        random.shuffle(buffer)
        while len(buffer) >= batch_size:
            yield [buffer.pop() for _ in range(batch_size)]

    def train(self) -> None:
        vocab = self.decoder.tokenizer.vocab
        vocab_size = len(self.decoder.tokenizer)
        if not isinstance(vocab, dict) or not vocab_size:
            raise ValueError("The REMI tokenizer must have one nonempty vocabulary")
        model = MusicRNN(vocab_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = self.dataloader.loader(load_music=self.dataloader.load_music)
        LOGGER.info(
            "RNN training on %s: vocab=%s, epochs=%s, training songs=%s",
            self.device,
            vocab_size,
            self.settings.rnn_epochs,
            len(loader.dataset),
        )
        for index in range(min(len(loader.dataset), 32)):
            ids = loader.dataset[index]["tokens"]
            if ids:
                self.seed_tokens.append(int(ids[0]))
        if not self.seed_tokens:
            raise ValueError("REMI training split has no nonempty token sequences")
        self.eos_id = vocab.get("EOS_None")
        self.forbidden_ids = {
            int(value)
            for name, value in vocab.items()
            if name.split("_", 1)[0] in {"PAD", "BOS", "MASK"}
        }
        for epoch in range(self.settings.rnn_epochs):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            progress = tqdm(
                loader,
                total=len(loader) if hasattr(loader, "__len__") else None,
                desc=f"RNN epoch {epoch + 1}/{self.settings.rnn_epochs}",
                unit="song batch",
                disable=None,
            )
            for batch in self._iter_training_batches(progress):
                x = torch.tensor(
                    [sample[0] for sample in batch], dtype=torch.long, device=self.device
                )
                y = torch.tensor(
                    [sample[1] for sample in batch], dtype=torch.long, device=self.device
                )
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                batch_count += 1
                if batch_count % 10 == 0:
                    progress.set_postfix(
                        batch=batch_count,
                        loss=f"{loss.item():.4f}",
                        mean=f"{loss_sum / batch_count:.4f}",
                    )
            if not batch_count:
                raise ValueError("No 256-token REMI training windows were available")
            mean_loss = loss_sum / batch_count
            self.epoch_losses.append(mean_loss)
            LOGGER.info(
                "RNN epoch %s/%s complete: %s batches, mean loss=%.4f",
                epoch + 1,
                self.settings.rnn_epochs,
                batch_count,
                mean_loss,
            )
        self.model = model

    def _trained_model(self) -> MusicRNN:
        if self.model is None:
            raise RuntimeError("Train the RNN before generating or saving")
        return self.model

    @torch.no_grad()
    def generate(self, piece_index: int, attempt: int) -> list[int] | None:
        model = self._trained_model()
        random.seed(f"{self.settings.seed}:{piece_index}:{attempt}")
        generator = torch.Generator(device=self.device).manual_seed(
            self.settings.seed + piece_index * self.settings.max_attempts + attempt
        )
        token = random.choice(self.seed_tokens)
        result = [token]
        hidden = None
        model.eval()
        for _ in range(self.settings.remi_max_tokens - 1):
            x = torch.tensor([[token]], dtype=torch.long, device=self.device)
            logits, hidden = model(x, hidden)
            scores = logits[0, -1].clone()
            if self.forbidden_ids:
                scores[list(self.forbidden_ids)] = -torch.inf
            if not torch.isfinite(scores).any():
                return None
            token = int(
                torch.multinomial(
                    torch.softmax(scores, dim=0), 1, generator=generator
                )
            )
            if token == self.eos_id:
                break
            result.append(token)
        return result

    def save(self, path: Path) -> None:
        model = self._trained_model()
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "vocab_size": model.embedding.num_embeddings,
                "embedding_dim": model.embedding.embedding_dim,
                "hidden_size": model.rnn.hidden_size,
                "seed_tokens": self.seed_tokens,
                "eos_id": self.eos_id,
                "forbidden_ids": sorted(self.forbidden_ids),
            },
            path,
        )
        model.to(self.device)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_loader.base_dataloader import BaseDataloader
from data_loader.music_dataset import MusicDataset


class NNDataloader(BaseDataloader):
    def __init__(self):
        super().__init__()

    def loader(self, batch_size=32) -> DataLoader:
        dataset = MusicDataset(self.train_df, self.representation_root, self.load_music)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.neural_collate,
        )
        return loader

    @staticmethod
    def neural_collate(samples):
        sequences = [
            torch.tensor(sample["tokens"], dtype=torch.long) for sample in samples
        ]

        lengths = torch.tensor([len(sequence) for sequence in sequences])

        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=0,
        )

        return {
            "piece_id": [sample["piece_id"] for sample in samples],
            "segment_id": [sample["segment_id"] for sample in samples],
            "tokens": padded,
            "lengths": lengths,
        }

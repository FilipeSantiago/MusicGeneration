from torch.utils.data import DataLoader

from data_loader.base_dataloader import BaseDataloader
from data_loader.music_dataset import MusicDataset


class RawDataloader(BaseDataloader):
    
    def __init__(self):
        super().__init__()
        self.dataset = None

    def loader(self, batch_size=32, load_music=None, num_workers=2, prefetch_factor=2) -> DataLoader:
        if load_music is None:
            self.dataset = MusicDataset(self.train_df, self.representation_root, self.load_music)
        else:
            self.dataset = MusicDataset(self.train_df, self.representation_root, load_music)
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self.raw_collate,
        )
        return loader

    @staticmethod
    def raw_collate(samples):
        return samples

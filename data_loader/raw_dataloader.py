from torch.utils.data import DataLoader

from data_loader.base_dataloader import BaseDataloader
from data_loader.music_dataset import MusicDataset


class RawDataloader(BaseDataloader):
    
    def __init__(self):
        super().__init__()

    def loader(self, batch_size=32) -> DataLoader:
        dataset = MusicDataset(self.train_df, self.representation_root, self.load_music)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.raw_collate,
        )
        return loader

    @staticmethod
    def raw_collate(samples):
        return samples

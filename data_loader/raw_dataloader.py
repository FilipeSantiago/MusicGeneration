from torch.utils.data import DataLoader

from data_loader.base_dataloader import BaseDataloader
from data_loader.music_dataset import MusicDataset, NoteTableDataset


class RawDataloader(BaseDataloader):
    
    def __init__(self):
        super().__init__()
        self.dataset = None

    def loader(self, batch_size=32, representation="remi", num_workers=2, prefetch_factor=2,
               multiprocessing_context=None) -> DataLoader:
        if representation == "note_table":
            self.dataset = NoteTableDataset(
                self.train_df, self.representation_root,
                self.config.note_table.velocity_bin_size,
            )
        else:
            self.dataset = MusicDataset(self.train_df, self.representation_root)
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self.raw_collate,
            multiprocessing_context=multiprocessing_context,
        )
        return loader

    @staticmethod
    def raw_collate(samples):
        return samples

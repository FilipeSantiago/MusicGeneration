from pathlib import Path

from torch.utils.data import Dataset


class MusicDataset(Dataset):
    def __init__(self, df, representation_root, load_fn):
        self.df = df.reset_index(drop=True)
        self.representation_root = Path(representation_root)
        self.load_fn = load_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self.load_fn(row, self.representation_root)

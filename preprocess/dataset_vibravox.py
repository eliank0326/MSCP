import os
import torch
from torch.utils.data import Dataset

class VibravoxDataset(Dataset):
    def __init__(self, data_dir):
        self.data_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir) if fname.endswith(".pt")
        ]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.data_paths[idx])

        mel = sample["mel"]
        piezo = sample["piezo"]
        chaos = sample["chaos"]
        label = sample["label"]

        return mel, piezo, chaos, label

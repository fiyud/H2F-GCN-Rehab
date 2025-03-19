import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, jcd, labels):
        self.data = data    # Original data
        self.jcd = jcd      # JCD data
        self.labels = labels  # Labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.jcd[idx], self.labels[idx]
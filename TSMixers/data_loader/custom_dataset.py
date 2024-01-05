import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from data_extractor import DataExtractor
import pytorch_lightning as pl



class CustomDataset(Dataset):
    def __init__(self, data_extractor: DataExtractor):
        self.data_extractor = data_extractor

    def __len__(self):
        return len(self.data_extractor)

    def __getitem__(self, index):
        data = self.data_extractor[index]
        inputs = torch.tensor(data["inputs"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.float32)
        return inputs, labels

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size=32):
        super(CustomDataModule, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Initialize datasets
        self.train_dataset = CustomDataset(self.train_df)
        self.val_dataset = CustomDataset(self.val_df)
        self.test_dataset = CustomDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

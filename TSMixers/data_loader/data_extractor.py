import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class DataExtractor:
    def __init__(self, df, row_length=10, tail_length=4):
        self.data = self.extract_contiguous_rows_with_stride(df, row_length, tail_length)


    def extract_contiguous_rows_with_stride(self, df, row_length=10, tail_length=4):
        num_rows = len(df)
        num_chunks = num_rows - row_length + 1

        contiguous_rows = []
        last_four_rows = []
        indices = []

        for i in range(num_chunks):
            chunk = df.iloc[i:i+row_length].values
            contiguous_rows.append(chunk[:row_length-tail_length])
            last_four_rows.append(chunk[-tail_length:])
            indices.append(i)  # Adding the index

        data = {
            "inputs": np.array(contiguous_rows),
            "labels": np.array(last_four_rows),
            "indices": np.array(indices)
        }

        return data

    def __len__(self):
        return len(self.data["indices"])

    def __getitem__(self, index):
        idx = self.data["indices"][index]
        return {
            "inputs": self.data["inputs"][idx],
            "labels": self.data["labels"][idx]
        }
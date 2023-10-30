import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as data
import pytorch_lightning as pl
import tensorflow as tf

DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'

class TSFDataLoader(pl.LightningDataModule):
    def __init__(self, data, batch_size, seq_len, pred_len, feature_type, target='OT'):
        super(TSFDataLoader, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)
        self.scaler = None

    def _read_data(self):
        if not os.path.isdir(LOCAL_CACHE_DIR):
            os.mkdir(LOCAL_CACHE_DIR)

        file_name = self.data + '.csv'
        cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
        if not os.path.isfile(cache_filepath):
            tf.io.gfile.copy(
          os.path.join(DATA_DIR, file_name), cache_filepath, overwrite=True
          )
            # Download the data from the cloud storage
            # (Implement cloud storage download here)


        df_raw = pd.read_csv(cache_filepath)
        df = df_raw.set_index('date')
        # S: univariate-univariate, M: multivariate-multivariate, MS:
    	# multivariate-univariate
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)
        # split train/valid/test
        n = len(df)
        if self.data.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.startswith('ETTh'):
             train_end = 12 * 30 * 24
             val_end = train_end + 4 * 30 * 24
             test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

        # Process and split the data
        # (The same data processing logic as in the original code)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def setup(self, stage=None):
        self._read_data()

    def train_dataloader(self):
        return self._make_dataloader(self.train_df, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_df, shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader(self.test_df, shuffle=False)

    def _split_window(self, data):
        inputs = data[:, :self.seq_len, :]
        labels = data[:, self.seq_len:, self.target_slice]
        inputs.set_shape([None, self.seq_len, None])
        labels.set_shape([None, self.pred_len, None])
        return inputs, labels

    def _make_dataloader(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        dataset = data.TensorDataset(torch.tensor(data))
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import os

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './data_loader/dataset/'

# modularise the code 
class TSFDataLoader:
  """Generate data loader from raw data."""

  def __init__(
      self, data_dir, data ,seq_len, pred_len, feature_type, target='OT'
  ):
    self.data_dir = data_dir
    self.data = data
    #self.batch_size = batch_size
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.feature_type = feature_type
    self.target = target
    #self.target_slice = slice(0, None)

    self.train_df, self.val_df, self.test_df = self._read_data()

  def _read_data(self):
    """Load raw data and split datasets."""

    # copy data from cloud storage if not exists
    LOCAL_CACHE_DIR = self.data_dir
    if not os.path.isdir(LOCAL_CACHE_DIR):
      os.mkdir(LOCAL_CACHE_DIR)

    file_name = self.data + '.csv'
    cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
    if not os.path.isfile(cache_filepath):
      tf.io.gfile.copy(
          os.path.join(DATA_DIR, file_name), cache_filepath, overwrite=True
      )

    df_raw = pd.read_csv(cache_filepath)

    # S: univariate-univariate, M: multivariate-multivariate, MS:
    # multivariate-univariate
    df = df_raw.set_index('date')
    if self.feature_type == 'S':
      df = df[[self.target]]
    # elif self.feature_type == 'MS':
    # target_idx = df.columns.get_loc(self.target)
    # self.target_slice = slice(target_idx, target_idx + 1)

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

    # standardize by training set
    self.scaler = StandardScaler()
    self.scaler.fit(train_df.values)

    def scale_df(df, scaler):
      data = scaler.transform(df.values)
      return pd.DataFrame(data, index=df.index, columns=df.columns)

    train_df = scale_df(train_df, self.scaler)
    val_df = scale_df(val_df, self.scaler)
    test_df = scale_df(test_df, self.scaler)
    #self.n_feature = self.train_df.shape[-1]

    return train_df, val_df, test_df


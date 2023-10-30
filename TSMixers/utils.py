import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResBlock(nn.Module):
    def __init__(self, input_dim, norm_type, activation, dropout, ff_dim):
        super(ResBlock, self).__init__()
        self.norm = nn.LayerNorm(input_dim) if norm_type == 'L' else nn.BatchNorm1d(input_dim)
        self.temporal_linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.feature_linear = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.temporal_linear(x.transpose(1, 2)).transpose(1, 2)
        x = x + res
        x = self.norm(x)
        x = self.feature_linear(x)
        x = x + res
        return x



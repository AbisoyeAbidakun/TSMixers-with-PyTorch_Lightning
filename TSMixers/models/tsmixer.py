import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from rev_in import RevNormLightning
from utils import ResBlock


class TSMixer(pl.LightningModule):
    """
        Constructor for TSMixer.

        Args:
            input_shape (tuple): Input tensor shape.
            pred_len (int): Length of the prediction.
            norm_type (str): Type of normalization ('L' for LayerNorm, 'B' for BatchNorm).
            activation (nn.Module): Activation function.
            n_block (int): Number of ResBlocks in TSMixer.
            dropout (float): Dropout probability.
            ff_dim (int): Feature dimension.
            target_slice (int): Target slice index.
            rev_norm_inst(bool): Flag for reverse normalisation
    """

    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, learning_rate:float = 0.001, rev_norm_inst: bool=False, target_slice: slice=None):
        super(TSMixer, self).__init__()
        self.input_shape = input_shape
        self.pred_len = pred_len
        self.target_slice = target_slice
        self.activation = activation
        self.n_block = n_block
        self.learning_rate = learning_rate

		# cconditions to check if reverse normalization is implmeneted
        self.rev_norm_inst = rev_norm_inst

        #Implement reverse normalisation if condition is set to true
        if self.rev_norm_inst:
              self.rev_norm = RevNormLightning(axis=-2)

        layers = []
        for _ in range(self.n_block):
            layers.append(ResBlock(self.input_shape[-1], norm_type, activation, dropout, ff_dim))
        self.blocks = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Linear(self.input_shape[-1], self.pred_len),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the TSMixer model.

        Args:
            x (torch.Tensor): Input tensor.
            mode (str): 'norm' for training, 'denorm' for inference.

        Returns:
            torch.Tensor: Output tensor after passing through the TSMixer.

        """

        if self.rev_norm_inst:
               x = self.rev_norm(x, mode='norm')
        x = self.blocks(x)
        if self.target_slice:
            x = x[:, :, self.target_slice]
        x = x.transpose(1, 2) # [Batch, Channel, Input Length]
        x = self.output_layer(x) # [Batch, Channel, Output Length]
        x = x.transpose(1, 2) # [Batch, Output Length, Channel]
        outputs = x
        if self.rev_norm_inst:
          outputs = self.rev_norm(x, 'denorm', self.target_slice)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        target_ = y.unsqueeze(dim=-1)
        loss = nn.MSELoss()(y_hat, target_)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        target_ = y.unsqueeze(dim=-1)
        loss = nn.MSELoss()(y_hat, target_)
        #loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

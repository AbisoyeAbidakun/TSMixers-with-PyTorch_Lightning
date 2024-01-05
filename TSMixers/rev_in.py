import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class RevNorm(nn.Module):
    """
    Reversible Instance Normalization.
    """

    def __init__(self, axis, eps=1e-5, affine=True):
        """
        Constructor for RevNorm.

        Args:
            axis (int): Axis or axes along which to compute mean and variance.
            eps (float): Small constant to avoid division by zero.
            affine (bool): If True, learnable affine parameters are applied.
        """
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode, target_slice=None):
        """
        Forward pass of the RevNorm layer.

        Args:
            x (torch.Tensor): Input tensor.
            mode (str): 'norm' for normalization, 'denorm' for denormalization.
            target_slice (int): Target slice index for denormalization.

        Returns:
            torch.Tensor: Normalized or denormalized tensor.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        """
        Calculate mean and standard deviation of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
        """
        self.mean = torch.mean(x, dim=self.axis, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=self.axis, keepdim=True) + self.eps).detach()

    def _normalize(self, x):
        """
        Normalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        """
        Denormalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            target_slice (int): Target slice index for denormalization.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x

class RevNormLightning(pl.LightningModule):
    """
    PyTorch Lightning module for RevNorm.
    """

    def __init__(self, axis, eps=1e-5, affine=True):
        """
        Constructor for RevNormLightning.

        Args:
            axis (int): Axis or axes along which to compute mean and variance.
            eps (float): Small constant to avoid division by zero.
            affine (bool): If True, learnable affine parameters are applied.
        """
        super(RevNormLightning, self).__init__()
        self.revnorm = RevNorm(axis, eps, affine)

    def forward(self, x, mode, target_slice=None):
        """
        Forward pass of the RevNormLightning model.

        Args:
            x (torch.Tensor): Input tensor.
            mode (str): 'norm' for normalization, 'denorm' for denormalization.
            target_slice (int): Target slice index for denormalization.

        Returns:
            torch.Tensor: Normalized or denormalized tensor.
        """
        return self.revnorm(x, mode, target_slice)

    def configure_optimizers(self):
        """
        Configure optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)


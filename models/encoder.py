import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """1D Convolutional Encoder projecting raw waveform into 2D time-frequency-like feature space."""

    def __init__(self, in_channels: int = 1, filters: int = 64, kernel_size: int = 16, stride: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Waveform tensor of shape (B, 1, T).

        Returns:
            Encoded representation of shape (B, filters, L).
        """
        return F.relu(self.conv(x))

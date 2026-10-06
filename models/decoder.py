import torch
import torch.nn as nn


class Decoder(nn.Module):
    """1D Transposed Convolutional Decoder reconstructing waveform from feature space."""

    def __init__(self, in_channels: int = 64, out_channels: int = 1, kernel_size: int = 16, stride: int = 8):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Feature representation of shape (B, in_channels, L).

        Returns:
            Reconstructed waveform tensor of shape (B, 1, T).
        """
        return self.deconv(x)

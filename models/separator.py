from typing import Optional
import torch
import torch.nn as nn

from models.config import ModelConfig
from models.encoder import Encoder
from models.decoder import Decoder
from models.dprnn import DPRNN_Separator


class ORPIT_Model(nn.Module):
    """Recursive Residual Speech Separation (R2S2) model using One-and-Rest

    Permutation Invariant Training (OR-PIT) with a DPRNN backbone.

    Given a mixture waveform of shape (B, T), predicts:
      - Channel 0: Separated target speaker audio of shape (B, T)
      - Channel 1: Residual mixture of all remaining speakers of shape (B, T)
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = ModelConfig(**kwargs)

        self.config = config
        self.encoder = Encoder(
            in_channels=1,
            filters=config.enc_filters,
            kernel_size=config.enc_kernel,
            stride=config.enc_stride,
        )
        self.separator = DPRNN_Separator(
            feat_dim=config.enc_filters,
            hidden=config.dprnn_hidden,
            num_layers=config.dprnn_layers,
            chunk_size=config.chunk_size,
            num_outputs=2,
        )
        self.decoder = Decoder(
            in_channels=config.enc_filters,
            out_channels=1,
            kernel_size=config.enc_kernel,
            stride=config.enc_stride,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Args:

            mixture: Raw mixture waveform tensor of shape (B, T).

        Returns:
            Separated audio tensor of shape (B, 2, T) where:
              out[:, 0, :] is estimated target speaker
              out[:, 1, :] is estimated residual mixture
        """
        B, T = mixture.shape
        mix_enc = self.encoder(mixture.unsqueeze(1))
        sep_enc = self.separator(mix_enc)

        # sep_enc: (B, 2, enc_filters, L) -> reshape to (B * 2, enc_filters, L) for decoder
        out = self.decoder(sep_enc.view(B * 2, self.config.enc_filters, -1)).view(B, 2, -1)
        return out[:, :, :T]


# Alias for readability
R2S2Separator = ORPIT_Model

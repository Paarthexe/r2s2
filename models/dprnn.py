import torch
import torch.nn as nn
import torch.nn.functional as F


class DPRNN_Block(nn.Module):
    """Dual-Path RNN Block comprising intra-chunk (local) and inter-chunk (global)

    bidirectional LSTMs with residual connections and Layer Normalization.
    """

    def __init__(self, feat_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.intra_rnn = nn.LSTM(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(hidden * 2, feat_dim)
        self.intra_ln = nn.LayerNorm(feat_dim)

        self.inter_rnn = nn.LSTM(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.inter_fc = nn.Linear(hidden * 2, feat_dim)
        self.inter_ln = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: 4D tensor of shape (B, N, K, S), where:
               B = batch size
               N = feature dimension (channels)
               K = chunk size (intra-chunk time steps)
               S = number of chunks (inter-chunk time steps)

        Returns:
            Processed 4D tensor of shape (B, N, K, S).
        """
        B, N, K, S = x.shape

        # --- Intra-chunk RNN (local dependencies within each chunk) ---
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        out, _ = self.intra_rnn(intra)
        out = self.intra_fc(out).view(B, S, K, N).permute(0, 3, 2, 1)
        x = self.intra_ln((x + out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # --- Inter-chunk RNN (global context across chunks) ---
        inter = x.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        out, _ = self.inter_rnn(inter)
        out = self.inter_fc(out).view(B, K, S, N).permute(0, 3, 1, 2)
        x = self.inter_ln((x + out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x


class DPRNN_Separator(nn.Module):
    """Dual-Path RNN Separator with 50% overlap chunking, deep DPRNN blocks,

    and vectorized overlap-add reconstruction via F.fold.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        hidden: int = 128,
        num_layers: int = 6,
        chunk_size: int = 100,
        num_outputs: int = 2,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.chunk_size = chunk_size
        self.num_outputs = num_outputs

        self.norm = nn.GroupNorm(1, feat_dim)
        self.input_proj = nn.Conv1d(feat_dim, feat_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [DPRNN_Block(feat_dim, hidden) for _ in range(num_layers)]
        )
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(feat_dim, num_outputs * feat_dim, kernel_size=1),
        )

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """Args:

            enc: Encoded representation of shape (B, N, L).

        Returns:
            Masked encoder features for each output source: shape (B, num_outputs, N, L).
        """
        B, N, L = enc.shape
        x = self.input_proj(self.norm(enc))

        chunk = self.chunk_size
        hop = chunk // 2
        pad_len = (hop - (L % hop)) % hop

        if pad_len:
            x = F.pad(x, (0, pad_len))
        L_pad = x.shape[2]

        # Vectorized chunk extraction: [B, N, L_pad] -> [B, N, num_chunks, chunk] -> [B, N, chunk, num_chunks]
        chunks = x.unfold(2, chunk, hop).permute(0, 1, 3, 2).contiguous()

        for block in self.blocks:
            chunks = block(chunks)

        # masks: [B, num_outputs * N, chunk, num_chunks] -> [B, num_outputs, N, chunk, num_chunks]
        masks = torch.sigmoid(self.mask_net(chunks))
        _, _, K, S = masks.shape
        masks = masks.view(B, self.num_outputs, N, K, S)

        # Vectorized overlap-add via F.fold
        masks_r = masks.reshape(B * self.num_outputs, N * K, S)
        out = F.fold(
            masks_r,
            output_size=(1, L_pad),
            kernel_size=(1, K),
            stride=(1, hop),
        )
        count = F.fold(
            torch.ones_like(masks_r),
            output_size=(1, L_pad),
            kernel_size=(1, K),
            stride=(1, hop),
        )
        out = (out / count.clamp(min=1.0)).view(B, self.num_outputs, N, L_pad)[:, :, :, :L]

        return out * enc.unsqueeze(1)

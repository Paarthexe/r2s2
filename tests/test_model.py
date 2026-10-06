import os
import torch
import numpy as np
import pytest

from models.config import ModelConfig
from models.encoder import Encoder
from models.decoder import Decoder
from models.dprnn import DPRNN_Block, DPRNN_Separator
from models.separator import ORPIT_Model
from models.losses import si_sdr, multi_res_stft_loss, pit_loss


def test_model_config_json(tmp_path):
    cfg_file = str(tmp_path / "test_model_config.json")
    cfg = ModelConfig(enc_filters=32, dprnn_hidden=64, dprnn_layers=2)
    cfg.to_json(cfg_file)

    loaded = ModelConfig.from_json(cfg_file)
    assert loaded.enc_filters == 32
    assert loaded.dprnn_hidden == 64
    assert loaded.dprnn_layers == 2
    assert loaded.chunk_size == 100


def test_encoder_decoder():
    B, T = 2, 16000
    enc = Encoder(in_channels=1, filters=32, kernel_size=16, stride=8)
    dec = Decoder(in_channels=32, out_channels=1, kernel_size=16, stride=8)

    x = torch.randn(B, 1, T)
    feat = enc(x)
    # Stride 8 -> feat length is approximately T // 8
    assert feat.shape[0] == B
    assert feat.shape[1] == 32

    rec = dec(feat)
    assert rec.shape[0] == B
    assert rec.shape[1] == 1


def test_dprnn_block():
    B, N, K, S = 2, 16, 20, 10
    block = DPRNN_Block(feat_dim=N, hidden=32)
    x = torch.randn(B, N, K, S)
    out = block(x)
    assert out.shape == (B, N, K, S)


def test_dprnn_separator():
    B, N, L = 2, 32, 250
    sep = DPRNN_Separator(feat_dim=N, hidden=32, num_layers=2, chunk_size=50, num_outputs=2)
    enc = torch.randn(B, N, L)
    out = sep(enc)
    # Expect (B, 2, N, L)
    assert out.shape == (B, 2, N, L)


def test_orpit_model_forward():
    B, T = 2, 8000
    cfg = ModelConfig(enc_filters=32, dprnn_hidden=32, dprnn_layers=2, chunk_size=50)
    model = ORPIT_Model(config=cfg)

    mix = torch.randn(B, T)
    out = model(mix)
    # Output must be (B, 2, T)
    assert out.shape == (B, 2, T)


def test_si_sdr():
    # If est == ref, SI-SDR should be very large (positive dB)
    ref = torch.randn(2, 4000)
    est = ref.clone()
    score = si_sdr(est, ref)
    assert torch.all(score > 30.0)

    # Inverted signal (negative) should have finite score
    score_inv = si_sdr(-ref, ref)
    assert torch.all(score_inv > 30.0)  # scale invariant


def test_stft_loss():
    ref = torch.randn(2, 4000)
    est = ref.clone()
    loss = multi_res_stft_loss(est, ref, fft_sizes=[512, 1024])
    assert loss.item() < 1e-4


def test_pit_loss():
    B, T = 2, 4000
    target_ref = torch.randn(B, T)
    residual_ref = torch.randn(B, T)
    mixture = target_ref + residual_ref

    # Perfect prediction in channel 0 (target) and channel 1 (residual)
    out = torch.stack([target_ref, residual_ref], dim=1)
    loss, metrics, target_est, residual_est = pit_loss(
        out, target_ref, residual_ref, mixture, stft_weight=0.01, consistency_weight=0.1
    )

    assert loss is not None
    assert metrics["sisdr"] > 30.0
    assert metrics["consistency"] < 1e-5
    assert target_est.shape == (B, T)
    assert residual_est.shape == (B, T)

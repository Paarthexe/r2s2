from typing import Dict, Tuple, List, Union
import torch
import torch.nn.functional as F

_hann_windows: Dict[Tuple[int, str], torch.Tensor] = {}


def get_hann_window(size: int, device: torch.device) -> torch.Tensor:
    """Caches and returns a Hann window for STFT computation."""
    key = (size, str(device))
    if key not in _hann_windows or _hann_windows[key].device != device:
        _hann_windows[key] = torch.hann_window(size, device=device)
    return _hann_windows[key]


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        est: Estimated audio waveform of shape (..., T).
        ref: Target reference audio waveform of shape (..., T).
        eps: Numerical stability constant.

    Returns:
        SI-SDR in decibels of shape (...).
    """
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    alpha = (ref_zm * est_zm).sum(-1, keepdim=True) / (ref_zm.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref_zm
    noise = est_zm - target

    return 10.0 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


def stft_loss_single(est: torch.Tensor, ref: torch.Tensor, fft_size: int) -> torch.Tensor:
    """Computes spectral convergence and log magnitude STFT loss for a single FFT size."""
    hop = fft_size // 4
    win = fft_size // 2
    window = get_hann_window(win, est.device)

    est_mag = torch.stft(
        est, fft_size, hop_length=hop, win_length=win, window=window, return_complex=True
    ).abs().clamp(min=1e-8)
    ref_mag = torch.stft(
        ref, fft_size, hop_length=hop, win_length=win, window=window, return_complex=True
    ).abs().clamp(min=1e-8)

    sc = torch.norm(ref_mag - est_mag, "fro") / torch.norm(ref_mag, "fro")
    lm = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
    return sc + lm


def multi_res_stft_loss(
    est: torch.Tensor, ref: torch.Tensor, fft_sizes: List[int] = [512, 1024, 2048]
) -> torch.Tensor:
    """Multi-resolution STFT loss over multiple FFT window sizes (default: 512, 1024, 2048)."""
    est_f = est.float().reshape(-1, est.shape[-1])
    ref_f = ref.float().reshape(-1, ref.shape[-1])
    return sum(stft_loss_single(est_f, ref_f, fft) for fft in fft_sizes) / len(fft_sizes)


def pit_loss(
    out: torch.Tensor,
    target_ref: torch.Tensor,
    residual_ref: torch.Tensor,
    mixture: torch.Tensor,
    stft_weight: float = 0.0,
    consistency_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """One-and-Rest Permutation Invariant Training (OR-PIT) loss with

    Scale-Invariant SDR, multi-resolution STFT, and mixture consistency.

    Args:
        out: Model output tensor of shape (B, 2, T) (estimated target and residual channels).
        target_ref: Ground truth target speaker audio of shape (B, T).
        residual_ref: Ground truth sum of remaining speakers of shape (B, T).
        mixture: Mixture waveform of shape (B, T).
        stft_weight: Weight lambda for multi-resolution STFT loss.
        consistency_weight: Weight for L1 consistency loss: |target + residual - mixture|.

    Returns:
        loss: Scalar training loss tensor.
        metrics: Dictionary with SI-SDR, STFT, and consistency values.
        target_est: Optimal assigned target estimate of shape (B, T).
        residual_est: Optimal assigned residual estimate of shape (B, T).
    """
    out1, out2 = out[:, 0], out[:, 1]

    sisdr_a = si_sdr(out1, target_ref) + si_sdr(out2, residual_ref)
    sisdr_b = si_sdr(out1, residual_ref) + si_sdr(out2, target_ref)

    use_a = sisdr_a >= sisdr_b
    target_est = torch.where(use_a.unsqueeze(-1), out1, out2)
    residual_est = torch.where(use_a.unsqueeze(-1), out2, out1)
    best_sisdr = torch.where(use_a, sisdr_a, sisdr_b) / 2.0

    if stft_weight > 0:
        stft = multi_res_stft_loss(target_est, target_ref) + multi_res_stft_loss(
            residual_est, residual_ref
        )
    else:
        stft = 0.0

    consistency = F.l1_loss(target_est + residual_est, mixture)

    loss = (-best_sisdr + stft_weight * stft).mean() + consistency_weight * consistency

    stft_val = float(stft.item() if isinstance(stft, torch.Tensor) else stft)
    metrics = {
        "sisdr": float(best_sisdr.mean().item()),
        "stft": stft_val,
        "consistency": float(consistency.item()),
    }
    return loss, metrics, target_est, residual_est

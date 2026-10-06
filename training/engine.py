import os
import time
import random
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.separator import ORPIT_Model
from models.losses import pit_loss
from training.config import TrainConfig
from training.dataset import get_loaders, get_curriculum_weights


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    is_train: bool,
    stft_weight: float,
    config: TrainConfig,
    epoch: int = 0,
    global_step: int = 0,
) -> Tuple[float, Dict[str, float], int, int]:
    """Runs a single training or validation epoch."""
    model.train() if is_train else model.eval()

    n = len(loader)
    if n == 0:
        return 0.0, {"sisdr": 0.0, "stft": 0.0, "consistency": 0.0}, global_step, 0

    device = config.resolved_device
    device_type = "cuda" if "cuda" in device else "cpu"

    total_loss = 0.0
    total_metrics = {"sisdr": 0.0, "stft": 0.0, "consistency": 0.0}
    unroll_batches = 0

    desc = f"Epoch {epoch:02d} [{'Train' if is_train else 'Val'}]"
    with tqdm(loader, desc=desc, leave=False) as pbar:
        for batch in pbar:
            if is_train:
                mix, target_ref, residual_ref, sources, target_idx, num_speakers = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                ]
            else:
                mix, target_ref, residual_ref = [b.to(device) for b in batch]

            if is_train and optimizer is not None:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                with autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    out = model(mix)
                    loss, metrics, target_est, residual_est = pit_loss(
                        out=out,
                        target_ref=target_ref,
                        residual_ref=residual_ref,
                        mixture=mix,
                        stft_weight=stft_weight,
                        consistency_weight=config.consistency_weight,
                    )

                    # --- Recursive unrolling during training ---
                    if (
                        is_train
                        and epoch >= config.unroll_start_epoch
                        and random.random() < config.unroll_prob
                    ):
                        unroll_batches += 1
                        mix2 = residual_est.detach()

                        B = mix.shape[0]
                        target_ref2 = torch.zeros_like(target_ref)
                        residual_ref2 = torch.zeros_like(residual_ref)

                        for b in range(B):
                            n_spk = int(num_speakers[b].item())
                            avail_idx = [i for i in range(n_spk) if i != int(target_idx[b].item())]
                            if len(avail_idx) > 0:
                                new_target_idx = random.choice(avail_idx)
                                target_ref2[b] = sources[b, new_target_idx]

                                rem_idx = [i for i in avail_idx if i != new_target_idx]
                                if len(rem_idx) > 0:
                                    residual_ref2[b] = sources[b, rem_idx].sum(dim=0)

                        out2 = model(mix2)
                        loss2, metrics2, _, _ = pit_loss(
                            out=out2,
                            target_ref=target_ref2,
                            residual_ref=residual_ref2,
                            mixture=mix2,
                            stft_weight=stft_weight,
                            consistency_weight=config.consistency_weight,
                        )
                        loss = loss + 0.5 * loss2
                        for k in metrics:
                            metrics[k] = metrics[k] + 0.5 * metrics2[k]

            if is_train and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                if global_step <= config.warmup_steps:
                    current_lr = config.lr * (global_step / config.warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr

            total_loss += loss.item()
            for k in metrics:
                total_metrics[k] += metrics[k]

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                sisdr=f"{metrics['sisdr']:.2f}",
                stft=f"{metrics['stft']:.3f}",
                cons=f"{metrics['consistency']:.3f}",
            )

    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return avg_loss, avg_metrics, global_step, unroll_batches


def train(config: TrainConfig) -> nn.Module:
    """Main training loop for R2S2 speech separator."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = config.resolved_device
    device_type = "cuda" if "cuda" in device else "cpu"

    print("=" * 60)
    print("R2S2 Model Training")
    print(f"Device        : {device}")
    print(f"Epochs        : {config.epochs}")
    print(f"Batch Size    : {config.batch_size}")
    print(f"Learning Rate : {config.lr}")
    print(f"Dataset Dir   : {config.dataset_dir}")
    print("=" * 60)

    train_dl, val_dl, train_ds = get_loaders(config)

    model = ORPIT_Model(config=config.model).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params / 1e6:.2f}M")
    print(f"Train samples    : {len(train_ds)} ({len(train_dl)} batches)")
    print(f"Val batches      : {len(val_dl)}")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler(device_type)

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    if config.resume_checkpoint and os.path.exists(config.resume_checkpoint):
        ckpt = torch.load(config.resume_checkpoint, map_location=device)
        state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed checkpoint from epoch {ckpt['epoch']}")

    if start_epoch > config.epochs:
        print(f"Already at epoch {config.epochs} - nothing to train.")
        return model

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()

        # Curriculum weighting
        if len(train_ds.speaker_counts) > 0:
            weights = get_curriculum_weights(epoch, config.epochs, train_ds.speaker_counts)
            train_dl.sampler.set_weights(weights)

        # STFT warmup schedule
        if epoch <= config.stft_warmup_start:
            stft_weight = config.stft_lambda_floor
        elif epoch <= config.stft_warmup_end:
            stft_weight = config.stft_lambda_floor + (
                config.stft_lambda_max - config.stft_lambda_floor
            ) * (epoch - config.stft_warmup_start) / (
                config.stft_warmup_end - config.stft_warmup_start
            )
        else:
            stft_weight = config.stft_lambda_max

        train_loss, train_metrics, global_step, unroll_batches = run_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            scaler=scaler,
            is_train=True,
            stft_weight=stft_weight,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )

        run_val = (
            (epoch % config.val_interval == 0)
            or (epoch == config.epochs)
            or (epoch == 1)
        )

        val_sisdr = 0.0
        val_loss = 0.0
        val_metrics = {}
        if run_val and len(val_dl) > 0:
            val_loss, val_metrics, _, _ = run_epoch(
                model=model,
                loader=val_dl,
                optimizer=None,
                scaler=scaler,
                is_train=False,
                stft_weight=stft_weight,
                config=config,
                epoch=epoch,
                global_step=global_step,
            )
            val_sisdr = val_metrics["sisdr"]
            if global_step > config.warmup_steps:
                scheduler.step(-val_sisdr)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        unroll_frac = unroll_batches / len(train_dl) if len(train_dl) > 0 else 0
        unroll_str = f" | Unroll {unroll_frac:.0%}" if unroll_batches > 0 else ""

        print(
            f"Epoch {epoch:03d}/{config.epochs} | {elapsed:.0f}s | stft_w={stft_weight:.3f}{unroll_str} | LR {lr:.2e}\n"
            f"  Train: Loss {train_loss:.4f} | SI-SDR {train_metrics['sisdr']:.2f} | STFT {train_metrics['stft']:.3f} | Cons {train_metrics['consistency']:.3f}"
        )
        if run_val and len(val_dl) > 0:
            print(
                f"  Val  : Loss {val_loss:.4f} | SI-SDR {val_metrics['sisdr']:.2f} | STFT {val_metrics['stft']:.3f} | Cons {val_metrics['consistency']:.3f}"
            )
        else:
            print("  Val  : Skipped")

        # Save last checkpoint
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        last_ckpt = {
            "epoch": epoch,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "config": config.to_json if hasattr(config, "to_json") else None,
        }
        torch.save(last_ckpt, os.path.join(config.checkpoint_dir, "last.pt"))

        if run_val and -val_sisdr < best_val_loss:
            best_val_loss = -val_sisdr
            best_ckpt = dict(last_ckpt)
            best_ckpt["best_val_loss"] = best_val_loss
            torch.save(best_ckpt, os.path.join(config.checkpoint_dir, "best.pt"))
            print(f"  --> Saved new best checkpoint (Score: {-best_val_loss:.2f} dB SI-SDR)")

    return model

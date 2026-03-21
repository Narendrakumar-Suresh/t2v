"""
train.py — Video Generation Training
=====================================
Usage:
  python train.py --model a   # TF + TF
  python train.py --model b   # Mamba2 + TF
  python train.py --model c   # Mamba2 + Mamba2
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from safetensors.torch import save_file

# ── ROCm environment flags ────────────────────────────────────────────
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"   # better BLAS on MI300X
os.environ["HIP_FORCE_DEV_KERNARG"]       = "1"   # kernel performance
os.environ["GPU_MAX_HW_QUEUES"]           = "2"   # limit HIP streams

# ── Config ────────────────────────────────────────────────────────────
class Config:
    # model
    dim        = 512
    n_layers   = 12
    n_heads    = 8
    latent_ch  = 16      # cosmos DV4x8x8 channels
    patch_size = 4
    short_k    = 64
    head_layers= 8
    noise_std  = 0.1
    dropout    = 0.0

    # training
    batch_size = 32
    lr         = 3e-4
    weight_decay = 0.1
    grad_clip  = 1.0
    max_steps  = 300_000
    warmup_steps = 1_000

    # logging
    log_every  = 100
    save_every = 25_000

    # data
    use_merged   = True   # False = stream from fal (before augment.py runs)
    max_text_len = 128
    num_workers  = 4

    # HF
    HF_TOKEN = os.environ.get("HF_TOKEN")


# ── Model selector ────────────────────────────────────────────────────
def get_model(model_type: str, cfg: Config):
    if model_type == "a":
        from models.model_a import ModelA
        return ModelA(
            dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
            short_k=cfg.short_k, head_layers=cfg.head_layers,
            noise_std=cfg.noise_std, dropout=cfg.dropout,
        )
    elif model_type == "b":
        from models.model_b import ModelB
        return ModelB(
            dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
            short_k=cfg.short_k, head_layers=cfg.head_layers,
            noise_std=cfg.noise_std, dropout=cfg.dropout,
        )
    elif model_type == "c":
        from models.model_c import ModelC
        return ModelC(
            dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
            short_k=cfg.short_k, head_layers=cfg.head_layers,
            noise_std=cfg.noise_std, dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Consistency loss ──────────────────────────────────────────────────
def consistency_loss(pred_1, pred_2):
    """
    Both predictions should denoise to same x0.
    Stop gradient on pred_2 — only update pred_1 network.
    """
    return F.mse_loss(pred_1, pred_2.detach())


def get_noisy_pair(x0: torch.Tensor):
    """
    Sample two adjacent noise levels for same clean latent.
    Returns noisy versions + noise levels.
    """
    B, C = x0.shape
    t1 = torch.rand(B, 1, device=x0.device)
    t2 = t1 + 0.05                              # adjacent noise levels
    t2 = t2.clamp(0, 1)

    x_t1 = x0 + torch.randn_like(x0) * t1
    x_t2 = x0 + torch.randn_like(x0) * t2

    return x_t1, x_t2, t1, t2


# ── LR scheduler (cosine with warmup) ────────────────────────────────
def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


# ── Save checkpoint ───────────────────────────────────────────────────
def save_checkpoint(model, step: int, model_type: str, cfg: Config):
    save_dir = f"./checkpoints/model_{model_type}"
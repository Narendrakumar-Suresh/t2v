"""
train_a.py — Model A: Transformer + Transformer — FIXED
=========================================================
497M | dim=768 | 22 layers | 12 heads
~11.8hrs MI300X | ~$23

Fixes:
1. loss = consistency_loss + reconstruction anchor (no collapse)
2. token_mask passed to model (no padding dilution)
3. x0 is current frame — separate from past input (no data leakage)
"""

import os, math, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from safetensors.torch import save_file
from dotenv import load_dotenv

load_dotenv()

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"]       = "1"
os.environ["GPU_MAX_HW_QUEUES"]           = "2"


class Config:
    model_type   = "a"
    dim          = 768
    n_layers     = 22
    n_heads      = 12
    latent_ch    = 16
    patch_size   = 4
    short_k      = 64
    head_layers  = 8
    noise_std    = 0.1
    dropout      = 0.0
    vocab_size   = 32128
    batch_size   = 32
    lr           = 3e-4
    weight_decay = 0.1
    grad_clip    = 1.0
    max_steps    = 300_000
    warmup_steps = 1_000
    log_every    = 100
    save_every   = 25_000
    use_merged   = True
    max_text_len = 128
    num_workers  = 4
    recon_weight = 0.5   # weight for reconstruction anchor loss
    HF_TOKEN     = os.environ.get("HF_TOKEN")


def get_model(cfg):
    from models.model_a import ModelA
    return ModelA(
        dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
        short_k=cfg.short_k, head_layers=cfg.head_layers,
        noise_std=cfg.noise_std, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size,
    )


def get_noisy_pair(x0: torch.Tensor):
    t1   = torch.rand(x0.shape[0], 1, device=x0.device)
    t2   = (t1 + 0.05).clamp(0.0, 1.0)
    # spatial broadcasting for noise level
    x_t1 = x0 + torch.randn_like(x0) * t1.view(-1, 1, 1, 1)
    x_t2 = x0 + torch.randn_like(x0) * t2.view(-1, 1, 1, 1)
    return x_t1, x_t2, t1, t2


def compute_loss(model, latents, tokens, token_mask, x0, cfg):
    """
    Two-part loss:
    1. Consistency: pred at t2 (noisier) ≈ pred at t1 (cleaner, stopgrad)
    2. Reconstruction: pred at t1 ≈ actual x0 (anchor — prevents collapse)
    """
    x_t1, x_t2, t1, t2 = get_noisy_pair(x0)

    pred_1 = model(latents, tokens, x_t1, t1, token_mask)
    pred_2 = model(latents, tokens, x_t2, t2, token_mask)

    # consistency loss — noisier should predict same x0 as cleaner
    loss_cons  = F.mse_loss(pred_2, pred_1.detach())

    # reconstruction anchor — cleaner should match actual x0
    loss_recon = F.mse_loss(pred_1, x0)

    return loss_cons + cfg.recon_weight * loss_recon, loss_cons, loss_recon


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * p))


def save_checkpoint(model, step, cfg):
    d    = f"./checkpoints/model_{cfg.model_type}"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/step_{step:07d}.safetensors"
    s    = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") \
           else model.state_dict()
    save_file({k: v.cpu() for k, v in s.items()}, path)
    print(f"  ✓ {path}")
    if cfg.HF_TOKEN:
        api = HfApi(token=cfg.HF_TOKEN)
        rid = f"entropyspace/videogen-model-{cfg.model_type}"
        api.create_repo(repo_id=rid, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=f"step_{step:07d}.safetensors",
                        repo_id=rid, repo_type="model")
        print(f"  ✓ → {rid}")


def train():
    cfg       = Config()
    use_accel = torch.accelerator.is_available()
    device    = torch.accelerator.current_accelerator() if use_accel \
                else torch.device("cpu")

    model = get_model(cfg).to(device)
    if use_accel:
        model = torch.compile(model)
    n_p = sum(p.numel() for p in model.parameters())

    print("=" * 55)
    print(f"  Model A — Transformer + Transformer")
    print(f"  Params:  {n_p/1e6:.1f}M")
    print(f"  Device:  {device}")
    print("=" * 55)

    optimizer = torch.optim.AdamW(model.parameters(),
                  lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    from data.dataloader import VideoLatentDataset
    loader = DataLoader(
        VideoLatentDataset(use_merged=cfg.use_merged,
                           max_text_len=cfg.max_text_len),
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=use_accel, persistent_workers=cfg.num_workers > 0,
    )

    model.train()
    step, total_loss = 0, 0.0
    print(f"  Training {cfg.max_steps:,} steps\n")

    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps:
                break

            latents    = batch["latents"].to(device)     # [B, C, T-1, H, W]
            tokens     = batch["tokens"].to(device)      # [B, seq]
            token_mask = batch["token_mask"].to(device)  # [B, seq]
            x0         = batch["x0"].to(device)          # [B, C] current frame

            lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            loss, lc, lr_ = compute_loss(model, latents, tokens,
                                         token_mask, x0, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % cfg.log_every == 0:
                avg        = total_loss / cfg.log_every
                total_loss = 0.0
                mem        = torch.cuda.memory_allocated()/1e9 if use_accel else 0.0
                print(f"step {step:>7,} | loss {avg:.5f} "
                      f"| cons {lc:.4f} | recon {lr_:.4f} "
                      f"| lr {lr:.2e} | mem {mem:.1f}GB")

            if step % cfg.save_every == 0:
                save_checkpoint(model, step, cfg)

    save_checkpoint(model, step, cfg)
    print(f"\n✅ Done — {step:,} steps")


def smoke_test():
    class C:
        model_type="a"; dim=64; n_layers=2; n_heads=4; latent_ch=16
        patch_size=4; short_k=8; head_layers=2; noise_std=0.1
        dropout=0.0; vocab_size=32128; recon_weight=0.5
    cfg   = C()
    model = get_model(cfg)

    B, C, H, W = 2, 16, 16, 16
    # past context (T-1=15 frames), target is separate
    latents    = torch.randn(B, C, 15, H, W)
    tokens     = torch.randint(0, 32128, (B, 32))
    token_mask = torch.ones(B, 32, dtype=torch.long)
    x0         = torch.randn(B, C, H, W)   # spatial target frame

    opt = torch.optim.AdamW(model.parameters())
    opt.zero_grad()
    loss, lc, lr_ = compute_loss(model, latents, tokens, token_mask, x0, cfg)
    loss.backward()
    opt.step()

    print(f"  loss={loss.item():.5f} cons={lc:.5f} recon={lr_:.5f}")
    print(f"✅ Model A smoke test passed!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    smoke_test() if args.smoke else train()
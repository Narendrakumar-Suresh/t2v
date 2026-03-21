"""
train_b.py — Model B: Mamba2 + Transformer
============================================
500M params | dim=768 | 28 layers | state_dim=128 | expand=2
~4.1hrs on MI300X | ~$8

Usage:
  python train_b.py
  python train_b.py --smoke
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


# ── Config — tuned for ~500M ──────────────────────────────────────────
class Config:
    model_type  = "b"
    dim         = 768    # ← tuned for 500M
    n_layers    = 28     # ← tuned for 500M
    n_heads     = 12     # for short TF ctx
    state_dim   = 128    # ← Mamba2 state (larger to hit 500M)
    expand      = 2      # Mamba2 expansion
    latent_ch   = 16
    patch_size  = 4
    short_k     = 64
    head_layers = 8
    noise_std   = 0.1
    dropout     = 0.0
    vocab_size  = 32128

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
    HF_TOKEN     = os.environ.get("HF_TOKEN")


def get_model(cfg):
    from models.model_b import ModelB
    return ModelB(
        dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
        short_k=cfg.short_k, head_layers=cfg.head_layers,
        noise_std=cfg.noise_std, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size,
    )

def consistency_loss(p1, p2): return F.mse_loss(p1, p2.detach())

def get_noisy_pair(x0):
    t1 = torch.rand(x0.shape[0], 1, device=x0.device)
    t2 = (t1 + 0.05).clamp(0, 1)
    return x0+torch.randn_like(x0)*t1, x0+torch.randn_like(x0)*t2, t1, t2

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    p = (step-cfg.warmup_steps) / max(1, cfg.max_steps-cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * p))

def save_checkpoint(model, step, cfg):
    d = f"./checkpoints/model_{cfg.model_type}"
    os.makedirs(d, exist_ok=True)
    path = f"{d}/step_{step:07d}.safetensors"
    s = model._orig_mod.state_dict() if hasattr(model,"_orig_mod") else model.state_dict()
    save_file({k: v.cpu() for k,v in s.items()}, path)
    print(f"  ✓ {path}")
    if cfg.HF_TOKEN:
        api = HfApi(token=cfg.HF_TOKEN)
        rid = f"entropyspace/videogen-model-{cfg.model_type}"
        api.create_repo(repo_id=rid, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=f"step_{step:07d}.safetensors",
                        repo_id=rid, repo_type="model")
        print(f"  ✓ uploaded → {rid}")

def train():
    cfg       = Config()
    use_accel = torch.accelerator.is_available()
    device    = torch.accelerator.current_accelerator() if use_accel \
                else torch.device("cpu")

    model = get_model(cfg).to(device)
    if use_accel: model = torch.compile(model)
    n_p = sum(p.numel() for p in model.parameters())

    print("=" * 55)
    print(f"  Model B — Mamba2 + Transformer")
    print(f"  Params:  {n_p/1e6:.1f}M")
    print(f"  Device:  {device}")
    print("=" * 55)

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9,0.95))
    from data.dataloader import VideoLatentDataset
    loader = DataLoader(VideoLatentDataset(use_merged=cfg.use_merged,
                                           max_text_len=cfg.max_text_len),
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        pin_memory=use_accel, persistent_workers=cfg.num_workers>0)

    model.train()
    step, total_loss = 0, 0.0
    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps: break
            latents = batch["latents"].to(device)
            tokens  = batch["tokens"].to(device)
            x0      = batch["x0"].to(device)
            x_t1,x_t2,t1,t2 = get_noisy_pair(x0)
            lr = get_lr(step, cfg)
            for g in optimizer.param_groups: g["lr"] = lr
            optimizer.zero_grad()
            loss = consistency_loss(model(latents,tokens,x_t1,t1),
                                    model(latents,tokens,x_t2,t2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            step += 1
            if step % cfg.log_every == 0:
                avg = total_loss/cfg.log_every; total_loss = 0.0
                mem = torch.cuda.memory_allocated()/1e9 if use_accel else 0.0
                print(f"step {step:>7,} | loss {avg:.5f} | lr {lr:.2e} | mem {mem:.1f}GB")
            if step % cfg.save_every == 0:
                save_checkpoint(model, step, cfg)
    save_checkpoint(model, step, cfg)
    print(f"\n✅ Done — {step:,} steps")

def smoke_test():
    class C:
        model_type="b"; dim=64; n_layers=2; n_heads=4; latent_ch=16
        patch_size=4; short_k=8; head_layers=2; noise_std=0.1
        dropout=0.0; vocab_size=32128
    cfg = C()
    model = get_model(cfg)
    x0 = torch.randn(2,16)
    x_t1,x_t2,t1,t2 = get_noisy_pair(x0)
    opt = torch.optim.AdamW(model.parameters())
    opt.zero_grad()
    loss = consistency_loss(
        model(torch.randn(2,16,4,16,16), torch.randint(0,32128,(2,32)), x_t1, t1),
        model(torch.randn(2,16,4,16,16), torch.randint(0,32128,(2,32)), x_t2, t2))
    loss.backward(); opt.step()
    print(f"✅ Model B smoke test passed | loss={loss.item():.5f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    smoke_test() if args.smoke else train()
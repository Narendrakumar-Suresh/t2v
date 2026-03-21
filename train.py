"""
train.py — Video Generation Training
=====================================
Usage:
  python train.py --model a          # TF + TF
  python train.py --model b          # Mamba2 + TF
  python train.py --model c          # Mamba2 + Mamba2
  python train.py --model a --smoke  # 1 step CPU smoke test
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from safetensors.torch import save_file
from dotenv import load_dotenv

load_dotenv()

# ── ROCm flags (ignored on non-AMD) ──────────────────────────────────
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"]       = "1"
os.environ["GPU_MAX_HW_QUEUES"]           = "2"


# ── Config ────────────────────────────────────────────────────────────
class Config:
    # model ~500M
    dim         = 896
    n_layers    = 24
    n_heads     = 16
    latent_ch   = 16
    patch_size  = 4
    short_k     = 64
    head_layers = 8
    noise_std   = 0.1
    dropout     = 0.0
    vocab_size  = 32128   # flan-t5-base

    # training
    batch_size   = 32
    lr           = 3e-4
    weight_decay = 0.1
    grad_clip    = 1.0
    max_steps    = 300_000
    warmup_steps = 1_000

    # logging
    log_every  = 100
    save_every = 25_000

    # data
    use_merged   = True
    max_text_len = 128
    num_workers  = 4

    # HF
    HF_TOKEN = os.environ.get("HF_TOKEN")


# ── Model selector ────────────────────────────────────────────────────
def get_model(model_type: str, cfg):
    kwargs = dict(
        dim             = cfg.dim,
        n_layers        = cfg.n_layers,
        n_heads         = cfg.n_heads,
        latent_channels = cfg.latent_ch,
        patch_size      = cfg.patch_size,
        short_k         = cfg.short_k,
        head_layers     = cfg.head_layers,
        noise_std       = cfg.noise_std,
        dropout         = cfg.dropout,
        vocab_size      = cfg.vocab_size,
    )
    if model_type == "a":
        from models.model_a import ModelA
        return ModelA(**kwargs)
    elif model_type == "b":
        from models.model_b import ModelB
        return ModelB(**kwargs)
    elif model_type == "c":
        from models.model_c import ModelC
        return ModelC(**kwargs)
    raise ValueError(f"Unknown model: {model_type}")


# ── Consistency loss ──────────────────────────────────────────────────
def consistency_loss(pred_1: torch.Tensor, pred_2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_1, pred_2.detach())


def get_noisy_pair(x0: torch.Tensor):
    B, C = x0.shape
    t1   = torch.rand(B, 1, device=x0.device)
    t2   = (t1 + 0.05).clamp(0.0, 1.0)
    x_t1 = x0 + torch.randn_like(x0) * t1
    x_t2 = x0 + torch.randn_like(x0) * t2
    return x_t1, x_t2, t1, t2


# ── LR scheduler ─────────────────────────────────────────────────────
def get_lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint ───────────────────────────────────────────────────────
def save_checkpoint(model, step: int, model_type: str, cfg):
    save_dir = f"./checkpoints/model_{model_type}"
    os.makedirs(save_dir, exist_ok=True)

    path  = f"{save_dir}/step_{step:07d}.safetensors"
    state = model._orig_mod.state_dict() \
        if hasattr(model, "_orig_mod") else model.state_dict()
    save_file({k: v.cpu() for k, v in state.items()}, path)
    print(f"  ✓ checkpoint → {path}")

    if cfg.HF_TOKEN:
        api     = HfApi(token=cfg.HF_TOKEN)
        repo_id = f"entropyspace/videogen-model-{model_type}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"step_{step:07d}.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  ✓ uploaded → huggingface.co/{repo_id}")


# ── Training loop ─────────────────────────────────────────────────────
def train(model_type: str):
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print(f"  Device:  {device}")
    print(f"  Model:   {model_type.upper()}")
    if device.type == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("=" * 55)

    model = get_model(model_type, cfg).to(device)
    if device.type == "cuda":                  # ← only compile on GPU
        model = torch.compile(model)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_p/1e6:.1f}M\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    from data.dataloader import VideoLatentDataset
    dataset    = VideoLatentDataset(
        use_merged=cfg.use_merged, max_text_len=cfg.max_text_len
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model.train()
    step, total_loss = 0, 0.0
    print(f"  Training {cfg.max_steps:,} steps | save every {cfg.save_every:,}\n")

    while step < cfg.max_steps:
        for batch in dataloader:
            if step >= cfg.max_steps:
                break

            latents = batch["latents"].to(device)   # [B, C, T, H, W]
            tokens  = batch["tokens"].to(device)    # [B, seq_len]
            x0      = batch["x0"].to(device)        # [B, C] clean latent

            x_t1, x_t2, t1, t2 = get_noisy_pair(x0)

            lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            pred_1 = model(latents, tokens, x_t1, t1)
            pred_2 = model(latents, tokens, x_t2, t2)
            loss   = consistency_loss(pred_1, pred_2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            step       += 1

            if step % cfg.log_every == 0:
                avg        = total_loss / cfg.log_every
                total_loss = 0.0
                mem        = torch.cuda.memory_allocated()/1e9 \
                             if device.type == "cuda" else 0.0
                print(f"step {step:>7,} | loss {avg:.5f} | lr {lr:.2e} | mem {mem:.1f}GB")

            if step % cfg.save_every == 0:
                save_checkpoint(model, step, model_type, cfg)

    save_checkpoint(model, step, model_type, cfg)
    print(f"\n✅ Done — {step:,} steps")


# ── Smoke test ────────────────────────────────────────────────────────
def smoke_test(model_type: str):
    print(f"\nSmoke test — Model {model_type.upper()} on CPU...")

    # fresh tiny config — never mutates Config class
    class SmokeCfg:
        dim         = 64
        n_layers    = 2
        n_heads     = 4
        latent_ch   = 16
        patch_size  = 4
        short_k     = 8
        head_layers = 2
        noise_std   = 0.1
        dropout     = 0.0
        vocab_size  = 32128

    cfg    = SmokeCfg()
    device = torch.device("cpu")
    model  = get_model(model_type, cfg).to(device)
    # NO torch.compile on CPU

    latents = torch.randn(2, 16, 4, 16, 16)
    tokens  = torch.randint(0, 32128, (2, 32))
    x0      = torch.randn(2, 16)

    x_t1, x_t2, t1, t2 = get_noisy_pair(x0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    pred_1 = model(latents, tokens, x_t1, t1)
    pred_2 = model(latents, tokens, x_t2, t2)
    loss   = consistency_loss(pred_1, pred_2)
    loss.backward()
    optimizer.step()

    assert pred_1.shape == (2, 16), f"Expected (2,16) got {pred_1.shape}"
    print(f"  pred:  {pred_1.shape}")
    print(f"  loss:  {loss.item():.5f}")
    print(f"✅ Smoke test passed — Model {model_type.upper()} works!")


# ── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", "c"], required=True,
                        help="a=TF+TF | b=Mamba2+TF | c=Mamba2+Mamba2")
    parser.add_argument("--smoke", action="store_true",
                        help="1-step CPU test only")
    args = parser.parse_args()

    if args.smoke:
        smoke_test(args.model)
    else:
        train(args.model)
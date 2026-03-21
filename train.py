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
from dotenv import load_dotenv

load_dotenv()

# ── ROCm environment flags ────────────────────────────────────────────
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"]       = "1"
os.environ["GPU_MAX_HW_QUEUES"]           = "2"


# ── Config ────────────────────────────────────────────────────────────
class Config:
    # model
    dim         = 512
    n_layers    = 12
    n_heads     = 8
    latent_ch   = 16
    patch_size  = 4
    short_k     = 64
    head_layers = 8
    noise_std   = 0.1
    dropout     = 0.0

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
def get_model(model_type: str, cfg: Config):
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
    else:
        raise ValueError(f"Unknown model: {model_type}")


# ── Consistency loss ──────────────────────────────────────────────────
def consistency_loss(pred_1: torch.Tensor, pred_2: torch.Tensor) -> torch.Tensor:
    """
    Both predictions should denoise to same x0.
    Stop gradient on pred_2.
    """
    return F.mse_loss(pred_1, pred_2.detach())


def get_noisy_pair(x0: torch.Tensor):
    """
    Sample two adjacent noise levels for same clean latent.
    Returns noisy versions + noise levels.
    """
    B, C   = x0.shape
    t1     = torch.rand(B, 1, device=x0.device)
    t2     = (t1 + 0.05).clamp(0, 1)
    x_t1   = x0 + torch.randn_like(x0) * t1
    x_t2   = x0 + torch.randn_like(x0) * t2
    return x_t1, x_t2, t1, t2


# ── LR scheduler (cosine with warmup) ────────────────────────────────
def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    import math
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Save checkpoint ───────────────────────────────────────────────────
def save_checkpoint(model, step: int, model_type: str, cfg: Config):
    save_dir = f"./checkpoints/model_{model_type}"
    os.makedirs(save_dir, exist_ok=True)

    path = f"{save_dir}/step_{step:07d}.safetensors"
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, path)
    print(f"  ✓ checkpoint → {path}")

    if cfg.HF_TOKEN:
        api     = HfApi(token=cfg.HF_TOKEN)
        repo_id = f"entropyspace/videogen-model-{model_type}"
        api.create_repo(
            repo_id=repo_id, repo_type="model", exist_ok=True
        )
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

    print("=" * 50)
    print(f"Device:  {device}")
    print(f"Model:   {model_type.upper()}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("=" * 50)

    # model
    model    = get_model(model_type, cfg).to(device)
    model    = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params:  {n_params/1e6:.1f}M\n")

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.95),
    )

    # dataloader
    from data.dataloader import VideoLatentDataset
    dataset    = VideoLatentDataset(
        use_merged   = cfg.use_merged,
        max_text_len = cfg.max_text_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )

    # training
    model.train()
    step       = 0
    total_loss = 0.0

    print(f"Training for {cfg.max_steps:,} steps...")
    print(f"Saving every {cfg.save_every:,} steps\n")

    while step < cfg.max_steps:
        for batch in dataloader:
            if step >= cfg.max_steps:
                break

            # move to device
            latents = batch["latents"].to(device)   # [B, C, T, H, W]
            tokens  = batch["tokens"].to(device)    # [B, seq_len]
            x0      = batch["x_noisy"].to(device)   # [B, C]

            # sample two adjacent noise levels
            x_t1, x_t2, t1, t2 = get_noisy_pair(x0)

            # update lr
            lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # forward
            optimizer.zero_grad()
            pred_1 = model(latents, tokens, x_t1, t1)
            pred_2 = model(latents, tokens, x_t2, t2)

            # loss
            loss = consistency_loss(pred_1, pred_2)

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            step       += 1

            # log
            if step % cfg.log_every == 0:
                avg   = total_loss / cfg.log_every
                total_loss = 0.0
                mem   = torch.cuda.memory_allocated()/1e9 if device.type=="cuda" else 0
                print(
                    f"step {step:>7,} | "
                    f"loss {avg:.4f} | "
                    f"lr {lr:.2e} | "
                    f"mem {mem:.1f}GB"
                )

            # checkpoint
            if step % cfg.save_every == 0:
                save_checkpoint(model, step, model_type, cfg)

    # final save
    save_checkpoint(model, step, model_type, cfg)
    print(f"\n✅ Done! {step:,} steps completed.")


# ── Smoke test (CPU, tiny model) ──────────────────────────────────────
def smoke_test(model_type: str):
    """Run 1 step on CPU with tiny model — validates pipeline."""
    print(f"Smoke test — model {model_type.upper()} on CPU...")

    cfg          = Config()
    cfg.dim      = 64
    cfg.n_layers = 2
    cfg.n_heads  = 4
    cfg.max_steps= 1

    device = torch.device("cpu")
    model  = get_model(model_type, cfg).to(device)

    latents = torch.randn(2, 16, 4, 16, 16)
    tokens  = torch.randint(0, 32000, (2, 32))
    x0      = torch.randn(2, 16)

    x_t1, x_t2, t1, t2 = get_noisy_pair(x0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    pred_1 = model(latents, tokens, x_t1, t1)
    pred_2 = model(latents, tokens, x_t2, t2)
    loss   = consistency_loss(pred_1, pred_2)
    loss.backward()
    optimizer.step()

    print(f"  loss:  {loss.item():.4f}")
    print(f"  pred:  {pred_1.shape}")
    print(f"✅ Smoke test passed — model {model_type.upper()} works!")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["a", "b", "c"], required=True,
        help="a=TF+TF  b=Mamba2+TF  c=Mamba2+Mamba2"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run 1 step on CPU to validate pipeline"
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test(args.model)
    else:
        train(args.model)
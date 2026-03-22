"""
eval.py — Model Performance Evaluation
========================================
Computes FVD, CLIP score on generated videos.
Saves sample videos for qualitative inspection.

Usage:
  python eval.py --model a --checkpoint checkpoints/model_a/step_0300000.safetensors
  python eval.py --model b --checkpoint checkpoints/model_b/step_0300000.safetensors
  python eval.py --model c --checkpoint checkpoints/model_c/step_0300000.safetensors
  python eval.py --all
"""

import os, io, json, argparse
import torch
import numpy as np
from safetensors.torch import load_file
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ── Per-model configs (each tuned to ~500M) ───────────────────────────
MODEL_CONFIGS = {
    "a": dict(dim=768,  n_layers=22, n_heads=12),   # 497M TF+TF
    "b": dict(dim=768,  n_layers=28, n_heads=12),   # 500M M2+TF
    "c": dict(dim=576,  n_layers=26, n_heads=9),    # 501M M2+M2
}

# ── Shared eval config ────────────────────────────────────────────────
class EvalConfig:
    latent_ch       = 16
    patch_size      = 4
    short_k         = 64
    head_layers     = 8
    noise_std       = 0.0    # no noise at inference
    dropout         = 0.0
    vocab_size      = 32128

    n_samples       = 256
    inference_steps = 4
    batch_size      = 8
    max_text_len    = 128
    save_videos     = True
    save_dir        = "./eval_outputs"
    HF_TOKEN        = os.environ.get("HF_TOKEN")

cfg = EvalConfig()
os.makedirs(cfg.save_dir, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────
def load_model(model_type: str, checkpoint_path: str, device):
    mc     = MODEL_CONFIGS[model_type]
    kwargs = dict(
        dim             = mc["dim"],
        n_layers        = mc["n_layers"],
        n_heads         = mc["n_heads"],
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
        model = ModelA(**kwargs)
    elif model_type == "b":
        from models.model_b import ModelB
        model = ModelB(**kwargs)
    elif model_type == "c":
        from models.model_c import ModelC
        model = ModelC(**kwargs)

    weights = load_file(checkpoint_path)
    model.load_state_dict(weights)
    model = model.to(device).eval()
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded Model {model_type.upper()} | {n_p/1e6:.1f}M params")
    return model


# ── Consistency inference ─────────────────────────────────────────────
@torch.no_grad()
def generate_latent(
    model,
    latents:    torch.Tensor,   # [B, C, T, H, W]
    tokens:     torch.Tensor,   # [B, seq_len]
    device,
    token_mask: torch.Tensor = None,
    video_mask: torch.Tensor = None,
    steps:      int = 4,
) -> torch.Tensor:
    """Pure noise → clean latent in `steps` consistency steps."""
    B, C, T, H, W = latents.shape
    # x should be spatial [B, C, H, W] to match model expectation
    x   = torch.randn(B, C, H, W, device=device)
    ts  = torch.linspace(1.0, 0.05, steps, device=device)

    for i, t_val in enumerate(ts):
        t   = torch.full((B, 1), t_val.item(), device=device)
        x0  = model(latents, tokens, x, t, token_mask=token_mask, video_mask=video_mask)
        if i < len(ts) - 1:
            x = x0 + torch.randn_like(x0) * ts[i + 1].item()
        else:
            x = x0

    return x0   # [B, C, H, W]


# ── Cosmos VAE decoder ────────────────────────────────────────────────
def get_decoder():
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        dec = CausalVideoTokenizer(
            checkpoint_dec="pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/decoder.jit"
        )
        print("✓ Cosmos decoder loaded")
        return dec
    except ImportError:
        print("⚠️  cosmos_tokenizer not installed — video decode disabled")
        return None


def decode_latent(decoder, latent: torch.Tensor) -> np.ndarray:
    """latent [C,T,H,W] → frames [T,H,W,3] uint8"""
    # ensure input is [B, C, T, H, W] for decoder
    v = decoder.decode(latent.unsqueeze(0)).squeeze(0)
    v = (v.float() + 1.0) / 2.0
    v = v.clamp(0, 1).cpu().numpy()
    v = (v * 255).astype(np.uint8)
    # v is [C, T, H, W] -> [T, H, W, C]
    return v.transpose(1, 2, 3, 0)


# ── CLIP scorer ───────────────────────────────────────────────────────
class CLIPScorer:
    def __init__(self, device):
        print("Loading CLIP...")
        self.device    = device
        self.model     = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    @torch.no_grad()
    def score(self, frames: np.ndarray, caption: str) -> float:
        from PIL import Image
        scores = []
        for idx in np.linspace(0, len(frames)-1, min(4, len(frames)), dtype=int):
            img    = Image.fromarray(frames[idx])
            inputs = self.processor(text=[caption], images=img,
                                    return_tensors="pt", padding=True).to(self.device)
            scores.append(self.model(**inputs).logits_per_image.item())
        return float(np.mean(scores))


# ── FVD ───────────────────────────────────────────────────────────────
def compute_fvd_features(frames_list, device):
    try:
        import torch_fvd
        return torch_fvd.compute_features(frames_list, device=device)
    except ImportError:
        print("⚠️  torch_fvd not available — FVD = -1")
        return None

def compute_fvd(real_feat, fake_feat) -> float:
    try:
        import torch_fvd
        return float(torch_fvd.compute_fvd(real_feat, fake_feat))
    except:
        return -1.0


# ── Save video ────────────────────────────────────────────────────────
def save_video(frames: np.ndarray, path: str, fps: float = 24.0):
    try:
        import torchvision
        torchvision.io.write_video(path, torch.from_numpy(frames), fps=fps)
    except Exception as e:
        print(f"  ⚠️  save_video: {e}")


# ── Main eval ─────────────────────────────────────────────────────────
def evaluate(model_type: str, checkpoint_path: str) -> dict:
    use_accel = torch.accelerator.is_available()
    device    = torch.accelerator.current_accelerator() if use_accel \
                else torch.device("cpu")

    print(f"\n{'='*55}")
    print(f"  Evaluating Model {model_type.upper()}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*55}")

    model     = load_model(model_type, checkpoint_path, device)
    decoder   = get_decoder()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    clip      = CLIPScorer(device)

    ds = load_dataset("joekraper/openvid-latents",
                      split="train", streaming=True)

    real_videos, gen_videos = [], []
    clip_scores             = []
    b_lat, b_tok, b_cap, b_real, b_tmask, b_vmask = [], [], [], [], [], []
    count = 0

    for row in tqdm(ds, total=cfg.n_samples):
        if count >= cfg.n_samples:
            break

        real_lat = torch.load(io.BytesIO(row["serialized_latent"]),
                              weights_only=True, map_location="cpu").float()
        
        # Tokenize with mask
        token_out = tokenizer(row["caption"], max_length=cfg.max_text_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        toks  = token_out["input_ids"].squeeze(0)
        tmask = token_out["attention_mask"].squeeze(0)

        b_lat.append(real_lat)
        b_tok.append(toks)
        b_tmask.append(tmask)
        b_cap.append(row["caption"])
        b_real.append(real_lat)
        count += 1

        if len(b_lat) == cfg.batch_size or count == cfg.n_samples:
            # pad T dimension for batching
            MAX_T_EVAL = 16  # 15 past + 1 current
            padded, vmasks = [], []
            for lat in b_lat:
                C, T, H, W = lat.shape
                # past context is everything but the last frame (if we follow dataloader logic)
                # But here lat is full video. generate_latent takes latents as past context.
                # So we should pass first T-1 frames as past context.
                past = lat[:, :-1, :, :]
                C_p, T_p, H_p, W_p = past.shape
                target_T = MAX_T_EVAL - 1
                
                if T_p > target_T:
                    past = past[:, -target_T:, :, :]
                    vmask = torch.ones(target_T)
                else:
                    pad = torch.zeros(C_p, target_T - T_p, H_p, W_p)
                    past = torch.cat([pad, past], dim=1)
                    vmask = torch.cat([torch.zeros(target_T - T_p), torch.ones(T_p)])
                
                padded.append(past)
                vmasks.append(vmask)

            # lat_batch must be scaled by 1/8.0 as it was during training
            lat_batch = (torch.stack(padded) / 8.0).to(device)
            tok_batch = torch.stack(b_tok).to(device)
            tm_batch  = torch.stack(b_tmask).to(device)
            vm_batch  = torch.stack(vmasks).to(device)

            gen_lats = generate_latent(model, lat_batch, tok_batch, device,
                                       token_mask=tm_batch, video_mask=vm_batch,
                                       steps=cfg.inference_steps)

            if decoder is not None:
                for i in range(len(b_lat)):
                    real_i = b_real[i] # [C, T, H, W]
                    C, T, H, W = real_i.shape
                    
                    # ── Improve visualization: Past + Gen ─────────────────────
                    # Past frames (raw range)
                    past_frames_lat = real_i[:, :-1, :, :]
                    # Generated frame (scaled back to raw range)
                    gen_frame_lat = (gen_lats[i] * 8.0).cpu().unsqueeze(1) # [C, 1, H, W]
                    # Full video latent
                    gen_i = torch.cat([past_frames_lat, gen_frame_lat], dim=1).contiguous()

                    gen_frames  = decode_latent(decoder, gen_i)
                    real_frames = decode_latent(decoder, real_i)

                    clip_scores.append(clip.score(gen_frames, b_cap[i]))
                    gen_videos.append(gen_frames)
                    real_videos.append(real_frames)

                    if cfg.save_videos and len(gen_videos) <= 10:
                        p = f"{cfg.save_dir}/model_{model_type}_{len(gen_videos):03d}"
                        save_video(gen_frames, f"{p}.mp4")
                        open(f"{p}.txt","w").write(b_cap[i])

            b_lat, b_tok, b_cap, b_real = [], [], [], []

    # FVD
    fvd = -1.0
    if real_videos and gen_videos:
        rf = compute_fvd_features(real_videos, device)
        gf = compute_fvd_features(gen_videos,  device)
        if rf is not None and gf is not None:
            fvd = compute_fvd(rf, gf)

    results = {
        "model":      model_type.upper(),
        "params_M":   round(sum(p.numel() for p in model.parameters())/1e6, 1),
        "checkpoint": checkpoint_path,
        "n_samples":  count,
        "fvd":        round(fvd, 2),
        "clip_score": round(float(np.mean(clip_scores)), 4) if clip_scores else -1,
    }

    print(f"\n  Params:     {results['params_M']}M")
    print(f"  FVD:        {results['fvd']}  ↓")
    print(f"  CLIP Score: {results['clip_score']}  ↑")

    out = f"{cfg.save_dir}/results_{model_type}.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"  ✓ → {out}")
    return results


# ── Compare all ───────────────────────────────────────────────────────
def compare_all():
    ckpts = {
        "a": "checkpoints/model_a/step_0300000.safetensors",
        "b": "checkpoints/model_b/step_0300000.safetensors",
        "c": "checkpoints/model_c/step_0300000.safetensors",
    }
    all_r = []
    for mt, ckpt in ckpts.items():
        if not os.path.exists(ckpt):
            print(f"⚠️  {ckpt} not found — skipping"); continue
        all_r.append(evaluate(mt, ckpt))

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':>8} {'FVD↓':>10} {'CLIP↑':>10}")
    print(f"{'─'*60}")
    for r in all_r:
        print(f"  {r['model']:<8} {r['params_M']:>7}M {r['fvd']:>10} {r['clip_score']:>10}")
    print(f"{'='*60}")

    with open(f"{cfg.save_dir}/comparison.json","w") as f:
        json.dump(all_r, f, indent=2)
    print(f"✓ → {cfg.save_dir}/comparison.json")


# ── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["a","b","c"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--samples",    type=int, default=256)
    parser.add_argument("--steps",      type=int, default=4)
    args = parser.parse_args()

    cfg.n_samples       = args.samples
    cfg.inference_steps = args.steps

    if args.all:
        compare_all()
    elif args.model and args.checkpoint:
        evaluate(args.model, args.checkpoint)
    else:
        parser.print_help()
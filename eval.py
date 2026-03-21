"""
eval.py — Model Performance Evaluation
========================================
Computes FVD, CLIP score, PSNR on generated videos.
Saves sample videos for qualitative inspection.

Usage:
  python eval.py --model a --checkpoint checkpoints/model_a/step_0300000.safetensors
  python eval.py --all    # compare all three models
"""

import os
import io
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────
class EvalConfig:
    dim         = 896
    n_layers    = 24
    n_heads     = 16
    latent_ch   = 16
    patch_size  = 4
    short_k     = 64
    head_layers = 8
    noise_std   = 0.0    # no noise at inference
    dropout     = 0.0
    vocab_size  = 32128

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
    kwargs = dict(
        dim=cfg.dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        latent_channels=cfg.latent_ch, patch_size=cfg.patch_size,
        short_k=cfg.short_k, head_layers=cfg.head_layers,
        noise_std=cfg.noise_std, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size,
    )
    if model_type == "a":
        from models.model_a import ModelA; model = ModelA(**kwargs)
    elif model_type == "b":
        from models.model_b import ModelB; model = ModelB(**kwargs)
    elif model_type == "c":
        from models.model_c import ModelC; model = ModelC(**kwargs)

    weights = load_file(checkpoint_path)
    model.load_state_dict(weights)
    model = model.to(device).eval()
    print(f"✓ Loaded model {model_type.upper()} from {checkpoint_path}")
    return model


# ── Consistency inference ─────────────────────────────────────────────
@torch.no_grad()
def generate_latent(
    model,
    latents: torch.Tensor,   # [B, C, T, H, W] past context
    tokens:  torch.Tensor,   # [B, seq_len]
    device,
    steps:   int = 4,
) -> torch.Tensor:
    """
    Consistency sampling: pure noise → clean latent in `steps` steps.
    Returns predicted clean latent [B, C].
    """
    B, C = latents.shape[0], cfg.latent_ch

    # start from pure noise
    x = torch.randn(B, C, device=device)

    # noise schedule: 1.0 → 0.0
    ts = torch.linspace(1.0, 0.05, steps, device=device)

    for t_val in ts:
        # ← FIX: correct way to create [B, 1] tensor from scalar
        t    = torch.full((B, 1), t_val.item(), device=device)
        x0   = model(latents, tokens, x, t)
        # re-noise at next level
        idx  = (ts == t_val).nonzero(as_tuple=True)[0]
        if idx < len(ts) - 1:
            t_next = ts[idx + 1].item()
            x      = x0 + torch.randn_like(x0) * t_next
        else:
            x = x0

    return x0   # [B, C]


# ── Cosmos VAE decoder ────────────────────────────────────────────────
def get_decoder():
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        decoder = CausalVideoTokenizer(
            checkpoint_dec="pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/decoder.jit"
        )
        print("✓ Cosmos decoder loaded")
        return decoder
    except ImportError:
        print("⚠️  cosmos_tokenizer not installed — video decode disabled")
        return None


def decode_latent(decoder, latent: torch.Tensor) -> np.ndarray:
    """
    latent: [C, T, H, W]
    returns: [T, H, W, 3] uint8
    """
    video = decoder.decode(latent.unsqueeze(0)).squeeze(0)   # [3, T, H, W]
    video = (video.float() + 1.0) / 2.0
    video = video.clamp(0, 1).cpu().numpy()
    video = (video * 255).astype(np.uint8)
    return video.transpose(1, 2, 3, 0)    # [T, H, W, 3]


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
        idxs   = np.linspace(0, len(frames) - 1, min(4, len(frames)), dtype=int)
        for idx in idxs:
            img    = Image.fromarray(frames[idx])
            inputs = self.processor(
                text=[caption], images=img,
                return_tensors="pt", padding=True
            ).to(self.device)
            out  = self.model(**inputs)
            scores.append(out.logits_per_image.item())
        return float(np.mean(scores))


# ── FVD helpers ───────────────────────────────────────────────────────
def compute_fvd_features(frames_list, device):
    try:
        import torch_fvd
        return torch_fvd.compute_features(frames_list, device=device)
    except ImportError:
        print("⚠️  torch_fvd not available — FVD will be -1")
        return None


def compute_fvd(real_feat, fake_feat) -> float:
    try:
        import torch_fvd
        return float(torch_fvd.compute_fvd(real_feat, fake_feat))
    except:
        return -1.0


# ── PSNR ─────────────────────────────────────────────────────────────
def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    return 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))


# ── Save video ────────────────────────────────────────────────────────
def save_video(frames: np.ndarray, path: str, fps: float = 24.0):
    try:
        import torchvision
        torchvision.io.write_video(path, torch.from_numpy(frames), fps=fps)
    except Exception as e:
        print(f"  ⚠️  save_video failed: {e}")


# ── Main eval ─────────────────────────────────────────────────────────
def evaluate(model_type: str, checkpoint_path: str) -> dict:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Model {model_type.upper()} | {checkpoint_path}")
    print(f"{'='*55}")

    model     = load_model(model_type, checkpoint_path, device)
    decoder   = get_decoder()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    clip      = CLIPScorer(device)

    ds = load_dataset("entropyspace/openvid-latents", split="train", streaming=True)

    real_videos  = []
    gen_videos   = []
    clip_scores  = []
    psnr_scores  = []

    b_latents, b_tokens, b_captions, b_reals = [], [], [], []
    count = 0

    for row in tqdm(ds, total=cfg.n_samples, desc=f"Model {model_type.upper()}"):
        if count >= cfg.n_samples:
            break

        real_lat = torch.load(
            io.BytesIO(row["serialized_latent"]),
            weights_only=True, map_location="cpu"
        ).float()

        toks = tokenizer(
            row["caption"],
            max_length=cfg.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        b_latents.append(real_lat)
        b_tokens.append(toks)
        b_captions.append(row["caption"])
        b_reals.append(real_lat)
        count += 1

        if len(b_latents) == cfg.batch_size or count == cfg.n_samples:
            lat_batch = torch.stack(b_latents).to(device)
            tok_batch = torch.stack(b_tokens).to(device)

            gen_lats = generate_latent(
                model, lat_batch, tok_batch,
                device, steps=cfg.inference_steps
            )  # [B, C]

            if decoder is not None:
                for i in range(len(b_latents)):
                    real_lat_i = b_reals[i]          # [C, T, H, W]
                    C, T, H, W = real_lat_i.shape

                    # expand generated [C] → [C, T, H, W] by repeating
                    gen_lat_i = gen_lats[i].cpu()     # [C]
                    gen_lat_i = gen_lat_i.view(C, 1, 1, 1).expand(C, T, H, W).contiguous()

                    gen_frames  = decode_latent(decoder, gen_lat_i)
                    real_frames = decode_latent(decoder, real_lat_i)

                    cs = clip.score(gen_frames, b_captions[i])
                    ps = compute_psnr(gen_frames, real_frames)
                    clip_scores.append(cs)
                    psnr_scores.append(ps)
                    gen_videos.append(gen_frames)
                    real_videos.append(real_frames)

                    if cfg.save_videos and len(gen_videos) <= 10:
                        vid_path = f"{cfg.save_dir}/model_{model_type}_{len(gen_videos):03d}.mp4"
                        txt_path = vid_path.replace(".mp4", ".txt")
                        save_video(gen_frames, vid_path)
                        with open(txt_path, "w") as f:
                            f.write(b_captions[i])

            b_latents, b_tokens, b_captions, b_reals = [], [], [], []

    # FVD
    fvd = -1.0
    if real_videos and gen_videos:
        rf = compute_fvd_features(real_videos, device)
        gf = compute_fvd_features(gen_videos,  device)
        if rf is not None and gf is not None:
            fvd = compute_fvd(rf, gf)

    results = {
        "model":      model_type.upper(),
        "checkpoint": checkpoint_path,
        "n_samples":  count,
        "fvd":        round(fvd, 2),
        "clip_score": round(float(np.mean(clip_scores)), 4) if clip_scores else -1,
        "psnr":       round(float(np.mean(psnr_scores)), 2) if psnr_scores else -1,
    }

    print(f"\n  FVD:        {results['fvd']}  ↓")
    print(f"  CLIP Score: {results['clip_score']}  ↑")
    print(f"  PSNR:       {results['psnr']} dB  ↑")

    out = f"{cfg.save_dir}/results_{model_type}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved → {out}")
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
            print(f"⚠️  {ckpt} not found — skipping")
            continue
        all_r.append(evaluate(mt, ckpt))

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<14} {'FVD↓':<12} {'CLIP↑':<12} {'PSNR↑':<10}")
    print(f"{'─'*60}")
    for r in all_r:
        print(f"  {r['model']:<12} {r['fvd']:<12} {r['clip_score']:<12} {r['psnr']}")
    print(f"{'='*60}")

    with open(f"{cfg.save_dir}/comparison.json", "w") as f:
        json.dump(all_r, f, indent=2)
    print(f"✓ → {cfg.save_dir}/comparison.json")


# ── Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["a", "b", "c"])
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
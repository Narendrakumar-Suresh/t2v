"""
eval.py — Model Performance Evaluation
========================================
Computes FVD, CLIP score, PSNR on generated videos.
Saves sample videos for qualitative inspection.

Usage:
  python eval.py --model a --checkpoint checkpoints/model_a/step_0300000.safetensors
  python eval.py --model b --checkpoint checkpoints/model_b/step_0300000.safetensors
  python eval.py --model c --checkpoint checkpoints/model_c/step_0300000.safetensors

  # compare all three
  python eval.py --all
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
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
)
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
class EvalConfig:
    # model
    dim         = 1024
    n_layers    = 24
    n_heads     = 16
    latent_ch   = 16
    patch_size  = 4
    short_k     = 64
    head_layers = 8
    noise_std   = 0.0    # no noise at inference
    dropout     = 0.0

    # eval
    n_samples       = 256     # number of videos to generate
    inference_steps = 4       # consistency model steps
    batch_size      = 8
    max_text_len    = 128
    save_videos     = True
    save_dir        = "./eval_outputs"

    # data
    HF_TOKEN = os.environ.get("HF_TOKEN")

cfg = EvalConfig()
os.makedirs(cfg.save_dir, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────
def load_model(model_type: str, checkpoint_path: str, device):
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
    print(f"✓ Loaded {model_type.upper()} from {checkpoint_path}")
    return model


# ── Inference (consistency sampling) ─────────────────────────────────
@torch.no_grad()
def generate_latent(
    model,
    latents:  torch.Tensor,   # [B, C, T, H, W] past context
    tokens:   torch.Tensor,   # [B, seq_len]
    device,
    steps:    int = 4,
) -> torch.Tensor:
    """
    Consistency model inference — 4 steps max.
    Starts from pure noise → progressively denoise.
    """
    B, C = latents.shape[0], cfg.latent_ch

    # start from pure noise
    x = torch.randn(B, C, device=device)

    # noise schedule: linearly from 1.0 → 0.0
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]

    for t_val in ts:
        t = t_val.expand(B, 1)
        # predict clean x0
        x0_pred = model(latents, tokens, x, t)
        # re-noise at next level (consistency step)
        if t_val > 0:
            t_next  = ts[ts < t_val]
            t_next  = t_next[0] if len(t_next) > 0 else torch.zeros(1, device=device)
            x       = x0_pred + torch.randn_like(x0_pred) * t_next
        else:
            x = x0_pred

    return x0_pred   # [B, C] predicted clean latent


# ── Cosmos VAE decoder ────────────────────────────────────────────────
def get_decoder():
    """Load Cosmos tokenizer decoder."""
    try:
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        decoder = CausalVideoTokenizer(
            checkpoint_dec="pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/decoder.jit"
        )
        return decoder
    except ImportError:
        print("⚠️  cosmos_tokenizer not installed — skipping video decode")
        return None


def decode_latent(decoder, latent: torch.Tensor) -> np.ndarray:
    """latent [C, T, H, W] → video [T, H, W, 3] uint8"""
    video  = decoder.decode(latent.unsqueeze(0)).squeeze(0)  # [3, T, H, W]
    video  = (video.float() + 1.0) / 2.0
    video  = video.clamp(0, 1).cpu().numpy()
    video  = (video * 255).astype(np.uint8)
    return video.transpose(1, 2, 3, 0)   # [T, H, W, 3]


# ── CLIP score ────────────────────────────────────────────────────────
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
        """
        frames: [T, H, W, 3] uint8
        returns: mean CLIP similarity across frames
        """
        from PIL import Image
        scores = []
        # sample 4 frames
        idxs = np.linspace(0, len(frames)-1, 4, dtype=int)
        for idx in idxs:
            img    = Image.fromarray(frames[idx])
            inputs = self.processor(
                text=[caption], images=img,
                return_tensors="pt", padding=True
            ).to(self.device)
            out    = self.model(**inputs)
            sim    = out.logits_per_image.item()
            scores.append(sim)
        return float(np.mean(scores))


# ── FVD (simplified) ──────────────────────────────────────────────────
def compute_fvd_features(frames_list: list, device) -> torch.Tensor:
    """
    Extract I3D features for FVD computation.
    frames_list: list of [T, H, W, 3] numpy arrays
    Returns feature matrix [N, 400]
    """
    try:
        import torch_fvd
        features = torch_fvd.compute_features(frames_list, device=device)
        return features
    except ImportError:
        print("⚠️  torch_fvd not available — using CLIP features as proxy")
        return None


def compute_fvd(real_features, fake_features) -> float:
    """Frechet Video Distance between real and generated."""
    try:
        import torch_fvd
        return torch_fvd.compute_fvd(real_features, fake_features)
    except:
        return -1.0   # unavailable


# ── PSNR ─────────────────────────────────────────────────────────────
def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ── Save video ────────────────────────────────────────────────────────
def save_video(frames: np.ndarray, path: str, fps: float = 24.0):
    """frames: [T, H, W, 3] uint8"""
    try:
        import torchvision
        tensor = torch.from_numpy(frames)   # [T, H, W, 3]
        torchvision.io.write_video(path, tensor, fps=fps)
    except Exception as e:
        print(f"  ⚠️  Could not save video: {e}")


# ── Main eval loop ────────────────────────────────────────────────────
def evaluate(model_type: str, checkpoint_path: str):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Evaluating Model {model_type.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*50}")

    # load model
    model    = load_model(model_type, checkpoint_path, device)
    decoder  = get_decoder()
    tokenizer= AutoTokenizer.from_pretrained("google/flan-t5-base")
    clip     = CLIPScorer(device)

    # load eval samples from dataset
    print(f"\nLoading {cfg.n_samples} eval samples...")
    ds = load_dataset(
        "entropyspace/openvid-latents",
        split="train",
        streaming=True,
    )

    # collect samples
    real_videos   = []
    gen_videos    = []
    captions      = []
    clip_scores   = []
    psnr_scores   = []

    batch_latents = []
    batch_tokens  = []
    batch_captions= []
    batch_reals   = []

    count = 0
    for row in tqdm(ds, total=cfg.n_samples):
        if count >= cfg.n_samples:
            break

        # deserialize real latent
        real_latent = torch.load(
            io.BytesIO(row["serialized_latent"]),
            weights_only=True, map_location="cpu"
        ).float()   # [C, T, H, W]

        # tokenize caption
        tokens = tokenizer(
            row["caption"],
            max_length=cfg.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        batch_latents.append(real_latent)
        batch_tokens.append(tokens)
        batch_captions.append(row["caption"])
        batch_reals.append(real_latent)
        count += 1

        if len(batch_latents) == cfg.batch_size or count == cfg.n_samples:
            # stack batch
            latents_batch = torch.stack(batch_latents).to(device)  # [B,C,T,H,W]
            tokens_batch  = torch.stack(batch_tokens).to(device)   # [B,seq]

            # generate latents
            gen_latents = generate_latent(
                model, latents_batch, tokens_batch,
                device, steps=cfg.inference_steps
            )  # [B, C]

            # decode + score
            for i in range(len(batch_latents)):
                cap = batch_captions[i]

                # decode generated
                if decoder is not None:
                    # reshape latent back to [C, T, H, W]
                    gen_lat = gen_latents[i].cpu()
                    real_lat= batch_reals[i]

                    # for now use mean spatial dims from real
                    T, H, W = real_lat.shape[1], real_lat.shape[2], real_lat.shape[3]
                    gen_spatial = gen_lat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    gen_spatial = gen_spatial.expand(-1, T, H, W)

                    gen_frames  = decode_latent(decoder, gen_spatial)
                    real_frames = decode_latent(decoder, real_lat)

                    # CLIP score
                    cs = clip.score(gen_frames, cap)
                    clip_scores.append(cs)

                    # PSNR
                    ps = compute_psnr(gen_frames, real_frames)
                    psnr_scores.append(ps)

                    gen_videos.append(gen_frames)
                    real_videos.append(real_frames)

                    # save sample video
                    if cfg.save_videos and len(gen_videos) <= 10:
                        save_video(
                            gen_frames,
                            f"{cfg.save_dir}/model_{model_type}_sample_{len(gen_videos):03d}.mp4"
                        )
                        # save caption
                        with open(
                            f"{cfg.save_dir}/model_{model_type}_sample_{len(gen_videos):03d}.txt",
                            "w"
                        ) as f:
                            f.write(cap)

            batch_latents  = []
            batch_tokens   = []
            batch_captions = []
            batch_reals    = []

    # compute FVD
    print("\nComputing metrics...")
    fvd = -1.0
    if real_videos and gen_videos:
        real_feat = compute_fvd_features(real_videos, device)
        gen_feat  = compute_fvd_features(gen_videos,  device)
        if real_feat is not None and gen_feat is not None:
            fvd = compute_fvd(real_feat, gen_feat)

    # results
    results = {
        "model":      model_type.upper(),
        "checkpoint": checkpoint_path,
        "n_samples":  count,
        "fvd":        round(fvd, 2),
        "clip_score": round(float(np.mean(clip_scores)),  4) if clip_scores  else -1,
        "psnr":       round(float(np.mean(psnr_scores)),  2) if psnr_scores  else -1,
    }

    # print results
    print(f"\n{'─'*40}")
    print(f"Model:       {results['model']}")
    print(f"Samples:     {results['n_samples']}")
    print(f"FVD:         {results['fvd']}  (lower = better)")
    print(f"CLIP Score:  {results['clip_score']}  (higher = better)")
    print(f"PSNR:        {results['psnr']} dB  (higher = better)")
    print(f"{'─'*40}")

    # save results
    out_path = f"{cfg.save_dir}/results_model_{model_type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved → {out_path}")

    return results


# ── Compare all three ─────────────────────────────────────────────────
def compare_all():
    checkpoints = {
        "a": "checkpoints/model_a/step_0300000.safetensors",
        "b": "checkpoints/model_b/step_0300000.safetensors",
        "c": "checkpoints/model_c/step_0300000.safetensors",
    }

    all_results = []
    for model_type, ckpt in checkpoints.items():
        if not os.path.exists(ckpt):
            print(f"⚠️  {ckpt} not found — skipping model {model_type.upper()}")
            continue
        results = evaluate(model_type, ckpt)
        all_results.append(results)

    # print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'FVD↓':<12} {'CLIP↑':<12} {'PSNR↑':<12}")
    print(f"{'─'*60}")
    for r in all_results:
        print(
            f"{r['model']:<12} "
            f"{r['fvd']:<12} "
            f"{r['clip_score']:<12} "
            f"{r['psnr']:<12}"
        )
    print(f"{'='*60}")

    # save full comparison
    with open(f"{cfg.save_dir}/comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Full comparison → {cfg.save_dir}/comparison.json")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b", "c"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--all", action="store_true",
                        help="Compare all three models")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    cfg.n_samples       = args.samples
    cfg.inference_steps = args.steps

    if args.all:
        compare_all()
    elif args.model and args.checkpoint:
        evaluate(args.model, args.checkpoint)
    else:
        parser.print_help()
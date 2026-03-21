"""
Build joekraper/openvid-latents
================================
Merges joekraper/openvid-filtered-5to10s metadata
+ serialized latents from fal/cosmos-openvid-1m
→ pushes to NEW repo: joekraper/openvid-latents

Run on laptop:
  pip install datasets huggingface_hub pandas torch
  export HF_TOKEN=hf_xxx
  python build_local.py
"""

import gc
import io
import os

import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load the .env file
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
SAVE_DIR = "./latent_shards"
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = "entropyspace/openvid-latents"  # ← NEW repo
SHARD_SIZE = 500

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Create new HF repo ────────────────────────────────────────────────
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None
if api:
    api.create_repo(
        repo_id=HF_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    print(f"✓ Repo ready: huggingface.co/datasets/{HF_REPO}")

# ── Load filter list ──────────────────────────────────────────────────
print("Loading joekraper/openvid-filtered-5to10s...")
filter_df = load_dataset(
    "joekraper/openvid-filtered-5to10s", split="train+test"
).to_pandas()
print(f"✓ {len(filter_df):,} rows")
filter_map = {row["video"]: row for _, row in filter_df.iterrows()}

# ── Resume ────────────────────────────────────────────────────────────
done_set = set()
shard_idx = 0
for fname in sorted(f for f in os.listdir(SAVE_DIR) if f.startswith("shard_")):
    tmp = pd.read_parquet(f"{SAVE_DIR}/{fname}")
    done_set |= set(tmp["video"].tolist())
    shard_idx = max(shard_idx, int(fname.split("_")[1].split(".")[0]) + 1)

remaining = {k: v for k, v in filter_map.items() if k not in done_set}
print(f"  Done: {len(done_set):,} | Remaining: {len(remaining):,}")

# ── Stream cosmos + match ─────────────────────────────────────────────
print("\nStreaming fal/cosmos-openvid-1m...")
cosmos_ds = load_dataset("fal/cosmos-openvid-1m", split="train", streaming=True)

shard_rows = []
matched = len(done_set)
scanned = 0

for cosmos_row in cosmos_ds:
    scanned += 1
    name = cosmos_row["video"]

    if name not in remaining:
        continue

    m = remaining[name]
    shard_rows.append(
        {
            "video": name,
            "caption": m["caption"],
            "serialized_latent": cosmos_row["serialized_latent"],
            "fps": float(m["fps"]),
            "seconds": float(m["seconds"]),
            "aesthetic_score": float(m["aesthetic_score"]),
            "motion_score": float(m["motion_score"]),
            "temporal_consistency_score": float(m["temporal_consistency_score"]),
            "camera_motion": m["camera_motion"],
            "split": m["split"],
        }
    )
    matched += 1

    if len(shard_rows) >= SHARD_SIZE:
        path = f"{SAVE_DIR}/shard_{shard_idx:05d}.parquet"
        pd.DataFrame(shard_rows).to_parquet(path, index=False, compression="lz4")
        size_mb = os.path.getsize(path) / 1e6
        print(
            f"  ✓ shard_{shard_idx:05d} | {len(shard_rows)} rows"
            f" | {size_mb:.0f}MB | matched={matched:,} / scanned={scanned:,}"
        )

        if api:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"data/train/shard_{shard_idx:05d}.parquet",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            os.remove(path)  # delete local after upload
            print(f"    → uploaded")

        shard_rows.clear()
        shard_idx += 1
        gc.collect()

    if matched >= len(filter_map):
        print("✓ All matched!")
        break

# flush remainder
if shard_rows:
    path = f"{SAVE_DIR}/shard_{shard_idx:05d}.parquet"
    pd.DataFrame(shard_rows).to_parquet(path, index=False, compression="lz4")
    if api:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"data/train/shard_{shard_idx:05d}.parquet",
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        os.remove(path)

print(f"\n✅ Done! {matched:,} rows → huggingface.co/datasets/{HF_REPO}")

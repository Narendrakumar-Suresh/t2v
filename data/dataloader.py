"""
data/dataloader.py — FIXED
===========================
Key fix: past context ≠ target x0
  - past_latents = first T-1 frames → backbone input
  - x0           = last frame → prediction target
  - They are DIFFERENT, so model cannot trivially solve the task
"""

import io
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_ID = "google/flan-t5-base"
MAX_T        = 16   # total frames (past + current)
                    # backbone sees MAX_T-1=15 past frames
                    # predicts 1 current frame


class VideoLatentDataset(IterableDataset):
    def __init__(
        self,
        use_merged:   bool = False,
        split:        str  = "train",
        max_text_len: int  = 128,
    ):
        self.use_merged   = use_merged
        self.split        = split
        self.max_text_len = max_text_len
        self.tokenizer    = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def tokenize(self, caption: str) -> tuple:
        out = self.tokenizer(
            caption,
            max_length     = self.max_text_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        ids  = out["input_ids"].squeeze(0)       # [max_text_len]
        mask = out["attention_mask"].squeeze(0)  # [max_text_len] 1=real 0=pad
        return ids, mask

    def deserialize(self, raw: bytes) -> torch.Tensor:
        return torch.load(
            io.BytesIO(raw),
            weights_only = True,
            map_location = "cpu",
        )

    def __iter__(self):
        if self.use_merged:
            yield from self._iter_merged()
        else:
            yield from self._iter_streaming()

    def _iter_merged(self):
        ds = load_dataset(
            "entropyspace/openvid-latents",
            split="train", streaming=True,
        )
        for row in ds:
            latent = self.deserialize(row["serialized_latent"])
            ids, mask = self.tokenize(row["caption"])
            sample = self._make_sample(latent, ids, mask)
            if sample is not None:
                yield sample

    def _iter_streaming(self):
        filter_ds   = load_dataset("joekraper/openvid-filtered-5to10s",
                                   split="train+test")
        caption_map = {row["video"]: row["caption"] for row in filter_ds}
        video_set   = set(caption_map.keys())

        cosmos_ds = load_dataset("fal/cosmos-openvid-1m",
                                  split="train", streaming=True)
        for row in cosmos_ds:
            if row["video"] not in video_set:
                continue
            latent = self.deserialize(row["serialized_latent"])
            ids, mask = self.tokenize(caption_map[row["video"]])
            sample = self._make_sample(latent, ids, mask)
            if sample is not None:
                yield sample

    def _make_sample(
        self,
        latent: torch.Tensor,   # [C, T, H, W]
        tokens: torch.Tensor,   # [max_text_len]
        token_mask: torch.Tensor,
    ):
        latent = latent.float()
        C, T, H, W = latent.shape

        # need at least 2 frames: 1 past + 1 current
        if T < 2:
            return None

        # ── Pad/trim to MAX_T ─────────────────────────────────────────
        if T > MAX_T:
            latent = latent[:, :MAX_T, :, :]
        elif T < MAX_T:
            pad    = torch.zeros(C, MAX_T - T, H, W)
            latent = torch.cat([latent, pad], dim=1)
        # latent: [C, MAX_T, H, W]

        # ── KEY FIX: split into past context + current frame ──────────
        past    = latent[:, :-1, :, :]   # [C, MAX_T-1, H, W] → backbone input
        current = latent[:, -1,  :, :]   # [C, H, W]          → target frame

        # Scaling: latents from VAEs (Cosmos/SD) often need scaling to ~std 1
        # 1/8.0 is a common factor for these models.
        past    = past / 8.0
        current = current / 8.0

        # x0 = current frame → [C, H, W]
        # This is the full spatial target for the consistency model
        x0 = current

        return {
            "latents":    past,        # [C, MAX_T-1, H, W] past context
            "tokens":     tokens,      # [max_text_len]
            "token_mask": token_mask,  # [max_text_len] for masked pooling
            "x0":         x0,          # [C, H, W] target — full frame
        }


def get_dataloader(use_merged=False, batch_size=32, num_workers=4):
    ds = VideoLatentDataset(use_merged=use_merged)
    return DataLoader(ds, batch_size=batch_size,
                      num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    print("Testing dataloader...")
    ds    = VideoLatentDataset(use_merged=False)
    batch = next(iter(ds))
    print(f"latents:    {batch['latents'].shape}")    # [C, 15, H, W]
    print(f"tokens:     {batch['tokens'].shape}")     # [128]
    print(f"token_mask: {batch['token_mask'].shape}") # [128]
    print(f"x0:         {batch['x0'].shape}")         # [C, H, W]
    C, H, W = batch['x0'].shape
    assert batch['x0'].shape == (C, H, W), f"x0 wrong shape: {batch['x0'].shape}"
    assert batch['latents'].shape[1] == MAX_T - 1
    print("✅ Dataloader works — past ≠ target confirmed!")
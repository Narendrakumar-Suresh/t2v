"""
data/dataloader.py
==================
Loads video latents + captions for training.

use_merged=False → stream from fal/cosmos-openvid-1m (before augment.py)
use_merged=True  → load from entropyspace/openvid-latents (after augment.py)
"""

import io
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_ID = "google/flan-t5-base"  # vocab_size = 32128


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

        # verify vocab size matches model
        assert self.tokenizer.vocab_size <= 32128, \
            f"Tokenizer vocab {self.tokenizer.vocab_size} > model vocab 32128"

    def tokenize(self, caption: str) -> torch.Tensor:
        out = self.tokenizer(
            caption,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return out["input_ids"].squeeze(0)   # [max_text_len]

    def deserialize(self, raw: bytes) -> torch.Tensor:
        return torch.load(
            io.BytesIO(raw),
            weights_only=True,
            map_location="cpu",
        )

    def __iter__(self):
        if self.use_merged:
            yield from self._iter_merged()
        else:
            yield from self._iter_streaming()

    def _iter_merged(self):
        """Load from entropyspace/openvid-latents (after augment.py runs)"""
        ds = load_dataset(
            "entropyspace/openvid-latents",
            split="train",
            streaming=True,
        )
        for row in ds:
            latent = self.deserialize(row["serialized_latent"])
            tokens = self.tokenize(row["caption"])
            yield self._make_sample(latent, tokens)

    def _iter_streaming(self):
        """Stream from fal + joekraper directly (before augment.py runs)"""
        filter_ds   = load_dataset(
            "joekraper/openvid-filtered-5to10s",
            split="train+test",
        )
        caption_map = {row["video"]: row["caption"] for row in filter_ds}
        video_set   = set(caption_map.keys())

        cosmos_ds = load_dataset(
            "fal/cosmos-openvid-1m",
            split="train",
            streaming=True,
        )
        for row in cosmos_ds:
            if row["video"] not in video_set:
                continue
            latent = self.deserialize(row["serialized_latent"])
            tokens = self.tokenize(caption_map[row["video"]])
            yield self._make_sample(latent, tokens)

    def _make_sample(self, latent: torch.Tensor, tokens: torch.Tensor) -> dict:
        """
        latent: [C, T, H, W]  — from Cosmos tokenizer

        Returns:
          latents: [C, T, H, W]  full latent (past context for backbone)
          tokens:  [max_text_len] tokenized caption
          x0:      [C]            CLEAN latent summary (no noise added here)
          ← noise is added in train.py via get_noisy_pair()
        """
        # clean x0 = mean over spatial-temporal dims
        # [C, T, H, W] → [C]
        x0 = latent.float().mean(dim=(1, 2, 3))

        return {
            "latents": latent.float(),   # [C, T, H, W]
            "tokens":  tokens,           # [max_text_len]
            "x0":      x0,               # [C] ← CLEAN, no noise
        }


def get_dataloader(
    use_merged:  bool = False,
    batch_size:  int  = 32,
    num_workers: int  = 4,
) -> DataLoader:
    dataset = VideoLatentDataset(use_merged=use_merged)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )


if __name__ == "__main__":
    print("Testing dataloader (streaming mode)...")
    ds    = VideoLatentDataset(use_merged=False)
    batch = next(iter(ds))

    print(f"latents: {batch['latents'].shape}")   # [C, T, H, W]
    print(f"tokens:  {batch['tokens'].shape}")    # [max_text_len]
    print(f"x0:      {batch['x0'].shape}")        # [C]
    assert batch['x0'].shape == (16,), \
        f"Expected x0 shape (16,) got {batch['x0'].shape}"
    print("✅ Dataloader works!")
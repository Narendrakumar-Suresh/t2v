# data/dataloader.py
import io
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_ID = "google/flan-t5-base"   # simple, small

class VideoLatentDataset(IterableDataset):
    def __init__(
        self,
        use_merged: bool = False,   # True once augment.py has run
        split: str = "train",
        max_text_len: int = 128,
    ):
        self.use_merged  = use_merged
        self.split       = split
        self.max_text_len = max_text_len
        self.tokenizer   = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def tokenize(self, caption: str) -> torch.Tensor:
        tokens = self.tokenizer(
            caption,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)   # [max_text_len]

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
        """Load from joekraper/openvid-latents (after augment.py)"""
        ds = load_dataset(
            "entropyspace/openvid-latents",
            split="train",
            streaming=True,
        )
        for row in ds:
            latent = self.deserialize(row["serialized_latent"])
            tokens = self.tokenize(row["caption"])
            yield self._make_batch(latent, tokens)

    def _iter_streaming(self):
        """Stream from fal + joekraper (before augment.py)"""
        # build caption lookup
        filter_ds = load_dataset(
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
            latent  = self.deserialize(row["serialized_latent"])
            tokens  = self.tokenize(caption_map[row["video"]])
            yield self._make_batch(latent, tokens)

    def _make_batch(self, latent, tokens):
        # latent: [C, T, H, W]
        C = latent.shape[0]

        # sample noise level t ∈ [0, 1]
        t = torch.rand(1)

        # create noisy latent (flatten C for consistency head)
        noise   = torch.randn(C) * t.item()
        x_noisy = latent.float().mean(dim=(1,2,3)) + noise  # [C]

        return {
            "latents": latent.float(),   # [C, T, H, W]
            "tokens":  tokens,           # [max_text_len]
            "x_noisy": x_noisy,          # [C]
            "t":       t,                # [1]
        }


def get_dataloader(
    use_merged: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
):
    from torch.utils.data import DataLoader
    dataset = VideoLatentDataset(use_merged=use_merged)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    print("Testing dataloader (streaming mode)...")
    ds = VideoLatentDataset(use_merged=False)

    for i, batch in enumerate(ds):
        print(f"latents: {batch['latents'].shape}")
        print(f"tokens:  {batch['tokens'].shape}")
        print(f"x_noisy: {batch['x_noisy'].shape}")
        print(f"t:       {batch['t'].shape}")
        print("✅ Dataloader works!")
        break
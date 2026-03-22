
import torch
from torch.utils.data import DataLoader
from data.dataloader import VideoLatentDataset

def test_fixed_dataloader():
    ds = VideoLatentDataset()
    
    # Mock some samples with different sizes
    sample1 = {
        "latent": torch.randn(16, 10, 32, 32),
        "caption": "test 1"
    }
    sample2 = {
        "latent": torch.randn(16, 5, 64, 64),
        "caption": "test 2"
    }
    sample3 = {
        "latent": torch.randn(16, 20, 16, 16),
        "caption": "test 3"
    }

    processed = []
    for s in [sample1, sample2, sample3]:
        ids, mask = ds.tokenize(s["caption"])
        out = ds._make_sample(s["latent"], ids, mask)
        processed.append(out)
        print(f"Processed shape: latents={out['latents'].shape}, x0={out['x0'].shape}")
        assert out['latents'].shape == (16, 15, 32, 32)
        assert out['x0'].shape == (16, 32, 32)

    # Test batching
    loader = DataLoader(processed, batch_size=3)
    batch = next(iter(loader))
    print(f"Batch shapes: latents={batch['latents'].shape}, x0={batch['x0'].shape}")
    print("✅ DataLoader fix verified!")

if __name__ == "__main__":
    test_fixed_dataloader()

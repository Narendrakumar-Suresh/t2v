
import torch
from torch.utils.data import DataLoader, IterableDataset

class MockDataset(IterableDataset):
    def __iter__(self):
        # Varying spatial sizes
        yield {"latents": torch.randn(16, 15, 32, 32), "x0": torch.randn(16, 32, 32)}
        yield {"latents": torch.randn(16, 15, 64, 64), "x0": torch.randn(16, 64, 64)}

def test():
    ds = MockDataset()
    loader = DataLoader(ds, batch_size=2)
    try:
        for batch in loader:
            print("Batch loaded successfully")
    except Exception as e:
        print(f"Caught expected crash: {e}")

if __name__ == "__main__":
    test()

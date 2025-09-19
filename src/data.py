import torch, numpy as np
from torch.utils.data import Dataset, DataLoader


class SyntheticMNIST(Dataset):
    def __init__(self, n: int = 60000, image_size: int = 28, num_classes: int = 10, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.x = rng.rand(n, 1, image_size, image_size).astype("float32")
        self.y = rng.randint(0, num_classes, size=(n,)).astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


def make_loaders(cfg):
    train = SyntheticMNIST(cfg["data"]["samples"], cfg["data"]["image_size"])
    val = SyntheticMNIST(cfg["data"]["val_samples"], cfg["data"]["image_size"], seed=2)
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    return (
        DataLoader(train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )


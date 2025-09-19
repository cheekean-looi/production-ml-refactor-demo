from __future__ import annotations
import os, random, time
import numpy as np
import torch
from rich.console import Console

console = Console()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_device(request: str = "auto") -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0


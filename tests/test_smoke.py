from src.model import TinyCNN
import torch


def test_forward():
    m = TinyCNN()
    y = m(torch.randn(2, 1, 28, 28))
    assert y.shape == (2, 10)


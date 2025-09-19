import torch, torch.nn as nn, torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64), fc_dim=128, num_classes=10):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * c2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


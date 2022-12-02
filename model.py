import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 3, 5, padding=2)
    def forward(self, x):
        y = nn.functional.relu(self.conv1(x))
        y = nn.functional.relu(self.conv2(y))
        y = self.conv3(y)
        return y
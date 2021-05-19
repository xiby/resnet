import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.fc1 = nn.Linear(20*20, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return out
if __name__ == "__main__":
    pass
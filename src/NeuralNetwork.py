import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))
        self.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-3)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def trainModel(self, dataloader: DataLoader):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            print(X.size())
            X, y = X.to(self.device), y.to(self.device)
            pred = self(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
    def testModel(self, dataloader: DataLoader):
        size = len(dataloader.dataset)
        self.eval()
        test_loss, correct = 0,0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):0.1f} %, Avg loss: {test_loss:>8f}\n")
        return correct

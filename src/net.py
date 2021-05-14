import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from resnet import ResNet, BasicBlock, Bottleneck

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

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

#model = torchvision.models.resnet50()
# model = ResNet50()
# print(model)

# 参数设置
if __name__ == "__main__":
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.manual_seed(random_seed)
    training_data = datasets.FashionMNIST(
        root = "../data/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="../data/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size_train)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test)
    epochs = 5
    model = NeuralNetwork()
    # for t in range(epochs):
    #     model.trainModel(train_dataloader)
    #     model.testModel(test_dataloader)
    # print("Done!")
    # torch.save(model.state_dict(), "../model/model.pth")
    # print("Saved Neural Network Model State to model.pth")

    resNet18Model = ResNet(BasicBlock, [2,2,2,2])
    correct1 = []
    for t in range(epochs):
        resNet18Model.trainModel(train_dataloader)
        correct1.append(resNet18Model.testModel(test_dataloader))
    
    torch.save(resNet18Model.state_dict(), "../model/res18.pth")
    print("Saved ResNet Model State to res18.pth")

    correct2 = []
    resNet50Model = ResNet(Bottleneck, [3, 4, 6, 3])
    for t in range(epochs):
        resNet50Model.trainModel(train_dataloader)
        correct2.append(resNet50Model.testModel(test_dataloader))
    torch.save(resMet50Model.state_dict(), "../model/res50.pth")
    print("Saved ResNet50 Model State to res50.pth")

    x = range(epochs)
    plt.figure(figsize = (20,8), dpi=80)
    plt.plot(x, correct1, label='res18', color='#F08080')
    plt.plot(x, correct2, label='res50', color='#DB7093', linestyle="--")
    _xtick_labels = ["{}".format(i + 1) for i in x]
    plt.xticks(x, _xtick_labels)
    plt.grid(alpha=0.4, linestyle=':')
    plt.show()


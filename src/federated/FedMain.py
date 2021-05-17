import sys
from .LearningClient import LearningClient
from .LearningServer import LearningServer
sys.path.append("..")
from NeuralNetwork import NeuralNetwork

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

import torchvision
import torch
import torch.nn as nn

def main():
    test_data = datasets.FashionMNIST(
        root = "/home/zhanglei/resnet/data/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testDataLoader = DataLoader(test_data, batch_size=1000)
    globalModel = NeuralNetwork()
    rounds = 5
    loss_fn = nn.CrossEntropyLoss()
    server = LearningServer(globalModel, loss_fn, testDataLoader, rounds)
    clientCount = 5

    for _ in range(clientCount):
        localModel = NeuralNetwork()
        training_data = datasets.FashionMNIST(
            root = "/home/zhanglei/resnet/data/",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        batch_size = 64
        dataloader = DataLoader(training_data, batch_size=batch_size)
        optimizer = torch.optim.SGD(localModel.parameters(), lr = 1e-3)
        client = LearningClient(server, localModel, dataloader, optimizer, loss_fn)
        server.addClient(client)
    server.trainLoop()
if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from resnet import ResNet, BasicBlock, Bottleneck
from FederatedDataSet import FederatedDataSet
from federated.FedMain import main
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

from NeuralNetwork import NeuralNetwork

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
    fl_training_data = FederatedDataSet(training_data, )
    fl_test_data = FederatedDataSet(test_data)
    train_dataloader = DataLoader(training_data, batch_size=batch_size_train)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test)
    epochs = 10
    # model = NeuralNetwork()
    # correct0 = []
    # for t in range(epochs):
    #     model.trainModel(train_dataloader)
    #     correct0.append(model.testModel(test_dataloader))
    # print("Done!")
    # torch.save(model.state_dict(), "../model/model.pth")
    # print("Saved Neural Network Model State to model.pth")
    # print(correct0)

    main()
    # resNet18Model = ResNet(BasicBlock, [2,2,2,2])
    # correct1 = []
    # for t in range(epochs):
    #     resNet18Model.trainModel(train_dataloader)
    #     correct1.append(resNet18Model.testModel(test_dataloader))
    
    # torch.save(resNet18Model.state_dict(), "../model/res18.pth")
    # print("Saved ResNet Model State to res18.pth")

    # # TODO find the reason why resnet50 is not better than resnet18
    # correct2 = []
    # resNet50Model = ResNet(Bottleneck, [3, 4, 6, 3])
    # for t in range(epochs):
    #     resNet50Model.trainModel(train_dataloader)
    #     correct2.append(resNet50Model.testModel(test_dataloader))
    # torch.save(resNet50Model.state_dict(), "../model/res50.pth")
    # print("Saved ResNet50 Model State to res50.pth")

    # print(correct1)
    # print(correct2)
    # x = range(epochs)
    # plt.figure(figsize = (20,8), dpi=80)
    # plt.plot(x, correct1, label='res18', color='#F08080', linestyle=".")
    # plt.plot(x, correct2, label='res50', color='#DB7093', linestyle="--")
    # plt.plot(x, correct0, label="nural", color='#1a75ff', linestyle='-')
    # _xtick_labels = ["{}".format(i + 1) for i in x]
    # plt.xticks(x, _xtick_labels)
    # plt.grid(alpha=0.4, linestyle=':')
    # plt.show()

from Server import Server
from Client import Client
from Router import Router
from Model import Model
from datas.SemiFlDataset import SemiFlDataset
from torchvision import datasets
from torch.utils.data import DataLoader

import torchvision
import torch
import time

import torch.nn as nn

if __name__ == "__main__":
    base = datasets.MNIST(
        root="../../data/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    semiDataset = SemiFlDataset(base, chooseLabel = 1)
    dataloader = DataLoader(semiDataset, batch_size=64)
    globalModel = Model()
    print("开始加载模型到gpu中")
    tic = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    globalModel.to(device)
    toc = time.time()
    print("模型加载到gpu完成，耗时：" + str(toc - tic))
    server = Server(globalModel.state_dict(), globalModel)
    routerCount = 10
    clients = 10
    for i in range(routerCount):
        router = Router(i)
        for j in range(clients):
            clientModel = Model()
            clientModel.to(device)
            # 每个router只有一个label
            clientDataset = SemiFlDataset(base, chooseLabel = j)
            dataloader = DataLoader(clientDataset, batch_size=20)
            client = Client(clientModel, dataloader, torch.optim.SGD(clientModel.parameters(), lr = 1e-2), nn.CrossEntropyLoss(), j, i)
            router.addClient(client)
        server.addRouter(router)
    server.trainLoop(50)
    testDataset = datasets.MNIST(
        root="../../data/",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    server.testModel(DataLoader(testDataset, batch_size=1000), nn.CrossEntropyLoss())

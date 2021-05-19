import random

from torch.utils.data import Dataset

class SemiFlDataset(Dataset):
    def __init__(self, baseDataset, size=600, transform=None):
        super().__init__()
        self.size = min(size, len(baseDataset))
        self.transform = transform
        indexList = random.sample(range(len(baseDataset)), self.size)
        self.datas = []
        self.labels = []
        for index in indexList:
            data, label = baseDataset[index]
            self.datas.append(data)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return self.size
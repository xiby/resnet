import random

from torch.utils.data import Dataset

class SemiFlDataset(Dataset):
    def __init__(self, baseDataset, size=600, transform=None, chooseLabel = None):
        super().__init__()
        self.size = min(size, len(baseDataset))
        self.transform = transform
        indexList = []
        if chooseLabel is None:
            indexList = random.sample(range(len(baseDataset)), self.size)
        else:
            indexList = random.sample(range(len(baseDataset)), len(baseDataset))
        self.datas = []
        self.labels = []
        count = 0
        for index in indexList:
            if count >= self.size:
                break
            data, label = baseDataset[index]
            if chooseLabel is None or chooseLabel == label:
                self.datas.append(data)
                self.labels.append(label)
                count += 1

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return self.size
'''
联邦学习下的数据集
继承自torch的dataset
'''
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import tensor

from PIL import Image
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FederatedDataSet(Dataset):
    def __init__(self, source, size = 6000):
        super().__init__()
        row_data = source.train_data
        row_labels = source.train_labels
        sampler = RandomSampler(row_data, nums=size)
        sample_list = []
        label_list = []
        count = 0
        for index in sampler:
            sample_list.append(row_data[index].numpy().tolist())
            label_list.append(row_labels[index].numpy().tolist())
            count += 1
            if count >= size:
                break
        self.train_data = torch.tensor(sample_list, dtype=torch.float)
        self.train_labels = torch.tensor(label_list, dtype=torch.long)
    def __len__(self):
        return len(self.train_data)
    def __getitem__(self, index):
        return self.train_data[index], self.train_labels[index]
class Sampler():
    '''
    采样基类
    '''
    def __init__(self, data_source):
        pass
    
    def __iter__(self):
        '''
        采样接口
        '''
        raise NotImplementedError

class RandomSampler(Sampler):
    '''
    随机采样器
    '''
    def __init__(self, datasource, replacement = False, nums = None):
        self.datasource = datasource
        self.replacement = replacement
        self._count = nums
    @property
    def count(self):
        if self._count is None:
            return len(self.datasource)
        return self._count
    
    def __len__(self):
        return self.count
    def __iter__(self):
        n = len(self.datasource)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.count,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

if __name__ == "__main__":
    training_data = datasets.FashionMNIST(
        root = "../data/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    training_row_data = training_data.train_data
    training_row_label = training_data.train_labels
    unloader = transforms.ToPILImage()
    image = training_row_data[0].cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    logger.info(training_row_data.size())
    logger.info(training_row_label.size())

'''
联邦学习下的数据集，继承自
'''
import torch
from torch.utils.data import Dataset
class FederatedDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
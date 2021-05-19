import json
import random

import torch
import torchaudio
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, MFCC
from torch.nn.functional import softmax

class VideoDataset(Dataset):
    def __init__(self, root, mode, size = None):
        super().__init__()
        mfcc = MelSpectrogram()
        self.root = root
        self.datas = []
        self.labels = []
        self.mode = mode
        jsonFileName = ""
        if self.mode == "train":
            # 训练集
            jsonFileName = "train.json"
        elif self.mode == "test":
            # 测试集
            jsonFileName = "test.json"
        else:
            raise RuntimeError("worong mode")
        with open(self.root + "/" + jsonFileName, 'r') as load_f:
            jsonDatas = json.load(load_f)
            if size is None:
                self.size = len(jsonDatas)
            else:
                self.size = min(size, len(jsonDatas))
            indexList = random.sample(range(len(jsonDatas)), self.size)
            for index in indexList:
                jsonData = jsonDatas[index]
                self.labels.append(jsonData['is_hotword'])
                filePath = self.root + "/" + jsonData['audio_file_path']
                # 将音频文件转成Tensor
                tensor, sample_rate = torchaudio.load(filePath)
                mel = MFCC().forward(tensor)[0]
                shape = mel.shape
                half = shape[1]//2
                mel = mel[:,(half - 16):(half + 16)]
                self.datas.append(mel)
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.datas[index], self.labels[index]
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = self._make_layer(16, )
        self.fc1 = None
        self.fc2 = None
        self.sm = softmax
    
    def _layer_process(self, data, in_channel, out_channel, kernel_size=5):
        '''
        层次处理
        '''
        conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, dilation=1)
        out1 = conv1(data)

        conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, dilation=2)
        out2 = conv2(data)

        conv3 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, dilation=3)
        out3 = conv2(data)

        conv4 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, dilation=4)
        out4 = conv2(data)

        return torch.cat((out1, out2, out3, out4), 2)
    def forward(self, x):
        train_data = x.permute(0, 2, 1)
        out_layer1 = self._layer_process(train_data, 32, 64)

        out_layer2 = self._layer_process(out_layer1, 64, 64)

        out_layer3 = self._layer_process(out_layer2, 64, 128)

        out_layer4 = self._layer_process(out_layer3, 128, 256)

        out_layer5 = self._layer_process(out_layer4, 256, 128)

        out_fc1 = self.fc1(out_layer5)

        out_fc2 = self.fc2(out_fc1)

        out = self.sm(out_fc2, dim=1)

        return out


        

if __name__ == "__main__":

    # dataSet = VideoDataset(root="D://learn/datas/hey_snips_fl_amt/", mode="train", size = 640)
    # dataloader = DataLoader(dataSet, batch_size=64, shuffle=True)
    # netModel = Net()
    # for batch, (X, y) in enumerate(dataloader):
    #     print(X.shape)
    #     netModel(X)
    #     break

    inputdata = torch.randn(64, 128, 31412)
    inputdata = torch.nn.ELU()(inputdata)
    linear = nn.Linear(31412, 64)
    out = linear(inputdata)
    print(out.shape)

    m = nn.Softmax(dim=2)
    out2 = m(out)
    print(out2.shape)

    # inputData = torch.randn(64, 1, 40, 32)
    # inputData = inputData.permute(0, 1, 3, 2)
    # print(inputData.shape)
    # conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, dilation=1)
    # out1 = conv1(inputData)
    # print(out1.shape)
    # conv2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, dilation=2)
    # out2 = conv2(inputData)
    # print(out2.shape)


    # conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=2)
    # out2 = conv2(inputData)
    # print(out2.shape)
    # conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=3)
    # out3 = conv3(inputData)
    # print(out3.shape)
    # conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, dilation=4)
    # out4 = conv4(inputData)
    # print(out4.shape)
    # out = torch.cat((out1, out2, out3, out4), 2)
    # print(out.shape)
    # nn.Sequential()



    # conv2 = nn.Conv1d(in_channels=25, out_channels=64, kernel_size=16, dilation=2)
    # out = conv2(out)
    # print(out.size())
    # out = out.permute(0, 2, 1)
    # conv3 = nn.Conv1d(in_channels=34, out_channels=64, kernel_size=16, dilation=3)
    # out = conv3(out)
    # print(out.size())
    # out = out.permute(0, 2, 1)
    # conv4 = nn.Conv1d(in_channels=19, out_channels=64, kernel_size=16, dilation=4)
    # out = conv4(out)
    # print(out.size())
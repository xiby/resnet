'''
联邦学习客户端
'''
from .LearningServer import LearningServer
from torch.utils.data import DataLoader

import uuid

class LearningClient():
    '''
    联邦学习中的学习客户端抽象
    '''
    def __init__(self, server, model, dataloader, optimizer, loss_fn, batchSize = 1000, device = 'cpu'):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.size = len(dataloader.dataset)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batchSize = batchSize
        self.device = device
        self.id = str(uuid.uuid4())
        self.server = server
    def register(self, server: LearningServer):
        '''
        向中心注册学习客户端
        '''
        server.addClient(self)
    def trainCircle(self):
        '''
        整个训练的循环
        '''
        self._trainModel()
        self._uploadModel()
        pass
    def _trainModel(self):
        '''
        完成模型的训练
        '''
        for batch, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item()
                batch * len(X)
                print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
    def _uploadModel(self):
        '''
        完成模型上传
        '''
        self.server.gatherModel(self.id, self.model.state_dict())
        pass
    def updateModel(self, globalModel):
        '''
        完成模型更新
        '''
        self.model.load_state_dict(globalModel)

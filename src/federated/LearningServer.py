'''
联邦学习服务器
'''
from .LearningClient import LearningClient
import torchvision.models as models
class LearningServer():
    def __init__(self, model, rounds):
        super().__init__()
        self.clients = dict()
        self.params = dict()
        self.rounds = rounds
        # 初始模型
        self.initModel = model
        self.globalParam = self.initModel.state_dict()
    def addClient(self, client: LearningClient):
        self.clients[client.id] = client

    def gaterModels(self, id, params):
        self.params[id] = params
    def aggregate(self):
        '''
        聚合参数，并产生新模型，使用最简单的联邦平均
        '''
        count = 0
        sumParam = None
        for value in self.params.values():
            if value is not None:
                count += 1
                if sumParam is None:
                    sumParam = value
                else:
                    for var in sumParam:
                        sumParam[var] += value[var]
        for var in sumParam:
            sumParam[var] = sumParam[var]/count
        self.globalParam = sumParam
    def transportGlobalModel(self, client):
        '''
        向客户端传输全局模型
        '''
        client.updateModel(self.globalParam)
    
    def startTrainCircle(self):
        for value in self.clients.values():
            self.transportGlobalModel(value)
            value.trainCircle()
        self.aggregate()
    def trainLoop(self):
        for i in range(self.rounds):
            self.startTrainCircle()
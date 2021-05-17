'''
联邦学习服务器
'''
import torchvision.models as models
import torch
class LearningServer():
    def __init__(self, model, loss_fn, testDataLoader, rounds):
        super().__init__()
        self.clients = dict()
        self.params = dict()
        self.rounds = rounds
        # 初始模型
        self.globalModel = model
        self.globalParam = self.globalModel.state_dict()
        self.testDataLoader = testDataLoader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = loss_fn
    def addClient(self, client):
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
        self.globalModel.load_state_dict(self.globalParam)
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
        self.testModel()
    def testModel(self):
        '''
        测试全局模型
        '''
        size = len(self.testDataLoader.dataset)
        self.globalModel.eval()
        test_loss, correct = 0,0
        with torch.no_grad():
            for X, y in self.testDataLoader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.globalModel(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):0.1f} %, Avg loss: {test_loss:>8f}\n")
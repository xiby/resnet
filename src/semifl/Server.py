'''
聚合中心服务器
'''
import torch
class Server():
    def __init__(self, param, model):
        super().__init__()
        self.paramList = []
        self.routers = []
        self.param = param
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def addRouter(self, router):
        self.routers.append(router)

    def gatherParams(self, params):
        self.paramList.append(params)
    
    def aggrate(self):
        '''
        聚合参数
        '''
        sumParam = None
        for param in self.paramList:
            if sumParam is None:
                sumParam = param
                continue
            else:
                for var in param:
                    sumParam[var]+=param[var]
        for var in param:
            sumParam[var] = sumParam[var]/len(self.paramList)
        self.param = sumParam
        self.paramList = []
    
    def _trainProcess(self):
        for router in self.routers:
            router.applyParam(self.param)
            router.startTrain()
            self.gatherParams(router.getParam())
        self.aggrate()
    
    def trainLoop(self, rounds):
        for _ in range(rounds):
            self._trainProcess()
    def testModel(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        self.model.load_state_dict(self.param)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):0.1f} %, Avg loss: {test_loss:>8f}\n")
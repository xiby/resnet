'''
cluster中的客户端
'''
import torch
class Client():
    def __init__(self, model, dataloader, optimizer, loss_fn, id=None, router_id=None):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.id = id
        self.router_id = router_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
    def trainModel(self, epoch):
        '''
        进行模型训练
        '''
        size = len(self.dataloader.dataset)
        for batch, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"epoch: {epoch} client: {self.router_id}--{self.id} loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
    def getParam(self):
        return self.model.state_dict()
    def loadParam(self, params):
        self.model.load_state_dict(params)
    

'''
集群内部的中心节点
'''

class Router():
    def __init__(self, id = None):
        super().__init__()
        self.clients = []
    def applyParam(self, param):
        self.params = param
    def getParam(self):
        return self.params
    def startTrain(self):
        '''
        开始一次训练
        '''
        for client in self.clients:
            client.loadParam(self.params)
            client.trainModel()
            self.params = client.getParam()
    def addClient(self, client):
        self.clients.append(client)
    

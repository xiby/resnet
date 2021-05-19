'''
聚合中心服务器
'''

class Server():
    def __init__(self, param, model):
        super().__init__()
        self.paramList = []
        self.routers = []
        self.param = param
        self.model = model
    
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
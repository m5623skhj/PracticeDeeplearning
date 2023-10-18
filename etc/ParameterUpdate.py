class SGD:
    def __init__(self, lr=0.01):
        self.learningRate = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learningRate * grads[key]
            

class Momentum:
    def __init__(self, learningRate=0.01, momentum=0.9):
        self.learningRate = learningRate;
        self.momentum = momentum;
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.learningRate * grads[key]
            params[key] += self.v[key]
            
            
class AdaGrad:
    def __init__(self, learningRate = 0.01):
        self.learningRate = learningRate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.learningRate * grads[key] / (np.sqrt(self.h[key]) + 1e + 7)
            

# class Adam:
# like Momentom + AdaGrad
    
    
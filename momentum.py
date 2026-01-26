#自适应调整梯度下降幅度，如果梯度下降快，则权重调整要动态变小
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= slef.lr*grads[key] / (np.sqrt(self.h[key]) + 1e-7)

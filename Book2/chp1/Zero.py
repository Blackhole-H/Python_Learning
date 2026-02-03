import numpy as np

class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None
        #函数是变量的creator，如果存在没有creator的变量，通常为用户给出的变量
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    
    #防止勿用基类，必须由子类完成，轻量级抽象方法
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    # 参数gy是从输出传播而来的导数
    def backward(self, gy):
        x = self.input.data
        gx = x * 2 * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = gy * np.exp(x)
        return gx

data = np.array(3.0)
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
   
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)



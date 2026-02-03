import numpy as np
from function import identity_function, sigmoid

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    x1 = np.dot(x, W1) + b1
    z1 = sigmoid(x1)
    x2 = np.dot(x1, W2) + b2
    z2 = sigmoid(x2)
    x3 = np.dot(x2, W3) + b3
    z3 = identity_function(x3)

    return z3

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
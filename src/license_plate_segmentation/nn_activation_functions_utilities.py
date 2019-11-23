import numpy as np

'''
    Define activation function section:
        1. relu 
        2. tanh 
        3. softmax -- used in the last layer
'''
def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims= True)
    
    return x_exp/x_sum

'''
    Derivative of activation function section:
        1. relu
        2. tanh
        3. softmax
        4. cross_entropy with softmax
'''
def relu_backward(dA):
    return (dA > 0).astype(int)

def tanh_backward(dA):
    return 1 - tanh(dA)*tanh(dA)

def softmax_backward(dA):
    s = dA.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def cross_entropy_softmax(x,y):
    m = y.shape[1]
    return 1./m * (x - y)

import numpy as np

'''
    Cost function section：
        1. Cross-entropy Cost
        2. Quadratic cost (MSE)
'''
def crossEntropy_cost(x,y):
    '''
        Implement the cross entropy cost function
        Arguments: x -- output from fully connected layer
                   y -- ground truth label
                   x and y have the same shape
        Return: crossEntropyCost -- value of the cost function
    '''
    m = y.shape[1]
    cost = -(np.multiply(np.log(x),y) + np.multiply(np.log(1-x),1-y))
    crossEntropyCost = 1./m * np.sum(cost)

    return crossEntropyCost

def mse_cost(x,y):
    '''
        Implement the MSE function
        Arguments: x -- output from fully connected layer
                   y -- ground truth label
                   x and y have the same shape
        Return: cost -- value of the cost function
    '''
    m = y.shape[1]
    cost = 1/m * np.sum(np.multiply(y-x,y-x), axis = 1)

    return cost
import numpy as np

'''
    Optimizers section: Implement two optimizers.
    1. Momentum optimizer
    2. Adam optimizer
'''
def momentum_init(parameters):
    '''
        Momentum optimizer
        Argument: 
            parameters (dictionary type) -- with keys: 'W1','b1',...,'Wn','bn'
        Return:
            momentumDict (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
    '''
    momentumDict = {}
    
    for l in range(len(parameters)//2):
        momentumDict["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        momentumDict["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return momentumDict

def adam_init(parameters):
    '''
        Adam optimizer
        Argument: 
            parameters (dictionary type) -- with keys: 'W1','b1',...,'Wn','bn'
        Returns: v(the exponentially weighted average of the gradient)
                 s(the exponentially weighted average of the squared gradient)
            v (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
            s (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
    '''
    v = {}
    s = {}
    for l in range(len(parameters)//2):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v,s
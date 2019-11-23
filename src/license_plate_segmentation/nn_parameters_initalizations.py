import numpy as np

'''
    initalization function sections: implement two initalization functions:
    1. he initalization
    2. random initalization
'''
def he_init(layerDims):
    '''
        Implement the weight and bias initalzation function use HE init.
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
        Return: [784,200,100,10]
            dicts (dictionary type) -- contains the weight and bias: 'W1', 'b1', 'W2', 'b2', ... , 'Wn', 'bn'
                                                               W1 -- weight matrix of shape (layerDims[1], layerDims[0])
                                                               b1 -- bias vector of shape (layerDims[1], 1) 
                                                               W2 -- weight matrix of shape (layerDims[2], layerDims[1])
                                                               b2 -- bias vector of shape (layerDims[2], 1) 
                                                               W3 -- weight matrix of shape (layerDims[3], layerDims[2])
                                                               b3 -- bias vector of shape (layerDims[3], 1) 
    '''
    np.random.seed(3)                   # random number generator
    dicts = {}

    for l in range(1, len(layerDims)):
        dicts['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1])*np.sqrt(2/layerDims[l-1])
        dicts['b' + str(l)] = np.zeros((layerDims[l],1))
 
    return dicts

def random_init(layerDims):
    '''
        Implement the weight and bias initalzation function use random init.
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
        Return:
            dicts (dictionary type) -- contains the weight and bias: 'W1', 'b1', 'W2', 'b2', ... , 'Wn', 'bn'
                                                               W1 -- weight matrix of shape (layerDims[1], layerDims[0])
                                                               b1 -- bias vector of shape (layerDims[1], 1) 
    '''
    np.random.seed(3)                   # random number generator
    dicts = {}

    for l in range(1, len(layerDims)):
        dicts['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1])*0.03
        dicts['b' + str(l)] = np.zeros((layerDims[l],1))
 
    return dicts

def parameters_init(layerDims,initialization):
    '''
        Implement the weight and bias initalzation function
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
            initialzation (string type) -- method used to initialze the weight and bias.
                                         1. initialzation = 'random'.   2. initialzation = 'he'
        Return:
            parameters (dictionary type) -- contains the weight and bias
    '''
    parameters = {}
    
    # Check whether initialization is valid
    assert(initialization == 'he' or initialization == 'random')    # Error: unrecognize initalization
 
    if(initialization == 'he'):
        parameters = he_init(layerDims)
    elif(initialization == 'random'):
        parameters = random_init(layerDims)

    return parameters
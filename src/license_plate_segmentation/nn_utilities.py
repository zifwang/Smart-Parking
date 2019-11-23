import numpy as np
import math

def random_mini_batches(x, y, mini_batch_size = 100, seed = 0):
        '''
            Implement the function to create random minibatches from input train_x and train_y
            Arguments:
                x -- Input training data: train_x.shape == (input size, number of samples)
                y -- GroundTruth Training data: train_y.shape == (output size, number of samples)
                mini_batch_size -- size of the mini-batches, integer
            Returns: 
                mini_batch (a list): (mini_batch_x, mini_batch_y)
        '''
        mini_batches = []       # return list
        np.random.seed(seed)
        # number of training samples 
        numSamples = x.shape[1]  
        # output data shape in one sample
        ySize = y.shape[0]

        # Data shuffle
        permutation = list(np.random.permutation(numSamples))
        shuffled_X = x[:, permutation]
        shuffled_Y = y[:, permutation].reshape((ySize,numSamples))
        
        # number of complete mini batches
        num_complete_minibatches = math.floor(numSamples/mini_batch_size)
        for k in range(0,num_complete_minibatches):
            mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handle reset of mini batch (last mini_batch < mini_batch_size)
        if numSamples % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : numSamples]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : numSamples]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


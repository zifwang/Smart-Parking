import os
import numpy as np
import random
import math
import h5py
from nn_activation_functions_utilities import relu, tanh, softmax, relu_backward, tanh_backward, softmax_backward, cross_entropy_softmax
from nn_cost_functions import crossEntropy_cost, mse_cost
from nn_parameters_initalizations import he_init, random_init, parameters_init
from nn_optimizers_initalizations import momentum_init, adam_init
from nn_utilities import random_mini_batches
from nn_parameters_update import update_parameters_gd, update_parameters_momentum, update_parameters_adam

class neural_network_model(object):
    def __init__(self):
        self.parameters = {}             # parameters dictionary: weight + bias
        self.costs = []                  # list to track the cost
        self.activations_functions = []  # list of activation functions 
        self.number_of_layers = 0        # nn size

    def __forward_propagation(self,x,parameters,activations):
        '''
            This function is for the forward propagation
            Arguments:
                x -- input dataset(in shape(inputSize,numOfSamples))
                parameters -- weight and bias
                activations -- a list of activation methods 
            Returns:
                cache_dicts -- contains all outputes of wx+b, outputes of activation, weightes, and biases
                keys(z,a,W,b)
        '''
        cache_dicts = {}
        cache_dicts['a0'] = x
        a = x
        for i in range (len(parameters)//2):
            z = np.dot(parameters["W"+str(i+1)],a) + parameters["b"+str(i+1)]   # linear
            cache_dicts["z"+str(i+1)] = z          # append output of wx+b
            # z to activation function 
            if(activations[i] == 'relu'):
                a = relu(z)
                cache_dicts["a"+str(i+1)] = a
            if(activations[i] == 'tanh'):
                a = tanh(z)
                cache_dicts["a"+str(i+1)] = a
            if(activations[i] == 'softmax'):
                a = softmax(z)
                cache_dicts["a"+str(i+1)] = a
            cache_dicts["W"+str(i+1)] = parameters["W"+str(i+1)]
            cache_dicts["b"+str(i+1)] = parameters["b"+str(i+1)]
        
        return cache_dicts["a"+str(len(parameters)//2)],cache_dicts

    def __backward_propagation(self,x,y,cache_dicts,activations):
        '''
            This function is for the backward propagation
            Arguments:
                x -- input dataset(in shape(inputSize,numOfSamples))
                y -- ground truth
                cache_dicts -- cache_dicts output from forward propagation
                activations -- a list of activation methods 
            Returns:
                gradients -- a gradient dictionary
        '''
        gradients = {}
        for i in range(len(activations)-1,-1,-1):
            if(activations[i] == 'softmax'):
                dz = cross_entropy_softmax(cache_dicts["a"+str(i+1)],y)
                dW = np.dot(dz, cache_dicts["a"+str(i)].T)
                db = np.sum(dz,axis=1, keepdims=True)
                gradients["dz"+str(i+1)] = dz
                gradients["dW"+str(i+1)] = dW
                gradients["db"+str(i+1)] = db
            if(activations[i] == 'relu'):
                da = np.dot(cache_dicts["W"+str(i+2)].T,gradients["dz"+str(i+2)])
                dz = np.multiply(da,relu_backward(cache_dicts["a"+str(i+1)]))
                dW = np.dot(dz, cache_dicts["a"+str(i)].T)
                db = np.sum(dz,axis=1, keepdims=True)
                gradients["da"+str(i+1)] = da
                gradients["dz"+str(i+1)] = dz
                gradients["dW"+str(i+1)] = dW
                gradients["db"+str(i+1)] = db
            if(activations[i] == 'tanh'):
                da = np.dot(cache_dicts["W"+str(i+2)].T,gradients["dz"+str(i+2)])
                dz = np.multiply(da,tanh_backward(cache_dicts["a"+str(i+1)]))
                dW = np.dot(dz, cache_dicts["a"+str(i)].T)
                db = np.sum(dz,axis=1, keepdims=True)
                gradients["da"+str(i+1)] = da
                gradients["dz"+str(i+1)] = dz
                gradients["dW"+str(i+1)] = dW
                gradients["db"+str(i+1)] = db

        return gradients

    '''
        prediction method
    '''
    def predict(self,x,parameters,activations):
        '''
            Predict the label of a single test example (image).
            Arguments:
                x : numpy.array
            Returns: int
                Predicted label of example (image).
        '''
        a3, _ = self.__forward_propagation(x,parameters,activations)
        return a3

    '''
        Validation method
    '''
    def validation(self,x,y,parameters,activations):
        predict_result = self.predict(x,parameters,activations)
        numCorrect = 0
        y = y.T
        predict_result = predict_result.T
        for i in range(y.shape[0]):
            if(np.argmax(y[i]) == np.argmax(predict_result[i])):
                numCorrect = numCorrect + 1
        return numCorrect/y.shape[0]

    '''
        Training accuracy
    '''
    def train_accuracy(self,train_result,groundTruth):
        numCorrect = 0
        groundTruth = groundTruth.T
        train_result = train_result.T
        for i in range(groundTruth.shape[0]):
            if(np.argmax(groundTruth[i]) == np.argmax(train_result[i])):
                numCorrect = numCorrect + 1

        return numCorrect/groundTruth.shape[0]

    '''
        Training process
    '''
    def model_train(self, X, Y, x_val, y_val, layersDims, activations, initialization, optimizer, learning_rate = 0.0007, learning_rate_decay = True, mini_batch_size = 100, beta = 0.9,
            beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 50, verbose = True):
        '''
        '''
        # L = len(layersDims)       # number of layers in nn
        train_accuracies = []       # list to track the train accuracies
        val_accuracies = []         # list to track the val accuracies
        t = 0                       # adam t parameter
        seed = 10
        self.activations_functions = activations
        self.number_of_layers = layersDims
        self.__optimizer = optimizer
        self.__learning_rate = learning_rate
        self.__learning_rate_decay = learning_rate_decay
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon

        # Initialize parameters
        self.parameters = parameters_init(layersDims,initialization)

        # Initialize the optimizer
        if optimizer == "gd":
            pass # no initialization required for gradient descent
        elif optimizer == "momentum":
            v = momentum_init(self.parameters)
        elif optimizer == "adam":
            v, s = adam_init(self.parameters)

        for epoch in range(num_epochs):
            # Define the random minibatches. 
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

            if(learning_rate_decay and epoch == num_epochs//2):
                learning_rate = learning_rate/2

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                train_result, caches = self.__forward_propagation(minibatch_X, self.parameters,activations)

                # Compute cost
                cost = crossEntropy_cost(train_result, minibatch_Y)

                # Backward propagation
                grads = self.__backward_propagation(minibatch_X, minibatch_Y, caches, activations)

                # Update parameters
                if optimizer == "gd":
                    self.parameters = update_parameters_gd(self.parameters, grads, learning_rate)
                elif optimizer == "momentum":
                    self.parameters, v = update_parameters_momentum(self.parameters, grads, v, beta, learning_rate)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    self.parameters, v, s = update_parameters_adam(self.parameters, grads, v, s,
                                                                t, learning_rate, beta1, beta2, epsilon)

            # Calculate train_accuracy and val_accuracy after each epoch
            train_accuracy = self.validation(X,Y,self.parameters,activations)
            train_accuracies.append(train_accuracy)
            val_accuracy = self.validation(x_val,y_val,self.parameters,activations)
            val_accuracies.append(val_accuracy)
            self.costs.append(cost)

            if(verbose):
                # Print the cost 
                print("Epoch %i/%i"%(epoch,num_epochs))
                print("-loss: %f - training_acc: %f - validation_acc: %f"%(cost,train_accuracy,val_accuracy))

        history = {}
        history['train_accuracies'] = train_accuracies
        history['val_accuracies'] = val_accuracies

        return history

    def save_model(self, filename='model.npy'):
        """
        """
        model_dictionary = {'parameters':self.parameters, 'activations':self.activations_functions, 'num_layers':self.number_of_layers}
        np.save(filename,model_dictionary)


    def load_model(self, filename='model.npy'):
        """
        """
        nnModel = np.load(filename).item()
        self.parameters = nnModel['parameters']
        self.activations_functions = nnModel['activations']
        self.number_of_layers = nnModel['num_layers']


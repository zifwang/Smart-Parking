import numpy as np

'''
    Update paramenter section:
        1. Stochastic Gradient Descent
        2. Momentum optimizer Gradient Descent
        3. Adam optimizer Gradient Descent
'''
def update_parameters_gd(parameters,gradients,learning_rate):
    '''
        Stochastic Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            learning_rate (double type): learning rate
        returns:
            parameters (dictionary type): contains updated weight and bias
    '''
    for l in range(len(parameters)//2):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradients['db' + str(l+1)]
    
    return parameters

def update_parameters_momentum(parameters, gradients, momentumDict, beta, learning_rate):
    '''
        Momentum optimizer Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            momentumDict (dictionary type): contains current velocities
            beta: the Momentum Parameter
            learning_rate (double type): learning rate
        returns:
            parameters (dictionary type): contains updated weight and bias
            momentumDict (dictionary type): contains updated velocities
    '''
    for l in range(len(parameters)//2):
        # velocities
        momentumDict["dW" + str(l+1)] = beta*momentumDict["dW" + str(l+1)]+(1-beta)*gradients['dW' + str(l+1)]
        momentumDict["db" + str(l+1)] = beta*momentumDict["db" + str(l+1)]+(1-beta)*gradients['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*momentumDict["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*momentumDict["db" + str(l+1)]
    
    return parameters, momentumDict

def update_parameters_adam(parameters, gradients, v, s, t,
                           learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    '''
        Momentum optimizer Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            v (dictionary type): contains gradient
            s (dictionary type): contains squared gradient
            t: time
            beta1: Exponential decay hyperparameter for the first moment estimates 
            beta2: Exponential decay hyperparameter for the second moment estimates 
            learning_rate (double type): learning rate
            epsilon -- hyperparameter preventing division by zero in Adam updates
        returns:
            parameters (dictionary type): contains updated weight and bias
            v (dictionary type): contains updated gradient
            s (dictionary type): contains updated squared gradient
    '''
    v_bias_correction = {}
    s_bias_correction = {}

    for l in range(len(parameters)//2):
        # Update gradient and square gradient
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * gradients['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * gradients['db' + str(l+1)]
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * gradients['dW' + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * gradients['db' + str(l+1)]**2
        # Compute the bias corrections of v and s
        v_bias_correction["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-beta1 ** t)
        v_bias_correction["db" + str(l+1)] = v["db" + str(l+1)] / (1-beta1 ** t)
        s_bias_correction["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2 ** t)
        s_bias_correction["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2 ** t)

        # Update the parameter
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_bias_correction["dW" + str(l+1)] / (s_bias_correction["dW" + str(l+1)]**0.5 + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_bias_correction["db" + str(l+1)] / (s_bias_correction["db" + str(l+1)]**0.5 + epsilon)
        
    return parameters, v, s
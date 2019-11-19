import numpy as np
import cv2
import os
import nn_model
from training_data_prepare import generate_training_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    if os.path.exists('data.npy') and os.path.exists('labels.npy'):
        data = np.load('data.npy')
        labels = np.load('labels.npy')
    else:
        data, labels = generate_training_data('/Users/zifwang/Desktop/Smart Parking/data/dataset_character_digit')
    x_train, x_validation, y_train, y_validation = train_test_split(data, labels, test_size=0.1, random_state=42)

    print(x_train.shape)
    print(x_validation.shape)
    print(y_train.shape)
    print(y_validation.shape)

    # x_train = x_train.T
    # x_validation = x_validation.T
    # y_train = y_train.T
    # y_validation = y_validation.T
    
    
    
    # my_NN = nn_model.neural_network_model()
    # # Define layers and activations here
    # layers_dims = [x_train.shape[0], 200, 100, 10]
    # activations = ['tanh', 'tanh', 'softmax']
    # hist = my_NN.model_train(x_train, y_train, x_validation, y_validation, layers_dims, 
    #                          activations, initialization = 'he', optimizer = "adam",
    #                          learning_rate = 0.001, learning_rate_decay = True, mini_batch_size = 100, 
    #                          beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 4, verbose = True)

    # # plot the accuracy
    # plt.plot(hist['train_accuracies'])
    # plt.plot(hist['val_accuracies'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # # little code to show the figure and predict values
    # x_1 = x_train[0]
    # x_1 = np.reshape(x_1,(1,784))
    # result,_ = my_NN.predict(x_1.T,my_NN.parameters,my_NN.activations_functions)
    # print(result.shape)
    # print("The digit in the following figure is: ",np.argmax(result.T))
    # # Show digit
    # plt.imshow(x_train[0].reshape(28,28))
    # plt.title('Correct Figure')
    # plt.show()

import numpy as np
import nn_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
    Data Preparation:
        In this training MLP: use mnist_traindata.hdf5 file which contains 60,000 images in the key 'xdata',
        and their corresponding labels in the key 'ydata'. 
        Split them into 50,000 images for training and 10,000 for validation.
        Mini_batches method
        test_data method
'''
def dataPrep(filename):
    '''
        Implement the function to read in the data file.
        Keys of the data file: 'xdata' and 'ydata'
        Argument: the pwd/filename of the object
        Returns: x_train, x_validation, y_train, y_validation
    '''
    mnist_traindata = h5py.File(filename,'r')
    keys = list(mnist_traindata.keys())
    print(keys)
    xData = np.asarray(mnist_traindata[keys[0]])        # xdata is in the keys[0]
    yData = np.asarray(mnist_traindata[keys[1]])        # xdata is in the keys[1]
    x_train, x_validation, y_train, y_validation = train_test_split(xData,yData,
                                                                    test_size = 0.16666,
                                                                    random_state = 42)

    return x_train, x_validation, y_train, y_validation

if __name__ == "__main__":
    my_NN = nn_model.neural_network_model()
    x_train, x_validation, y_train, y_validation = dataPrep('/Users/zifwang/Desktop/Smart Parking/data/mnist.hdf5')
    x_train = x_train.T
    x_validation = x_validation.T
    y_train = y_train.T
    y_validation = y_validation.T
    # Define layers and activations here
    layers_dims = [x_train.shape[0], 200, 100, 10]
    activations = ['tanh', 'tanh', 'softmax']
    hist = my_NN.model_train(x_train, y_train, x_validation, y_validation, layers_dims, 
                             activations, initialization = 'he', optimizer = "adam",
                             learning_rate = 0.001, learning_rate_decay = True, mini_batch_size = 100, 
                             beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 4, verbose = True)

    # plot the accuracy
    plt.plot(hist['train_accuracies'])
    plt.plot(hist['val_accuracies'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # little code to show the figure and predict values
    x_1 = x_train[0]
    x_1 = np.reshape(x_1,(1,784))
    result,_ = my_NN.predict(x_1.T,my_NN.parameters,my_NN.activations_functions)
    print(result.shape)
    print("The digit in the following figure is: ",np.argmax(result.T))
    # Show digit
    plt.imshow(x_train[0].reshape(28,28))
    plt.title('Correct Figure')
    plt.show()

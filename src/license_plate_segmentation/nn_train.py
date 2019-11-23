import numpy as np
import cv2
import os
import nn_model
from training_data_prepare import generate_training_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


if __name__ == "__main__":

    if os.path.exists('data.npy') and os.path.exists('labels.npy'):
        data = np.load('data.npy')
        labels = np.load('labels.npy')
    else:
        data, labels = generate_training_data('/Users/zifwang/Desktop/Smart Parking/data/dataset_character_digit')
    X_train, X_validation, Y_train, Y_validation = train_test_split(data, labels, test_size=0.1, random_state=42)
    
    x_train = np.resize(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    x_validation = np.resize(X_validation,(X_validation.shape[0],X_validation.shape[1]*X_validation.shape[2]))
    
    # CNN change shape (28,28,1)
    X_train = X_train.reshape(X_train.shape[0],28,28,1)
    X_validation = X_validation.reshape(X_validation.shape[0],28,28,1)

    
    x_train = x_train.T
    x_validation = x_validation.T
    y_train = Y_train.T
    y_validation = Y_validation.T
    
    """
        My MLP 
    """
    """
    my_NN = nn_model.neural_network_model()
    # Define layers and activations here
    layers_dims = [x_train.shape[0], 256, 64, 36]
    activations = ['tanh', 'tanh', 'softmax']
    hist = my_NN.model_train(x_train, y_train, x_validation, y_validation, layers_dims, 
                             activations, initialization = 'he', optimizer = "adam",
                             learning_rate = 0.001, learning_rate_decay = True, mini_batch_size = 100, 
                             beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 5, verbose = True)

    my_NN.save_model()

    # plot the accuracy
    plt.plot(hist['train_accuracies'])
    plt.plot(hist['val_accuracies'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # little code to show the figure and predict values
    newModel = nn_model.neural_network_model()
    newModel.load_model()

    x_train = x_train.T
    x_1 = x_train[0]
    x_1 = np.reshape(x_1,(1,784))
    result = newModel.predict(x_1.T,newModel.parameters,newModel.activations_functions)
    print(result.shape)
    alphabets_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D',
                      14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26: 'Q', 
                      27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35: 'Z'}
    print("The digit in the following figure is: ",alphabets_dict[np.argmax(result.T)])
    # Show digit
    plt.imshow(x_train[0].reshape(28,28))
    plt.title('Correct Figure')
    plt.show()
    """

    """ 
        CNN by keras
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train,validation_data=(X_validation, Y_validation), epochs=20, batch_size=64)

    model.save('cnn_classifier.h5')
    # Visualization
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
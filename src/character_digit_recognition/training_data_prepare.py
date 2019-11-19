import numpy as np
import cv2
import os
import pickle
from sklearn.preprocessing import OneHotEncoder

def get_training_data(file_path):
    train_x = []
    train_y = []

    for dirs in os.listdir(file_path):
        for dir in dirs:
            for filename in os.listdir(file_path + '/' + dir):
                if filename.endswith('.jpg'):
                    image = cv2.imread(file_path+'/'+dir+'/'+filename)
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    train_x.append(image_gray)
                    train_y.append(dir)
    
    pickle.dump(train_x, open('data.pickle', 'wb'))
    pickle.dump(train_y, open('labels.pickle', 'wb'))

    return train_x, train_y

def generate_training_data(file_path):

    if os.path.exists('data.pickle') and os.path.exists('labels.pickle'):
        d = open("data.pickle","rb")
        l = open("labels.pickle","rb")
        data = pickle.load(d)
        labels = pickle.load(l)
    else:
        data, labels = get_training_data(file_path)

    alphabets_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D',
                      14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26: 'Q', 
                      27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35: 'Z'}
    alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    labels_class = []
    for alphabet in alphabets:
        labels_class.append([alphabet])
    
    # One hot encoding
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
    one_hot_encoder.fit(labels_class)
    labels_list = []
    for label in labels:
        labels_list.append([label])

    one_hot_encoded_labels = one_hot_encoder.transform(labels_list).toarray()
    data = np.array(data)

    np.save('data.npy',data)
    np.save('labels.npy',one_hot_encoded_labels)

    return data, one_hot_encoded_labels

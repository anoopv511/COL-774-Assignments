import numpy as np
import os
from functions import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing.image import ImageDataGenerator

path = "../data/train/"
files = os.listdir(path)
x_train = np.array([],dtype=np.uint8).reshape(0,784)
y_train = np.array([],dtype=np.int64)
categories = {}
for idx,f in enumerate(files):
    data = np.load(path+f)
    x_train = np.concatenate((x_train,data),axis=0)
    y_train = np.concatenate((y_train,idx*np.ones(data.shape[0])),axis=0)
    categories[idx] = f.split(".")[0]
    
x_test = np.load("../data/test/test.npy")
y_train_categorical = to_categorical(y_train,num_classes=20)
print("Data loaded")

x_train = (x_train - np.mean(x_train, axis=1)[np.newaxis].T)/np.std(x_train, axis=1)[np.newaxis].T
x_test = (x_test - np.mean(x_test, axis=1)[np.newaxis].T)/np.std(x_test, axis=1)[np.newaxis].T
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

datagen = ImageDataGenerator()

datagen.fit(x_train)


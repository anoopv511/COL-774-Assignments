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

cnn =Sequential()
cnn.add(Conv2D(64, kernel_size=(5, 5),activation="relu",strides=2,padding='same',input_shape=(28,28,1)))
cnn.add(Conv2D(64, kernel_size=(3, 3),activation="relu",padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(32, kernel_size=(5, 5),activation="relu",padding='same'))
cnn.add(Conv2D(32, kernel_size=(3, 3),activation="relu",padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(100, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Dense(20, activation='softmax'))
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

for i in range(5):
	cnn.fit(x_train,y_train_categorical,epochs=5,batch_size=128,verbose=1)

	name = 'cnn16_' + str(5*(i+1))
	print(name)
	cnn.save("../out/cnn/" + name + ".h5")
	cnn = load_model("../out/cnn/" + name + ".h5")

	pred = np.argmax(cnn.predict(x_test),axis=1)

	with open("../out/cnn/" + name + ".csv","w") as f:
	    f.write("ID,CATEGORY\n")
	    for idx, p in enumerate(pred):
	        f.write("{0},{1}\n".format(idx,categories[p]))
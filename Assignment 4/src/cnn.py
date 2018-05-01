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

# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     featurewise_std_normalization=False,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     data_format="channels_last")

# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)

# cnn = Sequential()
# cnn.add(Conv2D(32,(3,3),activation='linear',input_shape=(28,28,1)))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.2))

# cnn.add(Conv2D(32,(3,3),activation='linear'))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.2))

# cnn.add(MaxPooling2D(pool_size=(2,2)))
# cnn.add(Dropout(0.2))

# cnn.add(Conv2D(32,(3,3),activation='linear'))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.2))

# cnn.add(Conv2D(64,(3,3),activation='linear'))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.2))

# # cnn.add(MaxPooling2D(pool_size=(1,1)))

# # cnn.add(Conv2D(16,(3,3),activation='linear'))
# # cnn.add(LeakyReLU(alpha=0.001))
# # cnn.add(Dropout(0.2))

# # cnn.add(Conv2D(16,(1,1),activation='linear'))
# # cnn.add(LeakyReLU(alpha=0.1))
# # cnn.add(Dropout(0.33))

# cnn.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
# # cnn.add(MaxPooling2D(pool_size=(2,2)))
# cnn.add(Flatten())
# cnn.add(Dense(512,activation="linear"))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.3))
# cnn.add(Dense(100,activation="linear"))
# cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.3))
# cnn.add(Dense(20,activation="softmax"))
# cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# cnn.fit_generator(datagen.flow(x_train,y_train_categorical,batch_size=32),steps_per_epoch=len(x_train)/32,epochs=50,verbose=1)

cnn =Sequential()
cnn.add(Conv2D(16, kernel_size=(5, 5),activation="linear",strides=2,padding='same',input_shape=(28,28,1)))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(Conv2D(64, kernel_size=(3, 3),activation="linear",padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.33))
cnn.add(Conv2D(64, kernel_size=(3, 3),activation="linear",padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(Conv2D(32, kernel_size=(1, 1),activation="linear",padding='same'))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.33))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.15))
cnn.add(Dense(100, activation='relu'))
cnn.add(Dropout(0.15))
cnn.add(Dense(20, activation='softmax'))
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

for i in range(5):
	# y_train_categorical = to_categorical(y_train_categorical,num_classes=20)
	cnn.fit(x_train,y_train_categorical,epochs=5,batch_size=128,verbose=1)

	name = 'cnn20_' + str(5*(i+1))
	# name = 'cnn11_5'
	print(name)
	cnn.save("../out/cnn/" + name + ".h5")
	cnn = load_model("../out/cnn/" + name + ".h5")

	pred = np.argmax(cnn.predict(x_test),axis=1)

	with open("../out/cnn/" + name + ".csv","w") as f:
	    f.write("ID,CATEGORY\n")
	    for idx, p in enumerate(pred):
	        f.write("{0},{1}\n".format(idx,categories[p]))
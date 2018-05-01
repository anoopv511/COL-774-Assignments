import numpy as np
import os
from functions import *
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical

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
print("Data loaded")

x_train = x_train/255
x_test = x_test/255

class kerasNN(BaseEstimator):
    def __init__(self,hidden_units=32,epochs=20,batch_size=32):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.model.add(Dense(hidden_units, input_dim=784, activation='sigmoid'))
        self.model.add(Dense(20, activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    def fit(self,x_train,y_train):
        print(self.hidden_units)
        y_train_categorical = to_categorical(y_train,num_classes=20)
        self.model.fit(x_train,y_train_categorical,epochs=self.epochs,batch_size=self.batch_size,verbose=1)
        return self

    def predict(self,x):
        return np.argmax(self.model.predict(x),axis=1)

# tuned_params = [{'hidden_units':[2,5,10,20,50,100,200,300,500,1000], 'epochs':[50], 'batch_size':[64]}]
# model = GridSearchCV(kerasNN(),param_grid=tuned_params,scoring='accuracy',cv=3)
# model.fit(x_train,y_train)
# print(model.best_params_)
# best_model = model.best_estimator_
# best_model.model.fit(x_train,to_categorical(y_train,num_classes=20),epochs=100,batch_size=64)
# best_model.model.save("../out/nn/best_nn.h5")
best_model = load_model("../out/nn/best_nn.h5")

train_pred = np.argmax(best_model.predict(x_train),axis=1)
print("Train Accuracy = {0}".format((train_pred == y_train).sum()/y_train.shape[0]))

pred = np.argmax(best_model.predict(x_test),axis=1)

with open("../out/nn/best_nn.csv","w") as f:
    f.write("ID,CATEGORY\n")
    for idx, p in enumerate(pred):
        f.write("{0},{1}\n".format(idx,categories[p]))

# Train Accuracy = 0.81200
# Test Accuracy = 0.66855
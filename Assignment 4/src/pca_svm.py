import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *

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

pca = PCA(n_components=50,random_state=0)
pca.fit(x_train)

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("PCA done")

# tuned_params = [{'kernel':['rbf'], 'gamma': [1e-2,1e-3], 'C': [0.01, 0.1, 1, 10, 100]}]
# rbf_model = GridSearchCV(SVC(),param_grid=tuned_params,scoring='accuracy',n_jobs=4,cv=3,verbose=1)
# rbf_model.fit(x_train_pca,y_train)
# save(rbf_model,"../out/pca_svm/rbf.pickle")
rbf_model = load("../out/pca_svm/rbf.pickle")
print("Model loaded")

rbf_best = rbf_model.best_estimator_
rbf_train_pred = rbf_best.predict(x_train_pca)
print("Train Accuracy = {0}".format((rbf_train_pred == y_train).sum()/y_train.shape[0]))

pred = rbf_best.predict(x_test_pca)

with open("../out/pca_svm/best_rbf.csv","w") as f:
    f.write("ID,CATEGORY\n")
    for idx, p in enumerate(pred):
        f.write("{0},{1}\n".format(idx,categories[p]))

# tuned_params = [{'kernel':['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]
# lin_model = GridSearchCV(SVC(),param_grid=tuned_params,scoring='accuracy',n_jobs=10,cv=3,verbose=1)
# lin_model.fit(x_train_pca,y_train)
# save(lin_model,"../out/pca_svm/lin.pickle")
# lin_model = load("../out/pca_svm/lin.pickle")
print("Model loaded")

lin_best = lin_model.best_estimator_
lin_train_pred = lin_best.predict(x_train_pca)
print("Train Accuracy = {0}".format((lin_train_pred == y_train).sum()/y_train.shape[0]))

pred = lin_best.predict(x_test_pca)

with open("../out/pca_svm/best_lin.csv","w") as f:
    f.write("ID,CATEGORY\n")
    for idx, p in enumerate(pred):
        f.write("{0},{1}\n".format(idx,categories[p]))
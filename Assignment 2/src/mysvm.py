import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from functions import *

############ Part (a) ############

# SVM Classifier
class SVM():
    def __init__(self,maxit,maxcount,batchsize,lambda_,c=1,project=False):
        self.maxit = maxit                  # Maximum Iterations
        self.maxcount = maxcount            # Maximum Count for stopping training
        self.batchsize = batchsize          # Batchsize
        self.lambda_ = lambda_              # Hyperparameter - lambda
        self.c = c                          # Hyperparameter - c
        self.project = project              # Optional - Projection Step
    
    # fit method
    def fit(self,x_train,y_train,printAfter=1):
        indices = np.arange(self.batchsize)
        self.w = np.zeros((x_train.shape[1],1))
        self.b = 0
        counter = 0
        prev_cost = 0
        for it in range(self.maxit):
            rand_idx = np.random.randint(0,x_train.shape[0],self.batchsize)
            sub_x, sub_y = x_train[rand_idx], y_train[rand_idx]
            loss = sub_y*(np.dot(sub_x,self.w) + self.b)
            cost = self.lambda_*(self.w**2).sum()/2 + self.c*loss.sum()/float(self.batchsize)
            if(it > 0 and it%printAfter == 0): print("{0} - {1}".format(it,cost))
            counter = counter + 1 if cost > prev_cost else 0
            prev_cost = cost
            idx = indices[loss.ravel() < 1]
            eta = 1/float(self.lambda_*(it+1))
            if(counter > self.maxcount):
                print("{0} - {1}".format(it,cost))
                break
            self.w = self.w*(1 - self.lambda_*eta) + (self.c*eta/float(self.batchsize))*(sub_x[idx]*sub_y[idx]).sum(axis=0).reshape(-1,1)
            self.b = (self.c*eta/float(self.batchsize))*sub_y[idx].sum()
            if self.project:
                self.w *= np.min(1,float(1/float(np.sqrt(lambda_*np.dot(self.w.T,self.w)))))
            
    # predict method
    def predict(self,x_test,conf=False):
        return ((np.dot(x_test,self.w) + self.b) > 0).astype(np.int64) if not conf else (np.dot(x_test,self.w) + self.b)

##################################

############ Part (b) ############

# One-vs-One Model
def onevsone(x_train,y_train,maxit,maxcount,batchsize,lambda_,c=1,project=False,printAfter=1):
    classifiers = []
    labels = np.unique(y_train)
    class_split = list(itertools.combinations(np.arange(10),2))
    indices = np.arange(x_train.shape[0])
    for split in class_split:
        idx_c1 = indices[(y_train == split[0]).ravel()]
        idx_c2 = indices[(y_train == split[1]).ravel()]
        idx = np.concatenate((idx_c1,idx_c2))
        sub_x, sub_y = x_train[idx], y_train[idx]
        sub_y = (sub_y == split[0]).astype(np.int64) - (sub_y == split[1]).astype(np.int64)
        classifier = SVM(maxit,maxcount,batchsize,lambda_,c)
        classifier.fit(sub_x,sub_y,printAfter)
        classifiers.append((classifier,split[0],split[1]))
    return classifiers

def pred_onevsone(classifiers,x_test):
    preds = np.zeros((x_test.shape[0],len(classifiers)))
    ones = np.ones((x_test.shape[0],1))
    for idx, c in enumerate(classifiers):
        pred = c[0].predict(x_test,conf=False).reshape(-1,1)
        preds[:,idx] = np.where(pred == 1,c[1]*ones,c[2]*ones).ravel()
    final_pred = -stats.mode(-preds,axis=1)[0]
    return final_pred

def indv_acc_onevsone(classifiers,x_test,y_test):
    for c in classifiers:
        indices = np.arange(x_test.shape[0])
        idx_c1 = indices[(y_test == c[1]).ravel()]
        idx_c2 = indices[(y_test == c[2]).ravel()]
        idx = np.concatenate((idx_c1,idx_c2))
        sub_x, sub_y = x_test[idx], y_test[idx]
        ones = np.ones((sub_x.shape[0],1))
        pred = c[0].predict(sub_x).reshape(-1,1)
        pred = np.where(pred == 1,c[1]*ones,c[2]*ones)
        print("Accuracy for classifier b/w {0}/{1} = {2}".format(c[1],c[2],(pred == sub_y).sum()/float(sub_y.shape[0])))

# One-vs-All Model
def onevsall(x_train,y_train,maxit,maxcount,batchsize,lambda_,c=1,project=False,printAfter=1):
    classifiers = []
    labels = np.sort(np.unique(y_train))
    for l in labels:
        sub_x, sub_y = x_train, (y_train == l).astype(np.int64) - (y_train != l).astype(np.int64)
        classifier = SVM(maxit,maxcount,batchsize,lambda_,c)
        classifier.fit(sub_x,sub_y,printAfter)
        classifiers.append((classifier,l))
    return classifiers
        
def pred_onevsall(classifiers,x_test):
    preds = np.zeros((x_test.shape[0],len(classifiers)))
    for idx, c in enumerate(classifiers):
        preds[:,idx] = c[0].predict(x_test,conf=True).ravel()
    final_pred = preds.argmax(axis=1).reshape(-1,1)
    return final_pred

def indv_acc_onevsall(classifiers,x_test,y_test):
    for c in classifiers:
        ones = np.ones((x_test.shape[0],1))
        pred = (c[0].predict(x_test,conf=True) > 0.5).astype(np.int64)
        pred = np.where(pred == 1,c[1]*ones,-1*ones)
        sub_y = np.where(y_test == c[1],c[1]*ones,-1*ones)
        print("Accuracy for classifier b/w {0}/Rest = {1}".format(c[1],(pred == sub_y).sum()/float(sub_y.shape[0])))

if __name__ == "__main__":
    # Train Data
    x_train = np.loadtxt('../data/mnist/train.csv',delimiter=',',dtype=np.float64)
    y_train = x_train[:,-1].reshape(-1,1)
    x_train = np.delete(x_train,-1,1)
    x_train /= 255              # Scaling

    # Test Data
    x_test = np.loadtxt('../data/mnist/test.csv',delimiter=',',dtype=np.float64)
    y_test = x_test[:,-1].reshape(-1,1)
    x_test = np.delete(x_test,-1,1)
    x_test /= 255               # Scaling

    svm_one = onevsone(x_train,y_train,5000,7,100,0.05,1,False,10000)
    save(svm_one,"../data/tmp/svm_one.pickle")

    # Train Accuracy
    pred_one_train = pred_onevsone(svm_one,x_train)
    accuracy_one_train = (pred_one_train == y_train).sum()/float(y_train.shape[0])
    print("One-vs-One Train Accuracy = {0}".format(accuracy_one_train))
    # indv_acc_onevsone(svm_one,x_train,y_train)

    # Test Accuracy
    pred_one_test = pred_onevsone(svm_one,x_test)
    accuracy_one_test = (pred_one_test == y_test).sum()/float(y_test.shape[0])
    print("One-vs-One Test Accuracy = {0}".format(accuracy_one_test))
    # indv_acc_onevsone(svm_one,x_test,y_test)

    # One-vs-One Train Accuracy = 0.9341
    # One-vs-One Test Accuracy = 0.9328

    svm_all = onevsall(x_train,y_train,5000,7,100,0.05,1,False,10000)
    save(svm_all,"../data/tmp/svm_all.pickle")

    # Train Accuracy
    pred_all_train = pred_onevsall(svm_all,x_train)
    accuracy_all_train = (pred_all_train == y_train).sum()/float(y_train.shape[0])
    print("One-vs-All Train Accuracy = {0}".format(accuracy_all_train))
    # indv_acc_onevsall(svm_all,x_train,y_train)

    # Test Accuracy
    pred_all_test = pred_onevsall(svm_all,x_test)
    accuracy_all_test = (pred_all_test == y_test).sum()/float(y_test.shape[0])
    print("One-vs-All Test Accuracy = {0}".format(accuracy_all_test))
    # indv_acc_onevsall(svm_all,x_test,y_test)

    # One-vs-All Train Accuracy = 0.87715
    # One-vs-All Test Accuracy = 0.887

##################################
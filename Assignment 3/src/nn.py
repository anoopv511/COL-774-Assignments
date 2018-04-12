import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,"../scripts/")
sys.path.insert(0,'../libsvm-3.22/python/')
from visualization import *
from svmutil import *

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x <= 0] = 0
    return x

class Layer():
    def __init__(self,in_size,out_size,random_state=0):
        np.random.seed(random_state)
        self.i = in_size
        self.o = out_size
        if(not in_size == -1):
            self.w = np.random.normal(0,0.05,in_size * (out_size-1)).reshape(in_size,out_size-1)
            self.w_ = 0
        self.delta = 0
        self.out = 0

class NeuralNet():
    def __init__(self,in_size,layers,act="sigmoid"):
        self.make_layers(in_size,layers)
        self.act = act

    def make_layers(self,in_size,layers):
        self.layers = []
        self.layers.append(Layer(-1,in_size+1))
        for idx, l in enumerate(layers):
            self.layers.append(Layer(self.layers[-1].o,l+1))
        self.layers.append(Layer(self.layers[-1].o,2))
        self.layers[-1].out_size = 1

    def forward(self,x):
        self.layers[0].out = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        for idx in range(1,len(self.layers)-1):
            if(self.act == "sigmoid"):
                self.layers[idx].out = sigmoid(np.dot(self.layers[idx-1].out,self.layers[idx].w))
            if(self.act == "relu"):
                self.layers[idx].out = relu(np.dot(self.layers[idx-1].out,self.layers[idx].w))
            self.layers[idx].out = np.concatenate((np.ones((self.layers[idx].out.shape[0],1)),self.layers[idx].out),axis=1)
        self.layers[-1].out = sigmoid(np.dot(self.layers[-2].out,self.layers[-1].w))

    def backward(self,y):
        for idx, layer in enumerate(self.layers[::-1]):
            if(idx == 0):
                layer.delta = -1 * (y.reshape(-1,1) - layer.out) * layer.out * (1 - layer.out)
            else:
                next_layer = self.layers[-idx]
                if(self.act == "sigmoid"):
                    layer.delta = np.einsum('ij,kj->ki',next_layer.w[1:,:],next_layer.delta) * layer.out[:,1:] * (1 - layer.out[:,1:])
                if(self.act == "relu"):
                    layer.delta = np.einsum('ij,kj->ki',next_layer.w[1:,:],next_layer.delta) * (layer.out[:,1:] > 0)
        for idx in range(1,len(self.layers)):
            self.layers[idx].w_ = np.einsum('ij,ik->jk',self.layers[idx-1].out,self.layers[idx].delta)

    def fit(self,x_train,y_train,eta,epochs,batchsize,eta_dynamic=False,verbose=False,printAfter=(100,1)):
        assert x_train.shape[0]%batchsize == 0, "Number of training examples should be divisible by batchsize"
        batches = int(x_train.shape[0]/batchsize)
        e = 0
        eta2 = eta
        while(e < epochs):
            for i in range(batches):
                self.forward(x_train[i*batchsize:(i+1)*batchsize])
                self.backward(y_train[i*batchsize:(i+1)*batchsize])
                for layer in self.layers[1:]:
                    layer.w -= eta * layer.w_
                cost = ((y_train[i*batches:(i+1)*batches] - self.layers[-1].out.flatten())**2).sum()/(2*batchsize)
                if(verbose and e%printAfter[0] == 0 and i%printAfter[1] == 0): print("Epoch = {0} | Cost = {1}".format(e,cost))
            if(verbose and e%printAfter[0] == 0): print("-----------")
            e += 1
            if(eta_dynamic): eta = eta2/np.sqrt(e)

    def predict(self,x):
        self.forward(x)
        return np.round(self.layers[-1].out).flatten().astype(np.int64)

    def accuracy(self,x,y):
        self.forward(x)
        pred = np.round(self.layers[-1].out).flatten().astype(np.int64)
        return (y == pred).sum()/y.shape[0]

if __name__ == "__main__":
    x_train = np.loadtxt("../data/toy_data/toy_trainX.csv",delimiter=',')
    y_train = np.loadtxt("../data/toy_data/toy_trainY.csv",delimiter=',')
    x_test = np.loadtxt("../data/toy_data/toy_testX.csv",delimiter=',')
    y_test = np.loadtxt("../data/toy_data/toy_testY.csv",delimiter=',')

    log = LogisticRegression(C=1e-3)
    log.fit(x_train,y_train)
    p = log.predict(x_train)
    print("Train Accuracy = {0:.5f}".format((p == y_train).sum()/y_train.shape[0]))
    p = log.predict(x_test)
    print("Test Accuracy = {0:.5f}".format((p == y_test).sum()/y_test.shape[0]))
    plot_decision_boundary(log.predict,x_train,y_train,"Logistic Regression (Train)")
    plt.show()
    plot_decision_boundary(log.predict,x_test,y_test,"Logistic Regression (Test)")
    plt.show()

    # Train Accuracy = 0.46053
    # Test Accuracy = 0.35000

    nn = NeuralNet(x_train.shape[1],[5])
    t0 = time.time()
    nn.fit(x_train,y_train,0.01,10000,380)
    print("Training Time = {0:.5f}".format(time.time()-t0))
    print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
    print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))
    plot_decision_boundary(nn.predict,x_train,y_train,"Neural Net - 5 Hidden Units (Train)")
    plt.show()
    plot_decision_boundary(nn.predict,x_test,y_test,"Neural Net - 5 Hidden Units (Test)")
    plt.show()

    # Training Time = 2.74530
    # Train Accuracy = 0.89737
    # Test Accuracy = 0.85000   

    hidden_units = [1,2,3,10,20,40]
    for h in hidden_units:
        nn = NeuralNet(x_train.shape[1],[h])
        t0 = time.time()
        nn.fit(x_train,y_train,0.01,10000,380)
        print("Training Time = {0:.5f}".format(time.time()-t0))
        print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
        print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))
        plot_decision_boundary(nn.predict,x_test,y_test,"Neural Net - {0} Hidden Units (Test)".format(h))
        plt.show()

    # Training Time = [1.51454, 1.88829, 2.07101, 2.93981, 4.50854, 10.57310]
    # Train Accuracy = [0.64211, 0.60789, 0.90000, 0.91053, 0.91316, 0.91053]
    # Test Accuracy = [0.56667, 0.58333, 0.85833, 0.82500, 0.82500, 0.82500]

    nn = NeuralNet(x_train.shape[1],[5,5])
    t0 = time.time()
    nn.fit(x_train,y_train,0.01,10000,380)
    print("Training Time = {0:.5f}".format(time.time()-t0))
    print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
    print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))
    plot_decision_boundary(nn.predict,x_train,y_train,"Neural Net - [5,5] Hidden Units (Train)")
    plt.show()
    plot_decision_boundary(nn.predict,x_test,y_test,"Neural Net - [5,5] Hidden Units (Test)")
    plt.show()

    # Training Time = 3.62151
    # Train Accuracy = 0.90000
    # Test Accuracy = 0.85833

    train = np.loadtxt("../data/mnist_data/MNIST_train.csv",delimiter=',')
    test = np.loadtxt("../data/mnist_data/MNIST_test.csv",delimiter=',')
    x_train = train[:,:-1].tolist()
    y_train = (train[:,-1] == 6).astype(np.int64)
    y_train -= y_train == 0
    y_train = y_train.tolist()
    x_test = test[:,:-1].tolist()
    y_test = (test[:,-1] == 6).astype(np.int64)
    y_test -= y_test == 0
    y_test = y_test.tolist()

    t0 = time.time()
    lin_svm = svm_train(y_train,x_train,'-t 0 -c 1')
    print("Training Time = {0:.5f}".format(time.time()-t0))
    train_pred, train_acc, train_val = svm_predict(y_train,x_train,lin_svm)
    test_pred, test_acc, test_val = svm_predict(y_test,x_test,lin_svm)
    print("Train Accuracy = {0:.5f}".format((train_pred == y_train).sum()/y_train.shape[0]))
    print("Test Accuracy = {0:.5f}".format((test_pred == y_test).sum()/y_test.shape[0]))

    # Training Time = 6.62021
    # Train Accuracy = 1.00000
    # Test Accuracy = 0.98472
    
    x_train = train[:,:-1]
    y_train = (train[:,-1] == 6).astype(np.int64)
    x_test = test[:,:-1]
    y_test = (test[:,-1] == 6).astype(np.int64)

    nn = NeuralNet(x_train.shape[1],[1])
    t0 = time.time()
    nn.fit(x_train,y_train,0.01,100,100,verbose=False,printAfter=(10,50),eta_dynamic=True)
    print("Training Time = {0:.5f}".format(time.time()-t0))
    print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
    print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))

    # Training Time = 10.87574
    # Train Accuracy = 0.97470
    # Test Accuracy = 0.97556

    nn = NeuralNet(x_train.shape[1],[100])
    t0 = time.time()
    nn.fit(x_train,y_train,0.01,100,100,verbose=False,printAfter=(10,50),eta_dynamic=True)
    print("Training Time = {0:.5f}".format(time.time()-t0))
    print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
    print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))

    # Training Time = 159.88187
    # Train Accuracy = 0.99880
    # Test Accuracy = 0.99306

    # nn = NeuralNet(x_train.shape[1],[100],act="relu")
    # t0 = time.time()
    # nn.fit(x_train,y_train,0.01,100,100,verbose=True,printAfter=(10,50),eta_dynamic=True)
    # print("Training Time = {0:.5f}".format(time.time()-t0))
    # print("Train Accuracy = {0:.5f}".format(nn.accuracy(x_train,y_train)))
    # print("Test Accuracy = {0:.5f}".format(nn.accuracy(x_test,y_test)))
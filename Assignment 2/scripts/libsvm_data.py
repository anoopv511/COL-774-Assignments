import numpy as np

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

print("Started copying")
with open('../data/mnist/libsvm_train.txt','w') as f:
    for idx in range(x_train.shape[0]):
        f.writelines("{0} ".format(y_train[idx][0]))
        _ = [f.writelines("{0}:{1} ".format(idy+1,x_train[idx][idy])) for idy in range(x_train.shape[1])]
        f.writelines("\n")
print("Finished copying train data")

with open('../data/mnist/libsvm_test.txt','w') as f:
    for idx in range(x_test.shape[0]):
        f.writelines("{0} ".format(y_test[idx][0]))
        _ = [f.writelines("{0}:{1} ".format(idy+1,x_test[idx][idy])) for idy in range(x_test.shape[1])]
        f.writelines("\n")
print("Finished copying test data")
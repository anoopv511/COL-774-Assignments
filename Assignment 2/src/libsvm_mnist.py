import sys
sys.path.insert(0,'../libsvm-3.22/python/')
from svmutil import *

# Scaled train and test data
y_train, x_train = svm_read_problem('../data/mnist/libsvm_train.txt')
y_test, x_test = svm_read_problem('../data/mnist/libsvm_test.txt')
print("Loaded Train and Test data")

############ Part (c) ############

# Linear Kernel
print("Using Linear Kernel SVM")
# model = svm_train(y_train,x_train,'-t 0 -c 1')
# svm_save_model('../mnist_libsvm/linear.model',model)
model = svm_load_model('../mnist_libsvm/linear.model')
print("Finished training Linear Kernel SVM")

print("Train Accuracy")
# train_pred, train_acc, train_val = svm_predict(y_train,x_train,model)
# with open('../mnist_libsvm/linear_train.out','w') as f:
#     _ = [f.writelines("{0}\n".format(p)) for p in train_pred]
with open('../mnist_libsvm/linear_train.out','r') as f:
    train_pred = [int(line.rstrip('\n')) for line in f]
train_acc, train_mse, train_scc = evaluations(y_train,train_pred)
print("Linear kernel SVM Train Accuracy = {0}".format(train_acc))

print("Test Accuracy")
# test_pred, test_acc, test_val = svm_predict(y_test,x_test,model)
# with open('../mnist_libsvm/linear_test.out','w') as f:
#     _ = [f.writelines("{0}\n".format(p)) for p in test_pred]
with open('../mnist_libsvm/linear_test.out','r') as f:
    test_pred = [int(line.rstrip('\n')) for line in f]
test_acc, test_mse, test_scc = evaluations(y_test,test_pred)    
print("Linear kernel SVM Test Accuracy = {0}".format(test_acc))

# Gaussian Kernel
print("Using Gaussian Kernel SVM")
# model = svm_train(y_train,x_train,'-t 2 -c 1 -g 0.05')
# svm_save_model('../mnist_libsvm/gaussian.model',model)
model = svm_load_model('../mnist_libsvm/gaussian.model')
print("Finished training Gaussian Kernel SVM")

print("Train Accuracy")
# train_pred, train_acc, train_val = svm_predict(y_train,x_train,model)
# with open('../mnist_libsvm/gaussian_train.out','w') as f:
#     _ = [f.writelines("{0}\n".format(p)) for p in train_pred]
with open('../mnist_libsvm/gaussian_train.out','r') as f:
    train_pred = [int(line.rstrip('\n')) for line in f]
train_acc, train_mse, train_scc = evaluations(y_train,train_pred)
print("Gaussian kernel SVM Train Accuracy = {0}".format(train_acc))

print("Test Accuracy")
# test_pred, test_acc, test_val = svm_predict(y_test,x_test,model)
# with open('../mnist_libsvm/gaussian_test.out','w') as f:
#     _ = [f.writelines("{0}\n".format(p)) for p in test_pred]
with open('../mnist_libsvm/gaussian_test.out','r') as f:
    test_pred = [int(line.rstrip('\n')) for line in f]
test_acc, test_mse, test_scc = evaluations(y_test,test_pred)    
print("Gaussian kernel SVM Test Accuracy = {0}".format(test_acc))

##################################

############ Part (d) ############



##################################
import sys
sys.path.insert(0,'../libsvm-3.22/python/')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scaled train and test data
y_train, x_train = svm_read_problem('../data/mnist/libsvm_train.txt')
y_test, x_test = svm_read_problem('../data/mnist/libsvm_test.txt')
print("Loaded Train and Test data")

############ Part (e) ############

best_c = 10
# model = svm_train(y_train,x_train,"-t 2 -g 0.05 -c {0} -h 0".format(best_c))
# svm_save_model('../mnist_libsvm/best.model',model)
model = svm_load_model('../mnist_libsvm/best.model')
print("Finished Training/Loading model")
# test_pred, test_acc, test_val = svm_predict(y_test,x_test,model)
# with open("../mnist_libsvm/best_test.out","w") as out:
#     _ = [out.writelines("{0}\n".format(int(p))) for p in test_pred]
with open('../mnist_libsvm/best_test.out','r') as f:
    test_pred = [int(line.rstrip('\n')) for line in f]
print("Finished predictions")

test_pred = np.asarray(test_pred,dtype=np.int64)
y_test = np.asarray(y_test,dtype=np.int64)

classes = np.arange(0,10,dtype=np.int64)
conf_matrix = np.zeros((classes.shape[0],classes.shape[0]),dtype=np.int64)
for l, p in zip(y_test,test_pred):
    conf_matrix[classes[int(p)],classes[int(l)]] += 1

fig = plt.figure(figsize=(10,8))
ax = fig.gca()
_ = sns.heatmap(conf_matrix,annot=True,cmap="Blues",xticklabels=classes,yticklabels=classes,fmt='g')
ax.set_xlabel("Actual Class")
ax.set_ylabel("Predicted Class")
plt.title("Confusion Matrix",y=1.08)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()

mis_pred = (np.array(test_pred) != np.array(y_test)).astype(np.int64).nonzero()
for idx in range(min(4,mis_pred[0].shape[0])):
    idt = mis_pred[0][idx]
    x = [x_test[idt][k] for k in range(1,785)]
    img = (np.asarray(x)*255).reshape(28,28)
    plt.title("Index = {0} | True Label = {1} | Predicted Label = {2}".format(idt,y_test[idt],test_pred[idt]))
    plt.imshow(img,cmap="gray")
    plt.show()

##################################
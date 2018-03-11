import sys, time
sys.path.insert(0,'../libsvm-3.22/python/')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scaled train and test data
# y_train, x_train = svm_read_problem('../data/mnist/libsvm_train.txt')
# y_test, x_test = svm_read_problem('../data/mnist/libsvm_test.txt')
# print("Loaded Train and Test data")

############ Part (d) ############

C = [1e-5,1e-3,1,5,10]
cv_accuracy, test_accuracy = [71.59,71.59,97.355,97.455,97.455], [72.1,72.1,97.23,97.29,97.29]

# cv_accuracy, test_accuracy = [], []
# for idx, c in enumerate(C):
#     start = time.time()
#     cv_acc = svm_train(y_train,x_train,"-t 2 -g 0.05 -c {0} -v 10 -h 0".format(c))
#     model = svm_train(y_train,x_train,"-t 2 -g 0.05 -c {0} -h 0".format(c))
#     test_pred, test_acc, test_val = svm_predict(y_test,x_test,model)
#     with open("../mnist_libsvm/cv_scores.txt","a+") as out:
#         out.write(str(c) + " - " + str(cv_acc) + " - " + str(test_acc[0]))
#     print("Finished Cross-Validation for c = {0} after {1:.4f} seconds".format(c,(time.time() - start)))
#     cv_accuracy.append(cv_acc)
#     test_accuracy.append(test_acc[0])

ax = plt.gca()
cv_patch = plt.plot(np.log10(C),cv_accuracy,color="red",label="Cross-Validation Accuracy")[0]
test_patch = plt.plot(np.log10(C),test_accuracy,color="blue",label="Test Accuracy")[0]
ax.set_xlabel(r"$\log_{10}{C}$")
ax.set_ylabel("Accuracy")
ax.add_artist(plt.legend(handles=[cv_patch,test_patch]))
plt.show()

##################################
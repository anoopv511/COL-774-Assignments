import sys
sys.path.insert(0,'../libsvm-3.22/python/')
from svmutil import *
from mysvm import *

if __name__ == "__main__":
    model = int(sys.argv[1])
    inputPath = "../"+sys.argv[2]
    outputPath = "../"+sys.argv[3]
    x_test = np.loadtxt(inputPath,delimiter=',',dtype=np.float64)/255
    if model == 1:
        svm_one = load("../data/tmp/svm_one.pickle")
        pred = pred_onevsone(svm_one,x_test)
        with open(outputPath,"w") as out:
            for p in pred:
                out.write(str(p[0]) + "\n")
    if model == 2:
        libsvm_model = svm_load_model('../mnist_libsvm/linear.model')
        x_test = x_test.tolist()
        pred, acc, val = svm_predict(np.zeros(len(x_test)).tolist(),x_test,libsvm_model)
        with open(outputPath,'w') as out:
            _ = [out.writelines("{0}\n".format(p)) for p in pred]
    if model == 3:
        libsvm_model = svm_load_model('../mnist_libsvm/best.model')
        x_test = x_test.tolist()
        pred, acc, val = svm_predict(np.zeros(len(x_test)).tolist(),x_test,libsvm_model)
        with open(outputPath,'w') as out:
            _ = [out.writelines("{0}\n".format(p)) for p in pred]
import sys
from nb import *

if __name__ == "__main__":
    model = int(sys.argv[1])
    inputPath = "../"+sys.argv[2]
    outputPath = "../"+sys.argv[3]
    x_test = np.loadtxt(inputPath,delimiter='\n',comments=None,dtype='str')
    labels = np.array([1,2,3,4,7,8,9,10])
    if model == 1:
        diction,log_theta,log_phi = load("../data/tmp/model1.pickle")
        labels = np.array([1,2,3,4,7,8,9,10])
        with open(outputPath,"w") as out:
            for doc in x_test:
                pred = predict(doc,log_theta,log_phi,diction,labels)
                out.write(str(pred) + "\n")
    elif model == 2:
        diction,log_theta,log_phi = load("../data/tmp/model2.pickle")
        with open(outputPath,"w") as out:
            for doc in x_test:
                pred = predict(doc,log_theta,log_phi,diction,labels)
                out.write(str(pred) + "\n")
    elif model == 3:
        diction,log_theta,log_phi = load("../data/tmp/model3.pickle")
        with open(outputPath,"w") as out:
            for doc in x_test:
                pred = predict(doc,log_theta,log_phi,diction,labels)
                out.write(str(pred) + "\n")
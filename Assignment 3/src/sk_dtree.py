import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocess import *
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

train = pd.read_csv("../data/dtree/train.csv",index_col=None)
x_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1].as_matrix()

valid = pd.read_csv("../data/dtree/valid.csv",index_col=None)
x_valid = valid.iloc[:,:-1]
y_valid = valid.iloc[:,-1].as_matrix()

test = pd.read_csv("../data/dtree/test.csv",index_col=None)
x_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1].as_matrix()

encodings = get_encodings(x_train,median_spit=True)
x_train = preprocess(x_train,encodings)
columns = x_train.columns
x_train = x_train.as_matrix()
x_valid = preprocess(x_valid,encodings).as_matrix()
x_test = preprocess(x_test,encodings).as_matrix()

############ Part (d) ############

criterion = ['gini','entropy']
max_depth = [3,5,8,10,12] + [None]
min_samples_split = [0.001,0.005,0.01,0.05,0.1] + [2]
min_samples_leaf = [0.001,0.005,0.01,0.05,0.1] + [1]
max_features = [5,10,'sqrt','log2'] + [None]
random_state = 0

params = list(itertools.product(criterion,max_depth,min_samples_split,min_samples_leaf,max_features))

dtrees = []
for p in params:
    dtrees.append(DecisionTreeClassifier(criterion=p[0],max_depth=p[1],min_samples_split=p[2],min_samples_leaf=p[3],max_features=p[4],random_state=random_state))

train_acc = []
valid_acc = []
test_acc = []
for tree in dtrees:
    _ = tree.fit(x_train,y_train)
    p = tree.predict(x_train)
    train_acc.append((p == y_train).sum()/y_train.shape[0])
    p = tree.predict(x_valid)
    valid_acc.append((p == y_valid).sum()/y_valid.shape[0])
    p = tree.predict(x_test)
    test_acc.append((p == y_test).sum()/y_test.shape[0])

params = np.array(params)
dtrees = np.array(dtrees)
train_acc = np.array(train_acc)
valid_acc = np.array(valid_acc)
test_acc = np.array(test_acc)

# With Criterion = "gini"
# Best Parameter setting - max_depth = 12, min_samples_split = 0.005, min_samples_leaf = 0.001, max_features = None
# Train Accuracy = 0.83507
# Validation Accuracy = 0.82633
# Test Accuracy = 0.82914

best = valid_acc[params[:,0] == 'gini'].argmax()
print("Model = {0}".format(dtrees[best]))
print("Train Accuracy = {0:.5f}".format(train_acc[best]))
print("Validation Accuracy = {0:.5f}".format(valid_acc[best]))
print("Test Accuracy = {0:.5f}".format(test_acc[best]))

# With Criterion = "entropy"
# Best Parameter setting - max_depth = 10, min_samples_split = 0.005, min_samples_leaf = 0.001, max_features = None
# Train Accuracy = 0.83574
# Validation Accuracy = 0.82633
# Test Accuracy = 0.83000

best = valid_acc[params[:,0] == 'entropy'].argmax() + int(dtrees.shape[0]/2)
print("Model = {0}".format(dtrees[best]))
print("Train Accuracy = {0:.5f}".format(train_acc[best]))
print("Validation Accuracy = {0:.5f}".format(valid_acc[best]))
print("Test Accuracy = {0:.5f}".format(test_acc[best]))

##################################

############ Part (e) ############

criterion = ['gini','entropy']
n_estimators = [2,5,10,20]
max_depth = [3,5,8,10,12] + [None]
min_samples_split = [0.001,0.005,0.01,0.05,0.1] + [2]
min_samples_leaf = [0.001,0.005,0.01,0.05,0.1] + [1]
max_features = [5,10,'sqrt','log2'] + [None]
bootstrap = [True,False]
random_state = 0

params = list(itertools.product(criterion,n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features,bootstrap))

rforests = []
for p in params:
    rforests.append(RandomForestClassifier(criterion=p[0],n_estimators=p[1],max_depth=p[2],min_samples_split=p[3],min_samples_leaf=p[4],max_features=p[5],bootstrap=p[6],random_state=random_state))

train_acc = []
valid_acc = []
test_acc = []
for idx, tree in enumerate(rforests):
    if(idx%100 == 0): print(idx)
    _ = tree.fit(x_train,y_train)
    p = tree.predict(x_train)
    train_acc.append((p == y_train).sum()/y_train.shape[0])
    p = tree.predict(x_valid)
    valid_acc.append((p == y_valid).sum()/y_valid.shape[0])
    p = tree.predict(x_test)
    test_acc.append((p == y_test).sum()/y_test.shape[0])

params = np.array(params)
rforests = np.array(rforests)
train_acc = np.array(train_acc)
valid_acc = np.array(valid_acc)
test_acc = np.array(test_acc)

# With Criterion = "gini"
# Best Parameter setting - n_estimators = 2, max_depth = 8, min_samples_split = 0.01, min_samples_leaf = 1, max_features = 10, bootstrap = False
# Train Accuracy = 0.83263
# Validation Accuracy = 0.82867
# Test Accuracy = 0.82586

best = valid_acc[params[:,0] == 'gini'].argmax()
print("Model = {0}".format(rforests[best]))
print("Train Accuracy = {0:.5f}".format(train_acc[best]))
print("Validation Accuracy = {0:.5f}".format(valid_acc[best]))
print("Test Accuracy = {0:.5f}".format(test_acc[best]))

# With Criterion = "entropy"
# Best Parameter setting - n_estimators = 2, max_depth = 12, min_samples_split = 0.01, min_samples_leaf = 0.001, max_features = 10, bootstrap = True
# Train Accuracy = 0.83119
# Validation Accuracy = 0.82933
# Test Accuracy = 0.82771

best = valid_acc[params[:,0] == 'entropy'].argmax() + int(rforests.shape[0]/2)
print("Model = {0}".format(rforests[best]))
print("Train Accuracy = {0:.5f}".format(train_acc[best]))
print("Validation Accuracy = {0:.5f}".format(valid_acc[best]))
print("Test Accuracy = {0:.5f}".format(test_acc[best]))

##################################
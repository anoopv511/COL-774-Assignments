import numpy as np
import pandas as pd
from preprocess import *
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

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

class Node():
    def __init__(self,id_,condition,col,pred):
        self.id = id_
        self.cond = condition
        self.col = col
        self.pred = pred
        self.children = []

    def get_split(self,x):
        return [np.where(c(x[:,self.col]))[0] for c in self.cond] if self.col != -1 else [np.arange(x.shape[0])]

    def is_leaf(self):
        return self.cond == None

class DTree():
    def __init__(self):
        self.root = None
        self.node_count = 0

    def growTree(self,x_train,y_train,median_split=False):
        self.node_count += 1
        if(y_train.sum() == 0):
            self.root = Node(self.node_count,None,-1,0)
            return
        elif(y_train.sum() == y_train.shape[0]):
            self.root = Node(self.node_count,None,-1,1)
            return
        else:
            xjs, col = self.choose_best_attr(x_train,y_train)
            c1 = np.count_nonzero(y_train)
            c0 = y_train.shape[0] - c1
            if(xjs is None):
                self.root = Node(self.node_count,None,-1,np.argmax([c0,c1]))
                return
            conds = [(lambda x,c=c: x == c) for c in xjs]           
            self.root = Node(self.node_count,conds,col,np.argmax([c0,c1]))
            nodes = deque([self.root])
            indices = deque([self.root.get_split(x_train)])
            while(len(nodes) != 0):
                curr = nodes.popleft()
                index = indices.popleft()
                for idx in index:
                    self.node_count += 1
                    if(y_train[idx].sum() == 0):
                        curr.children.append(Node(self.node_count,None,-1,0))
                        continue
                    elif(y_train[idx].sum() == y_train[idx].shape[0]):
                        curr.children.append(Node(self.node_count,None,-1,1))
                        continue
                    else:
                        xjs, col = self.choose_best_attr(x_train[idx],y_train[idx])
                        c1 = np.count_nonzero(y_train[idx])
                        c0 = y_train[idx].shape[0] - c1
                        if(xjs is None):
                            curr.children.append(Node(self.node_count,None,-1,np.argmax([c0,c1])))
                            continue
                        conds = [(lambda x,c=c: x == c) for c in xjs]
                        node = Node(self.node_count,conds,col,np.argmax([c0,c1]))
                        curr.children.append(node)
                        nodes.append(node)
                        split = node.get_split(x_train[idx])
                        indices.append([idx[s] for s in split])

    def choose_best_attr(self,x_train,y_train):
        xjss, info_loss = [], []
        for col in range(x_train.shape[1]):
            xjs = np.unique(x_train[:,col])
            if(xjs.shape[0] == 1):
                info_loss.append(100)
                xjss.append(xjs)
                continue
            xjss.append(xjs)
            P_xj = [(x_train[:,col] == xj).sum()/x_train.shape[0] for xj in xjs]
            ys = np.unique(y_train)
            H_y_xj = np.zeros(xjs.shape)
            for idx, xj in enumerate(xjs):
                P_y_xj = np.zeros(ys.shape)
                for idy, y in enumerate(ys):
                    P_y_xj[idy] = ((x_train[:,col] == xj)*(y_train == y)).sum()/(y_train == y).sum()
                H_y_xj[idx] = -1*(P_y_xj*np.log(P_y_xj + (P_y_xj == 0))).sum()
            info_loss.append((P_xj*H_y_xj).sum())
        best_col = np.argmin(info_loss)
        if(info_loss[best_col] == 100): return None, None
        return xjss[best_col], best_col
    
    def predict(self,x_test,node_lim=-1,numbering='bfs'):
        if(numbering == 'dfs'):
            self.dfs_numbering()
        else:
            self.bfs_numbering()
        if(node_lim == -1): node_lim = self.node_count
        def rec_predict(node,x_test,idx,pred):
            pred[idx] = node.pred
            if(not node.is_leaf()):
                split = node.get_split(x_test)
                for c, i in zip(node.children,split):
                    if(c.id <= node_lim):
                        rec_predict(c,x_test[i],idx[i],pred)
        pred = np.ones(x_test.shape[0],dtype=np.int64) * -1
        if(self.root.id <= node_lim):
            rec_predict(self.root,x_test,np.arange(x_test.shape[0]),pred)
        return pred
    
    def bfs_numbering(self):
        q = deque([self.root])
        number = 0
        while(len(q) != 0):
            curr = q.popleft()
            number += 1
            curr.id = number
            q.extend(curr.children)
            
    def dfs_numbering(self):
        s = [self.root]
        number = 0
        while(len(s) != 0):
            curr = s.pop()
            number += 1
            curr.id = number
            s.extend(curr.children[::-1])

tree = DTree()
tree.growTree(x_train,y_train)

p_train = tree.predict(x_train)
print("Train Accuracy = {0:.4f}".format((p_train == y_train).sum()/y_train.shape[0]))
p_valid = tree.predict(x_valid)
print("Validation Accuracy = {0:.4f}".format((p_valid == y_valid).sum()/y_valid.shape[0]))
p_test = tree.predict(x_test)
print("Test Accuracy = {0:.4f}".format((p_test == y_test).sum()/y_test.shape[0]))

# x_data = np.concatenate((x_train,x_valid,x_test),axis=0)
# data_acc = np.zeros((tree.node_count,3))
# for idx in range(1,tree.node_count+1):
#     p = tree.predict(x_data,idx,'bfs')
#     p1 = (p[:y_train.shape[0]] == y_train).sum()/y_train.shape[0]
#     p2 = (p[y_train.shape[0]:y_train.shape[0]+y_valid.shape[0]] == y_valid).sum()/y_valid.shape[0]
#     p3 = (p[y_train.shape[0]+y_valid.shape[0]:] == y_test).sum()/y_test.shape[0]
#     data_acc[idx-1] = [p1,p2,p3]

# sns.set()
# fig = plt.figure(figsize=(7,5))
# train_patch = plt.plot(np.arange(1,tree.node_count+1),data_acc[:,0],color='b',label='Train Accuracy')[0]
# valid_patch = plt.plot(np.arange(1,tree.node_count+1),data_acc[:,1],color='g',label='Validation Accuracy')[0]
# test_patch = plt.plot(np.arange(1,tree.node_count+1),data_acc[:,2],color='r',label='Test Accuracy')[0]
# plt.xlabel("Number of Nodes")
# plt.ylabel("Accuracy")
# plt.legend(handles=[train_patch,valid_patch,test_patch])
# plt.title("Accuracy vs Number of Nodes")
# plt.show()
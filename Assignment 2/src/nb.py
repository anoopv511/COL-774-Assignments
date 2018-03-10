import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from functions import *

# Create/Load N-gram Dictionary
# [ngram_type = 0 - tokenSize = n | 1 - tokenSize = 1 or n | otherwise - tokenSize <= n]
def get_diction(x,fname,ngram_type=1,n=1,case=False,save_data=True,load_data=False,dictName="Dictionary"):
    if load_data:
        diction = load(fname)
        print(dictName + " loaded")        
    else:
        if n == 1:
            words = []
            for doc in x:
                words += set(tokenize(doc,case))
            words = set(words)
            diction = dict(zip(list(words),np.arange(len(words))))
        else:
            words = [] if ngram_type == 0 else [[],[]] if ngram_type == 1 else [[] for i in range(n)]
            for doc in x:
                w = tokenize(doc,case)
                if ngram_type == 0:
                    words += set(ngram(w,n))
                elif ngram_type == 1:
                    words[0] += set(w)
                    words[1] += set(ngram(w,n))
                else:
                    words[0] += set(w)
                    for i in range(1,n):
                        words[i] += set(ngram(w,i+1))
            features = set([i for l in words for i in l]) if ngram_type != 0 else set(words)
            diction = dict(zip(list(features),np.arange(len(features))))
        if save_data: save(diction,fname)
        print(dictName + " created")
    return diction

# Calculate Feature Counts/Sums per class
def features_count(x,y,fname,diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=False,featureName="Word counts"):
    if load_data:
        features_per_class = load(fname)
        print(featureName + " loaded")
    else:
        features_per_class = np.zeros((labels.shape[0],len(diction.keys())),dtype=dtype)
        if n == 1 and featureType == "count" and classifierType == "NB":
            for idx, doc in enumerate(x):
                doc_words = tokenize(doc,case)
                unique_doc_words, word_count = np.unique(doc_words,return_counts=True)
                for idy, w in enumerate(unique_doc_words):
                    features_per_class[class_label_map[y[idx]],diction[w]] += word_count[idy]
        elif featureType == "sum" and classifierType == "CNB":
            for idx, doc in enumerate(x):
                classes = np.arange(labels.shape[0])
                doc_words = tokenize(doc,case)
                unique_doc_words, word_count = np.unique(doc_words,return_counts=True)
                for idy, w in enumerate(unique_doc_words):
                    features_per_class[~(classes == class_label_map[y[idx]]),diction[w]] += featureMap[w][idx]*word_count[idy]
        elif featureType == "sum" and classifierType == "NB":
            for idx, doc in enumerate(x):
                doc_words = tokenize(doc,case)
                unique_doc_words, word_count = np.unique(doc_words,return_counts=True)
                for idy, w in enumerate(unique_doc_words):
                    features_per_class[class_label_map[y[idx]],diction[w]] += featureMap[w][idx]*word_count[idy]
        else:
            for idx, doc in enumerate(x):
                doc_words = tokenize(doc,case)
                unique_doc_words, word_count = np.unique(doc_words,return_counts=True)
                for idy, w in enumerate(unique_doc_words):
                    features_per_class[class_label_map[y[idx]],diction[w]] += word_count[idy]
                for i in range(1,n):
                    doc_features = ngram(doc_words,i+1)
                    unique_doc_features, feature_count = np.unique(doc_features,return_counts=True)
                    for idy, f in enumerate(unique_doc_features):
                        features_per_class[class_label_map[y[idx]],diction[f]] += feature_count[idy]
        save(features_per_class,fname)
        print(featureName + " calculated")    
    return features_per_class

# Naive Bayes Prediction
def predict(doc,log_theta,log_phi,diction,classes):
    words = tokenize(doc)
    probs = np.zeros(len(classes))
    for idx, w in enumerate(words):
        try:
            probs += log_theta[:,diction[w]]
        except KeyError:
            continue
    probs += log_phi
    return classes[np.argmax(probs)]

# Complementary Naive Bayes Prediction
def cpredict(doc,log_theta,diction,classes):
    words = tokenize(doc)
    probs = np.zeros(len(classes))
    for idx, w in enumerate(words):
        try:
            probs += log_theta[:,diction[w]]
        except KeyError:
            continue
    return classes[np.argmin(probs)]

# Calculate Accuracy
def accuracy(x,y,log_theta,log_phi,diction,classes,predictMethod="NB",train=False,modelName="Naive Bayes"):
    pred = np.array([predict(doc,log_theta,log_phi,diction,classes) for doc in x]) if predictMethod == "NB" else np.array([cpredict(doc,log_theta,diction,classes) for doc in x])
    accuracy = (y == pred).sum()/float(y.shape[0])
    if train:
        print("{0} Train Accuracy = {1}".format(modelName,accuracy))
    else:
        print("{0} Test Accuracy = {1}".format(modelName,accuracy))
    return pred

#Confusion Matrix
def get_conf_matrix(y,pred,classes,class_label_map,eval=True):
    conf_matrix = np.zeros((classes.shape[0],classes.shape[0]),dtype=np.int64)
    for l, p in zip(y,pred):
        conf_matrix[class_label_map[p],class_label_map[l]] += 1

    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    _ = sns.heatmap(conf_matrix,annot=True,cmap="Blues",xticklabels=classes,yticklabels=classes,fmt='g')
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix",y=1.08)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.show()

    # Evaluation
    if eval:
        true_pos = conf_matrix.trace()
        false_pos = conf_matrix.sum(axis=0)
        false_neg = conf_matrix.sum(axis=1)
        prec = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        micro_prec = true_pos.sum()/(true_pos.sum()+false_pos.sum())
        micro_recall = true_pos.sum()/(true_pos.sum()+false_neg.sum())
        macro_prec = prec.sum()/conf_matrix.shape[0]
        macro_recall = recall.sum()/conf_matrix.shape[0]
        micro_f1 = 2*micro_prec*micro_recall/(micro_prec+micro_recall)
        macro_f1 = 2*macro_prec*macro_recall/(macro_prec+macro_recall)
        print("Micro F1 score = {0:.4f}".format(micro_f1))
        print("Macro F1 score = {0:.4f}".format(macro_f1))

# Get length of documents using given tokenization method
def get_doc_len(x,fname,case=False,save_data=True,load_data=False):
    if load_data:
        doc_len = load(fname)
        print("Document Lengths loaded")
    else:
        doc_len = []
        for doc in x:
            doc_len.append(len(tokenize(doc)))
        if save_data: save(doc_len,fname)
        print("Document Lengths created")            
    return doc_len

# Create/Load Inverted Index for Documents
def get_inv_index(diction,x,doc_len,fname,save_data=True,load_data=False):
    if load_data:
        inverted_index = load(fname)
        print("Inverted Index loaded")        
    else:
        words = list(diction.keys())
        inverted_index = dict(zip(words,[{} for _ in range(len(words))]))
        for idx, doc in enumerate(x):
            w, c = np.unique(tokenize(doc),return_counts=True)
            for idy in range(len(w)):
                inverted_index[w[idy]][idx] = c[idy]
        if save_data: save(inverted_index,fname)
        print("Inverted Index created")
    return inverted_index

# Calculate TF
def calc_tf(x,inverted_index,doc_len,words,fname,mod=None,save_data=True,load_data=False,featureMapName="TF"):
    if load_data:
        tf = load(fname)
        print(featureMapName + " loaded")
    else:
        tf = dict(zip(words,[{} for _ in range(len(words))]))
        for idx, w in enumerate(words):
            for doc_id in inverted_index[w].keys():
                tf[w][doc_id] = inverted_index[w][doc_id]/float(doc_len[doc_id]) if mod == None else mod(inverted_index[w][doc_id]/float(doc_len[doc_id]))
        if save_data: save(tf,fname)
        print(featureMapName + " calculated")
    return tf

# Calculate TF-IDF
def calc_tfidf(x,inverted_index,doc_len,words,fname,norm=None,save_data=True,load_data=False,featureMapName="TF-IDF"):
    if load_data:
        tfidf = load(fname)
        print(featureMapName + " loaded")
    else:
        idf = [np.log(x.shape[0]/len(inverted_index[w].keys())) for w in words]
        tfidf = dict(zip(words,[{} for _ in range(len(words))]))
        for idx, w in enumerate(words):
            for doc_id in inverted_index[w].keys():
                tf = inverted_index[w][doc_id]/doc_len[doc_id]
                tfidf[w][doc_id] = tf*(idf[idx]+1)
        if norm is None and save_data:
            save(tfidf,fname)
            print(featureMapName + " calculated")            
        else:
            normalized_tfidf = dict(zip(words,[{} for _ in range(len(words))]))
            for idx, doc in enumerate(x):
                w = np.unique(tokenize(doc))
                total_tfidf_per_doc = 0
                for idy in range(len(w)):
                    total_tfidf_per_doc += (tfidf[w[idy]][idx])**2 if norm == "L2" else tfidf[w[idy]][idx]
                for idy in range(len(w)):
                    normalized_tfidf[w[idy]][idx] = tfidf[w[idy]][idx]/np.sqrt(total_tfidf_per_doc) if norm == "L2" else tfidf[w[idy]][idx]/total_tfidf_per_doc
            if save_data: save(normalized_tfidf,fname)
            tfidf = normalized_tfidf
            print(featureMapName + " calculated")
    return tfidf

if __name__ == "__main__":
    ############ Part (a) ############

    # Train and Test Data
    x_train = np.loadtxt('../data/imdb/imdb_train_text.txt',delimiter='\n',dtype='str',comments=None)
    y_train = np.loadtxt('../data/imdb/imdb_train_labels.txt',delimiter='\n',dtype=np.int64)
    x_test = np.loadtxt('../data/imdb/imdb_test_text.txt',delimiter='\n',dtype='str',comments=None)
    y_test = np.loadtxt('../data/imdb/imdb_test_labels.txt',delimiter='\n',dtype=np.int64)
    print("Loaded Train and Test Data")

    # Create Dictionary
    # diction = get_diction(x_train,"../data/tmp/diction.pickle",case=False,save_data=True,load_data=False)
    diction = get_diction(x_train,"../data/tmp/diction.pickle",case=False,save_data=True,load_data=True)

    # Calculate Word Counts per Class
    labels, label_counts = np.unique(y_train,return_counts=True)
    class_label_map = dict(zip(labels,np.arange(labels.shape[0])))
    # word_counts_per_class = features_count(x_train,y_train,"../data/tmp/word_counts_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=False,featureName="Word counts")
    word_counts_per_class = features_count(x_train,y_train,"../data/tmp/word_counts_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=True,featureName="Word counts")

    # Calculate Parameters
    theta = np.divide(word_counts_per_class+1,word_counts_per_class.sum(axis=1).reshape(-1,1)+len(diction.keys()))
    phi = label_counts/y_train.shape[0]
    log_theta = np.log(theta)
    log_phi = np.log(phi)

    # Train Accuracy
    train_pred = accuracy(x_train,y_train,log_theta,log_phi,diction,labels,"NB",True,"Naive Bayes")

    # Test Accuracy
    test_pred = accuracy(x_test,y_test,log_theta,log_phi,diction,labels,"NB",False,"Naive Bayes")

    # Train Accuracy = 0.68448
    # Test Accuracy = 0.38752

    del x_train,x_test,diction,word_counts_per_class

    ##################################

    ############ Part (b) ############

    # Random Accuracy
    np.random.seed(0)
    random_labels = np.random.randint(0,labels.shape[0],y_test.shape[0])
    random_accuracy = (y_test == random_labels).sum()/float(y_test.shape[0])
    print("Random accuracy = {0}".format(random_accuracy))

    # MaxCount Accuracy
    maxcount_accuracy = (y_test == labels[label_counts.argmax()]).sum()/float(y_test.shape[0])
    print("MaxCount accuracy = {0}".format(maxcount_accuracy))

    # Random Accuracy = 0.07264
    # MaxCount Accuracy = 0.20088

    ##################################

    ############ Part (c) ############

    get_conf_matrix(y_test,test_pred,labels,class_label_map,True)

    ##################################

    ############ Part (d) ############

    # Stemmed Train and Test Data
    x_train = np.loadtxt('../data/imdb/stem_imdb_train_text.txt',delimiter='\n',dtype='str',comments=None)
    x_test = np.loadtxt('../data/imdb/stem_imdb_test_text.txt',delimiter='\n',dtype='str',comments=None)
    print("Loaded Stemmed Train and Test Data")

    # Create Dictionary
    # diction = get_diction(x_train,"../data/tmp/stem_diction.pickle",case=False,save_data=True,load_data=False,dictName="Stemmed Dictionary")
    diction = get_diction(x_train,"../data/tmp/stem_diction.pickle",case=False,save_data=True,load_data=True,dictName="Stemmed Dictionary")

    # Calculate Stemmed Word Counts per Class
    # word_counts_per_class = features_count(x_train,y_train,"../data/tmp/stem_word_counts_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=False,featureName="Stemmed Word counts")
    word_counts_per_class = features_count(x_train,y_train,"../data/tmp/stem_word_counts_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=True,featureName="Stemmed Word counts")

    # Calculate Parameters
    theta = np.divide(word_counts_per_class+1,word_counts_per_class.sum(axis=1).reshape(-1,1)+len(diction.keys()))
    phi = label_counts/y_train.shape[0]
    log_theta = np.log(theta)
    log_phi = np.log(phi)

    # Train Accuracy
    train_pred = accuracy(x_train,y_train,log_theta,log_phi,diction,labels,"NB",True,"Stemmed Naive Bayes")

    # Test Accuracy
    test_pred = accuracy(x_test,y_test,log_theta,log_phi,diction,labels,"NB",False,"Stemmed Naive Bayes")

    # Train Accuracy = 0.6798
    # Test Accuracy = 0.38684

    get_conf_matrix(y_test,test_pred,labels,class_label_map,True)

    del diction,word_counts_per_class

    ##################################
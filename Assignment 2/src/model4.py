from nb import *

# Normalized TF-IDF

# Stemmed Train and Test Data
x_train = np.loadtxt('../data/imdb/stem_imdb_train_text.txt',delimiter='\n',dtype='str',comments=None)
y_train = np.loadtxt('../data/imdb/imdb_train_labels.txt',delimiter='\n',dtype=np.int64)
x_test = np.loadtxt('../data/imdb/stem_imdb_test_text.txt',delimiter='\n',dtype='str',comments=None)
y_test = np.loadtxt('../data/imdb/imdb_test_labels.txt',delimiter='\n',dtype=np.int64)
print("Loaded Stemmed Train and Test Data")

# Create Inverted Index for Documents
diction = get_diction(x_train,"../data/tmp/stem_diction.pickle",case=False,save_data=True,load_data=False,dictName="Stemmed Dictionary")
words = list(diction.keys())
doc_len = get_doc_len(x_train,"../data/tmp/doc_len.pickle",case=False,save_data=True,load_data=False)
inverted_index = get_inv_index(diction,x_train,doc_len,"../data/tmp/inverted_index.pickle",save_data=True,load_data=False)

# Calculate Normalized TF-IDF
norm_tfidf = calc_tfidf(x_train,inverted_index,doc_len,words,"../data/tmp/norm_tfidf.pickle",norm="L2",save_data=True,load_data=False,featureMapName="Normalized TF-IDF")

# Calculate Normalized TF-IDF sum per Class
labels, label_counts = np.unique(y_train,return_counts=True)
class_label_map = dict(zip(labels,np.arange(labels.shape[0])))
tfidf_sum_per_class = features_count(x_train,y_train,"../data/tmp/norm_tfidf_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=norm_tfidf,featureType="sum",ngram_type=1,n=1,dtype=np.float64,save_data=True,load_data=False,featureName="Normalized TF-IDF sum")

# Calculate Parameters
theta = np.divide(tfidf_sum_per_class+1,tfidf_sum_per_class.sum(axis=1).reshape(-1,1)+len(diction.keys()))
phi = label_counts/y_train.shape[0]
log_theta = np.log(theta)
log_phi = np.log(phi)

# Train Accuracy
train_pred = accuracy(x_train,y_train,log_theta,log_phi,diction,labels,"NB",True,"Normalized TF-IDF Naive Bayes")

# Test Accuracy
test_pred = accuracy(x_test,y_test,log_theta,log_phi,diction,labels,"NB",False,"Normalized TF-IDF Naive Bayes")

# Train Accuracy = 0.384
# Test Accuracy = 0.34604

del words,diction,doc_len,inverted_index,norm_tfidf,tfidf_sum_per_class
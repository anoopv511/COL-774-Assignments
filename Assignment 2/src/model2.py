from nb import *

# Stemmed Train and Test Data
x_train = np.loadtxt('../data/imdb/stem_imdb_train_text.txt',delimiter='\n',dtype='str',comments=None)
y_train = np.loadtxt('../data/imdb/imdb_train_labels.txt',delimiter='\n',dtype=np.int64)
x_test = np.loadtxt('../data/imdb/stem_imdb_test_text.txt',delimiter='\n',dtype='str',comments=None)
y_test = np.loadtxt('../data/imdb/imdb_test_labels.txt',delimiter='\n',dtype=np.int64)
print("Loaded Stemmed Train and Test Data")

# Create Dictionary
diction = get_diction(x_train,"../data/tmp/stem_diction.pickle",case=False,save_data=True,load_data=False,dictName="Stemmed Dictionary")

# Calculate Stemmed Word Counts per Class
labels, label_counts = np.unique(y_train,return_counts=True)
class_label_map = dict(zip(labels,np.arange(labels.shape[0])))
word_counts_per_class = features_count(x_train,y_train,"../data/tmp/stem_word_counts_per_class.pickle",diction,labels,class_label_map,classifierType="NB",featureMap=None,featureType="count",ngram_type=1,n=1,dtype=np.int64,case=False,save_data=True,load_data=False,featureName="Stemmed Word counts")

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

# Save Model 2
save((diction,log_theta,log_phi),"../data/tmp/model2.pickle")

del diction,word_counts_per_class
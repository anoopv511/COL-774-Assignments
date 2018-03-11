import numpy as np
from nltk.tokenize import RegexpTokenizer
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#initializing tokenizer and stemmer
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()

def tokenize(raw,case=False):
    if not case:
        raw = raw.lower().replace("<br />"," ")
    else:
        raw = raw.replace("<br />"," ")
    return tokenizer.tokenize(raw)

def save(obj,fname):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load(fname):
    with open(fname,'rb') as f:
        obj = pickle.load(f)
    return obj

def ngram(doc,n):
    assert n >= 2, "N-gram should have atleast n = 2"
    ng = []
    for idx in range(len(doc)-(n-1)):
        ng.append(" ".join(doc[idx:idx+n]))
    return ng

def getStemmedDocument(doc):
    raw = doc.lower().replace("<br /><br />", " ")
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [token for token in tokens if token not in en_stop]
    stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens]
    stem_doc = ' '.join(stemmed_tokens)
    return stem_doc
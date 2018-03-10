import numpy as np
from nltk.tokenize import RegexpTokenizer
import pickle

tokenizer = RegexpTokenizer(r'\w+')

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
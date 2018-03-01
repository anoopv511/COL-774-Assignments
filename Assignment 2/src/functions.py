import numpy as np
from nltk.tokenize import RegexpTokenizer
import pickle

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(raw):
    raw = raw.replace("<br />"," ")
    return tokenizer.tokenize(raw)

def save(obj,fname):
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load(fname):
    with open(fname,'rb') as f:
        obj = pickle.load(f)
    return obj
import pandas as pd
import numpy as np
from functions import *

def get_encodings(x_train,median_spit=True):
    data = x_train.copy(deep=True)
    cat_cols = data.select_dtypes(['object']).columns
    for col in cat_cols:
        data[col] = data[col].astype('category')
    encodings = dict(zip(cat_cols,[dict(zip(data[col].cat.categories,range(len(data[col].cat.categories)))) for col in cat_cols]))
    if median_spit:
        num_cols = list(set(data.columns) - set(cat_cols))
        for col in num_cols:
            encodings[col] = np.median(data[col])
    return encodings

def preprocess(x,encoding):
    cat_cols = x.select_dtypes(['object']).columns
    for col in cat_cols:
        x[col] = x[col].astype('category')
    for col in encoding:
        if type(encoding[col]) == dict:
            x[col].replace(encoding[col],inplace=True)
        else:
            x[col] = (x[col] > encoding[col]).astype(np.int64)
    return x
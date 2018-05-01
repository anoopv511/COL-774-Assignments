import numpy as np
import os
from functions import *
import pandas as pd

path = "../combo/results/"
files = os.listdir(path)
print(files)
results = []
for f in files:
	results.append(pd.read_csv(path + f).as_matrix())

combined_result = pd.read_csv(path + files[0]).as_matrix()

for idx in range(results[0].shape[0]):
	votes = [res[idx,1] for res in results]
	val, counts = np.unique(votes,return_counts=True)
	combined_result[idx,0] = idx
	combined_result[idx,1] = val[np.argmax(counts)]

with open("../combo/combo.csv","w") as out:
	out.write("ID,CATEGORY\n")
	for idx in range(results[0].shape[0]):
		out.write("{0},{1}\n".format(combined_result[idx,0],combined_result[idx,1]))
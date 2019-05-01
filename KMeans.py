from sklearn.cluster import KMeans
import numpy as np
from numpy import array

NUM_CLASS = 16

train_X = []
read_feature = open("train_feature.txt", "r")

lines = read_feature.readlines()
print(len(lines))
for line in lines:
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  train_X.append(line_split)
train_X = array(train_X)

kmeans = KMeans(n_clusters=NUM_CLASS, random_state=0).fit(train_X)
labels = kmeans.labels_

label_f =  open('cluster_label.txt', 'w+')
np.savetxt(label_f, labels, fmt='%d')

#  topk closest pojects of controid
count = 16
topk = 11
indexs = np.zeros((count,topk), dtype=int)
for j in range(count):
    d = kmeans.transform(train_X)[:, j]
    ind = np.argsort(d)[::][:topk]
    indexs[j] = ind
closed_f =  open('closed_index.txt', 'w+')
np.savetxt(closed_f, indexs, fmt='%d')

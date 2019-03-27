from sklearn.cluster import KMeans
import numpy as np
from numpy import array

NUM_POINT = 1024
NUM_CLASS = 40

# train_X = np.empty([TRAIN_NUM, NUM_POINT], dtype=float)
train_X = []
read_feature = open("train_cluster.txt", "r")
count = 0

lines = read_feature.readlines()
for line in lines:
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  train_X.append(line_split)
train_X = array(train_X)

kmeans = KMeans(n_clusters=NUM_CLASS, random_state=0).fit(train_X)
labels = kmeans.labels_

label_f =  open('cluster_label.txt', 'w+')
np.savetxt(label_f, labels, fmt='%d')
# array([1, 1, 1, 0, 0, 0], dtype=int32)

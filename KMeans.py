from sklearn.cluster import KMeans
import numpy as np
from numpy import array

TRAIN_NUM = 9800
NUM_POINT = 1024
NUM_CLASS = 40

train_X = np.empty([TRAIN_NUM, NUM_POINT], dtype=float)
read_feature = open("train_cluster.txt", "r")
count = 0
for l in range(TRAIN_NUM):
  line = read_feature.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  train_X[l] = array(line_split)

kmeans = KMeans(n_clusters=NUM_CLASS, random_state=0).fit(train_X)
labels = kmeans.labels_

label_f =  open('cluster_label.txt', 'w+')
np.savetxt(label_f, labels, fmt='%d')
# array([1, 1, 1, 0, 0, 0], dtype=int32)

# from sklearn import datasets
# digits = datasets.load_digits()
# # Take the first 500 data points: it's hard to see 1500 points
# X = digits.data[:500]
import numpy as np
from numpy import array

num = 2000
NUM_POINT = 256
X = np.empty([num, NUM_POINT], dtype=float)
read_feature = open("feature.txt", "r")
count = 0
for l in range(num):
  line = read_feature.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  X[count] = array(line_split)
  count = count + 1

print(len(X))
X = X[:num]

y = np.empty([num], dtype=int)
read_label = open("label.txt", "r")
count = 0
for l in range(num):
  line = read_label.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  y[count] = array(line_split)
  count = count + 1
y = y[:num]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

target_ids = range(len(y))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c in zip(target_ids, colors):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c)
plt.legend()
plt.show()

# from sklearn import datasets
# digits = datasets.load_digits()
# # Take the first 500 data points: it's hard to see 1500 points
# X = digits.data[:500]
import numpy as np
from numpy import array
from matplotlib import pyplot as plt

X = []
read_feature = open("feature.txt", "r")
lines = read_feature.readlines()
for line in lines:
  line = line.split(" ")
  line = [float(i) for i in line]
  X.append(line)
X = np.array(X)

print(len(X))

y = []
read_label = open("label.txt", "r")
lines = read_label.readlines()
y = [float(i) for i in lines]
y = np.array(y)

print(len(y))

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

target_ids = range(len(y))

plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
# for i, c in zip(target_ids, colors):
for i in target_ids:
    if i<=15:
      plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=np.random.rand(1,3))
plt.legend()
plt.show()

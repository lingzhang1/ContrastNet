import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from tsne import bh_sne

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
y = [int(i) for i in lines]

y = np.array(y)
print(len(y))

tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)
target_ids = range(len(y))
plt.figure(figsize=(6, 5))
colors = ['red', 'blue', 'navy', 'green', 'violet', 'brown', 'gold', 'lime', 'teal', 'olive']
# c=np.random.rand(1,3)
for i in target_ids:
    if i<10:
      plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=colors[i])
plt.legend()
plt.show()

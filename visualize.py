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

# X_tmp = X
# X = []
# for i in range(len(X_tmp)):
#     if i % 5 == 0:
#         X.append(X_tmp[i])

X = np.array(X)

print(len(X))

y = []
read_label = open("label.txt", "r")
lines = read_label.readlines()
y = [int(i) for i in lines]

# y_label = y
# y = []
# for i in range(len(y_label)):
#     if i % 5 == 0:
#         y.append(y_label[i])

y = np.array(y)
print(len(y))


# x_data = X.astype('float64')
# x_data = x_data.reshape((x_data.shape[0], -1))
# y_data = y
# # perform t-SNE embedding
# vis_data = bh_sne(x_data)
# # plot the result
# vis_x = vis_data[:, 0]
# vis_y = vis_data[:, 1]
# plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()


tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

target_ids = range(len(y))

plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
# for i, c in zip(target_ids, colors):
colors = ['red', 'blue', 'navy', 'green', 'violet', 'brown', 'gold', 'lime', 'teal', 'olive']
# c=np.random.rand(1,3)
for i in target_ids:
    if i<10:
      plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=colors[i])
plt.legend()
plt.show()


# from yellowbrick.text import TSNEVisualizer
# from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# tfidf  = TfidfVectorizer()
# docs = X
# labels = y
#
# cates = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
#
# colors = ['red', 'blue', 'navy', 'green', 'violet', 'brown', 'gold', 'lime', 'teal', 'olive']
# # colors = [[9, 97, 239], [9, 239, 227], [239, 9, 20], [239, 9, 20], [9, 239, 43], [9, 239, 43], [239, 223, 9], [239, 120, 9], [146, 9, 239], [9, 162, 239]]
#
# tsne = TSNEVisualizer(color = colors)
# tsne.fit(docs, [cates[c] for c in labels])
# tsne.poof()

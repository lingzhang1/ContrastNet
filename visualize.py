# from sklearn import datasets
# digits = datasets.load_digits()
# # Take the first 500 data points: it's hard to see 1500 points
# X = digits.data[:500]
# y = digits.target[:500]

X = []
read_feature = open("feature.txt", "r")
for line in read_feature:
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  X.append(line_split)
print(X.shape)
X = X[:1000]

y = []
read_label = open("label.txt", "r")
for line in read_label:
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  y.append(line_split)
y = y[:1000]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

target_ids = range(len(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.show()

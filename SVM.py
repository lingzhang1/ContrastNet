import numpy as np
from numpy import array
from sklearn.svm import SVC

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

clf = SVC(gamma='auto')
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    
clf.score(X, y)
# print(clf.predict([[-0.8, -1]]))

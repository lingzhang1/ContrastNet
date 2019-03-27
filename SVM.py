import numpy as np
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

# train featrue
train_X = []
read_feature = open("train_feature.txt", "r")
lines = read_feature.readlines()
for line in lines:
    line_split = line.split(" ")
    line_split = [float(i) for i in line_split]
    train_X.append(line_split)
train_X = array(train_X)
train_X = train_X[0:9500, :]

# train label
train_y = []
read_label = open("train_label.txt", "r")
lines = read_label.readlines()
for line in lines:
    line_split = line.split(" ")
    line_split = [float(i) for i in line_split]
    train_y.append(line_split)
train_y = array(train_y)
train_y = train_y[0:9500, :]

# test featrue
X = []
read_feature = open("feature.txt", "r")
lines = read_feature.readlines()
for line in lines:
    line_split = line.split(" ")
    line_split = [float(i) for i in line_split]
    X.append(line_split)
X = array(X)
X = X[0:1400, :]

# test label
y = []
read_label = open("label.txt", "r")
lines = read_label.readlines()
for l in lines:
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  y.append(line_split)
y = array(y)
y = y[0:1400, :]

print('Training SVM...')
clf = SVC(gamma='auto')
clf.fit(train_X, train_y)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print('Testing SVM...')
print(clf.score(X, y))

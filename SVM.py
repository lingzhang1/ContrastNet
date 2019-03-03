import numpy as np
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

test_num = 2400
train_num = 9800
NUM_POINT = 256

# train featrue
train_X = np.empty([train_num, NUM_POINT], dtype=float)
read_feature = open("train_feature.txt", "r")
count = 0
for l in range(train_num):
  line = read_feature.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  train_X[l] = array(line_split)

# train_X = normalize(train_X, norm='l2', axis=1, copy=True, return_norm=False)

# train label
train_y = np.empty([train_num], dtype=int)
read_label = open("train_label.txt", "r")
for l in range(train_num):
  line = read_label.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  train_y[l] = array(line_split)


# test featrue
X = np.empty([test_num, NUM_POINT], dtype=float)
read_feature = open("feature.txt", "r")
for l in range(test_num):
  line = read_feature.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  X[l] = array(line_split)

# X = normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
# X = array(X)

# test label
y = np.empty([test_num], dtype=int)
read_label = open("label.txt", "r")
for l in range(test_num):
  line = read_label.readline()
  line_split = line.split(" ")
  line_split = [float(i) for i in line_split]
  y[l] = array(line_split)

print('Training SVM...')
clf = SVC(gamma='auto')
clf.fit(train_X, train_y)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print('Testing SVM...')
print(clf.score(X, y))

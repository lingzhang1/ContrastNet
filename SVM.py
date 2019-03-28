import numpy as np
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

num_votes = 12

results = []

# train label
train_y = []
read_label = open("features/train_label.txt", "r")
train_y = read_label.readlines()
train_y = [float(i) for i in train_y]
train_y = array(train_y)

# test label
y = []
read_label = open("features/label.txt", "r")
y = read_label.readlines()
y = [float(i) for i in y]
y = array(y)

for i in range(num_votes):
    print("VOTE = ", num_votes)
    # train featrue
    train_X = []
    read_feature = open("features/train_feature_"+str(num_votes)+".txt", "r")
    lines = read_feature.readlines()
    for line in lines:
        line_split = line.split(" ")
        line_split = [float(i) for i in line_split]
        train_X.append(line_split)
    train_X = array(train_X)

    # test featrue
    X = []
    read_feature = open("features/feature_"+str(num_votes)+".txt", "r")
    lines = read_feature.readlines()
    for line in lines:
        line_split = line.split(" ")
        line_split = [float(i) for i in line_split]
        X.append(line_split)
    X = array(X)

    print('Training SVM...')
    clf = SVC(gamma='auto')
    clf.fit(train_X, train_y)

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    print('Testing SVM...')
    results.append(clf.score(X, y))
    print(clf.score(X, y))

results = array(results)
print(np.mean(results))

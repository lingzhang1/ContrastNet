import numpy as np
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

num_votes = 12

results = []

# train label
train_y = []
read_label = open("features/train_label.npy", "r")
train_y = np.load(read_label)

# test label
y = []
read_label = open("features/label.npy", "r")
y = np.load(read_label)

for vote_id in range(num_votes):
    print("VOTE = ", vote_id)
    # train featrue
    train_X = []
    read_feature = open("features/train_feature_"+str(vote_id)+".npy", "r")
    train_X = np.load(read_feature)

    # test featrue
    X = []
    read_feature = open("features/feature_"+str(vote_id)+".npy", "r")
    X = np.load(read_feature)

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

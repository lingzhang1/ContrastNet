import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

TRAIN_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/shapenet_cut/train_files.txt'))

gt_labels = np.empty([len(TRAIN_FILES),1], dtype=int)

print('Loading groundtruth labels ...')
for fn in range(len(TRAIN_FILES)):
    _, _, label = provider.loadDataFile_cut_2(TRAIN_FILES[fn], False)
    label = np.squeeze(label)
    gt_labels[fn] = label

CLASS_NUM = 16
classes = np.zeros((CLASS_NUM, CLASS_NUM))

print('Loading cluster labels ...')
cluster_f = open("cluster_label.txt", "r")
cluster_labels = cluster_f.readlines()
cluster_labels = [int(i) for i in cluster_labels]

print('Caculating proportion ...')
for i in range(len(cluster_labels)):
    gt_l = gt_labels[i]
    ct_l = cluster_labels[i]
    classes[ct_l][gt_l] = classes[ct_l][gt_l] + 1

proportion = np.zeros(CLASS_NUM)

for i in range(CLASS_NUM):
    max = np.amax(classes[i])
    sum =  np.sum(classes[i])
    proportion[i] = max / sum
print(proportion)
print('total = ',np.sum(proportion) / CLASS_NUM)

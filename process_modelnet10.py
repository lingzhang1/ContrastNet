import os
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
import h5py

out_folder = 'data/modelnet10_hdf5'
os.mkdir(out_folder)

mainPath = 'data/ModelNet10'
class_folders = [join(mainPath, o) for o in os.listdir(mainPath)
                    if isdir(join(mainPath,o))]

f_test = open(out_folder + '/test_files.txt', 'w+')
f_train = open(out_folder + '/train_files.txt', 'w+')

print(class_folders)

for c in class_folders:
    test_path = join(c, 'test')
    test_files = [f for f in listdir(test_path) if isfile(join(test_path, f))]
    print('test_files = ', test_files[0])
    print('test_files = ', test_files[0][0:-4])
    test_name = [join(out_folder, f[0:-4]+'.h5') for f in listdir(test_path) if isfile(join(test_path, f))]
    test_name = np.array(test_name)
    print('test_name[0] = ',test_name[0])
    np.savetxt(f_test, test_name, fmt='%s')
    for name in test_files:
        path = join(test_path, name)
        print('path = ', path)
        f = open(path, 'w+')
        lines = f.readlines()
        lines = lines[2:]
        print(lines)
        print('lines = ',len(lines))
        data = []
        for l in lines:
            l = l.split(' ')
            l = [float(i) for i in l]
            data.append(l)
        data = np.array(data)
        hf = h5py.File(join(out_folder, name[0:-4]+'.h5'), 'w')
        hf.create_dataset('data', data=data)

    train_path = join(c, 'train')
    train_files = [f for f in listdir(train_path) if isfile(join(train_path, f))]

    train_name = [join(out_folder, f[0:-4]+'.h5') for f in listdir(train_path) if isfile(join(train_path, f))]
    train_name = np.array(train_name)
    np.savetxt(f_train, train_name, fmt='%s')
    for name in train_files:
        path = join(train_path, name)
        f = open(path, 'w+')
        lines = f.readlines()
        lines = lines[2:]
        data = []
        for l in lines:
            l = l.split(' ')
            l = [float(i) for i in l]
            data.append(l)
        data = np.array(data)

        hf = h5py.File(join(out_folder, name[0:-4]+'.h5'), 'w')
        hf.create_dataset('data', data=data)

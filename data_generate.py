import numpy as np
import random
import h5py
import math

x = np.random.normal(0, 10, 10000).T
y = np.random.normal(0, 5, 10000).T

features = np.stack([x, y], axis = 1)
features0 = features
print(features.shape, '\n', features[0:10])

eig_vec = np.array([[1, 0], [0, 1]], dtype = np.float)
print(eig_vec.shape, '\n', eig_vec)

labels = np.tile(eig_vec.ravel(), (10000,1))
print(labels.shape, '\n', labels)

data = locals()

for i in range(10):
    ang = (1+i) * 10 / 180 * math.pi
    rot = np.array([[math.cos(ang), -math.sin(ang)],
                    [math.sin(ang), math.cos(ang)]],
                    dtype = np.float)
    print('rot shape ', rot.shape)
    data['data' + str(i)] = np.dot(features0, rot)
    print('data shape: ', data['data' + str(i)].shape)
    labels_rot = np.tile(rot.ravel(), (10000,1))
    features = np.vstack((features, data['data' + str(i)]))
    labels = np.vstack((labels, labels_rot))

np.random.shuffle(features)
print('features shape: ', features.shape)
np.random.shuffle(labels)
print(labels[0:10], '\n', labels.shape)

f = h5py.File('sim_data.h5', 'w')
f.create_dataset('features', data = features)
f.create_dataset('labels', data = labels)
f.close()

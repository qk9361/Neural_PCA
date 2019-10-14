import numpy as np
import random
import h5py
import math
from sklearn.decomposition import PCA, KernelPCA
import numpy.linalg as LA
import matplotlib.pyplot as plt

x = np.random.normal(0, 10, 1000).T
y = np.random.normal(0, 5, 1000).T

features = np.stack([x, y], axis = 1)
print(features.shape, '\n', features[0:10])

eig_vec = np.array([[1, 0], [0, 1]], dtype = np.float)
# ang = np.random.normal(45, 45, 10000).T
ang = [45, 135, 225, 315]
# rot = np.array([[math.cos(ang), -math.sin(ang)],
#                 [math.sin(ang), math.cos(ang)]],
#                 dtype = np.float)

# eig_vec_rot = np.empty([10000, 2])
# features_rot = np.empty([10000, 2])
features_rot = locals()

for i in range(len(ang)):
    rot = np.array([[math.cos(ang[i]), -math.sin(ang[i])],
                    [math.sin(ang[i]), math.cos(ang[i])]],
                    dtype = np.float)
    # eig_vec_rot[i,:] = rot[:,0].T
    features_rot['features_rot'+str(i)] = np.dot(features, rot)

features = features_rot0
for i in range(1,len(ang)):
    features = np.vstack([features, features_rot['features_rot'+str(i)]])

# eig_vec_rot = np.tile(rot[:,0].ravel(), (10000,1))
# features_rot = np.dot(features, rot)
#
# eig_vec_rot = np.vstack([np.tile([1,0], (10000,1)), eig_vec_rot])
# features_rot = np.vstack([features, features_rot])
# print(eig_vec_rot.shape)
print(features.shape)
class1 = np.tile([1,0,0,0], (1000,1))
class2 = np.tile([0,1,0,0], (1000,1))
class3 = np.tile([0,0,1,0], (1000,1))
class4 = np.tile([0,0,0,1], (1000,1))
labels = np.vstack([class1,class2,class3,class4])
# data = locals()

# for i in range(10):
#     ang = (1+i) * 10 / 180 * math.pi
#     rot = np.array([[math.cos(ang), -math.sin(ang)],
#                     [math.sin(ang), math.cos(ang)]],
#                     dtype = np.float)
#     print('rot shape ', rot.shape)
#     data['data' + str(i)] = np.dot(features0, rot)
#     print('data shape: ', data['data' + str(i)].shape)
#     labels_rot = np.tile(rot.ravel(), (10000,1))
#     features = np.vstack((features, data['data' + str(i)]))
#     labels = np.vstack((labels, labels_rot))

np.random.shuffle(features)
print('features shape: ', features.shape)
np.random.shuffle(labels)
print(labels[0:10], '\n', labels.shape)

f = h5py.File('sim_data.h5', 'w')
f.create_dataset('features', data = features)
f.create_dataset('labels', data = labels)
f.close()

# pca = PCA(n_components = 1)
# features_pca = pca.fit_transform(features)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
# print(pca.components_)
#
# plt.figure()
# plt.subplot(3,1,1)
# plt.scatter(features[:,0], features[:,1])
# plt.subplot(3,1,2)
# plt.plot(features_pca[:,0])
#
# pred = pca.fit_transform(features)
# print(pred.shape, type(pred))
# print(pred[0:10,:])
#
# kpca = KernelPCA(n_components = 1, kernel = 'cosine')
# features_kpca = kpca.fit_transform(features)
# plt.subplot(3,1,3)
# plt.plot(features_kpca[:,0])
# plt.show()
# print(kpca.explained_variance_ratio_)
# print(kpca.explained_variance_)
# print(kpca.components_)

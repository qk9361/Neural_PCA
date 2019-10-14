import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA
import torch
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# features, labels = make_blobs(n_samples = 10000, n_features = 4, centers =[
# [0, 0, 0, 0], [10, 10, 10, 10], [10, 20, 40, 80], [10, 30, 90, 270]], cluster_std = [0.1, 0.2, 0.2, 0.2],
# random_state = 9)
#
# print(features.shape, type(features))
#
# np.random.shuffle(features)
#
# pca = PCA(n_components = 2)
# pca.fit(features)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)
#
# labels = pca.fit_transform(features)
# print(labels.shape, type(labels))
# print(labels[0:10,:])

# f = h5py.File('sim_data', 'w')
# f = h5py.File('sim_data.h5', 'w')
# f.create_dataset('features', data = features)
# f.create_dataset('labels', data = labels)
# f.close()

f = h5py.File('sim_data.h5', 'r')
features = np.array(f['features'])
labels = np.array(f['labels'])
f.close()

in_feature = 2
out_feature = 4
hid_feature1 = 300
hid_feature2 = 600
hid_feature3 = 300

# from scipy import io
# features = io.loadmat('features.mat')
# features = features['features']
# print(features.shape, type(features))

# use eigen vector as output
# eig_vec = io.loadmat('eig_vec.mat')['eig_vecCopy'][:,0:2].T.ravel()
# print(eig_vec.shape)
# labels = np.ones((10000,1)) * eig_vec
# print(labels.shape, labels[0:10,:])

# labels = io.loadmat('score.mat')['score'][:,0:2]
# print(labels.shape, type(labels))

in_feature = 4
out_feature = 8
hid_feature1 = 200
>>>>>>> c10dafe37c4438da822ea82b93c699513d3220a4

net = nn.Sequential(nn.Linear(in_feature, hid_feature1),
                    nn.Sigmoid(),
                    # nn.BatchNorm1d(hid_feature1),
                    nn.Linear(hid_feature1, hid_feature2),
                    nn.Sigmoid(),
                    # nn.BatchNorm1d(hid_feature2),
                    nn.Linear(hid_feature2, hid_feature3),
                    nn.Sigmoid(),
                    # nn.BatchNorm1d(hid_feature3),
                    nn.Linear(hid_feature3, out_feature)
                    )

features = torch.tensor(features, dtype = torch.float)
label = torch.tensor(labels, dtype = torch.float)
train_set = Data.TensorDataset(features[0:8000,:], label[0:8000,:])
test_set = Data.TensorDataset(features[8000:-1,:], label[8000:-1,:])

batch_size = 300
num_workers = 4

train_iter = Data.DataLoader(
                            dataset = train_set,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = num_workers,
                            )
test_iter = Data.DataLoader(
                            dataset = test_set,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = num_workers,
                            )
for name, param in net.named_parameters():
    print(name, param.shape, type(param))
    init.normal_(param, mean = 0, std = 1)

loss = nn.MSELoss()

num_epochs = 50

optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay = 0.1)
print(optimizer)

for epoch in range(1, num_epochs+1):
    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    train_l_sum, n, start = 0.0, 0, time.time()
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # train_l_sum += l.cpu().item()
        # n += y.shape[0]
        batch_count += 1
    print('epoch: %d,  loss: %.4f, time: %.3f sec' %(epoch, l.cpu().item(), time.time() - start))

xx, yy = iter(test_iter).next()
xx = xx.to(device)
dif = net(xx).cpu() - yy

print(dif, '\n', torch.norm(dif) / len(yy))
print(dif, '\n', torch.norm(dif), torch.norm(yy))
print(net(xx), '\n', yy)

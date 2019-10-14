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
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ang_dif(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        dif_ang = torch.empty(len(x), 1)
        for i in range(len(x)):
            dif_ang[i] = torch.dot(x[i,:], y[i,:]) / (torch.norm(x) * torch.norm(y))
        dif_ang = dif_ang / math.pi * 180
        return torch.mean(dif_ang ** 2)

class judge_layer(nn.Module):
    def __init__(self):
        super(judge_layer, self).__init__()

    def forward(self, x):
        out = torch.empty(len(x), 4)
        for i in range(len(x)):
            if x[i,0] > 0 and x[i,1] > 0:
                out[i,:] = torch.tensor([1, 0, 0, 0])
            if x[i,0] < 0 and x[i,1] > 0:
                out[i,:] = torch.tensor([0, 1, 0, 0])
            if x[i,0] < 0 and x[i,1] < 0:
                out[i,:] = torch.tensor([0, 0, 1, 0])
            if x[i,0] > 0 and x[i,1] < 0:
                out[i,:] = torch.tensor([0, 0, 0, 1])
        return out

def adjust_lr(optimizer, epoch, init_lr):
    lr = init_lr ** (epoch // 10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
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

in_feature = 2
out_feature = 2
hid_feature1 = 300
hid_feature2 = 600
hid_feature3 = 900
hid_feature4 = 600
hid_feature5 = 300

net = nn.Sequential(nn.Linear(in_feature, hid_feature1),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature1),
                    nn.Linear(hid_feature1, hid_feature2),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature2),
                    nn.Linear(hid_feature2, hid_feature3),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature3),
                    nn.Linear(hid_feature3, hid_feature4),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature4),
                    nn.Linear(hid_feature4, hid_feature5),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature5),
                    nn.Linear(hid_feature5, out_feature),
                    judge_layer()
                    )

features = torch.tensor(features, dtype = torch.float)
label = torch.tensor(labels)
train_set = Data.TensorDataset(features[0:3000,:], label[0:3000,:])
test_set = Data.TensorDataset(features[3000:-1,:], label[3000:-1,:])

batch_size = 50
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

xx, yy = iter(test_iter).next()
print(net(xx), net(xx).dtype)
print(yy, yy.dtype)

# loss = nn.MSELoss()
# loss = ang_dif()
loss = nn.CrossEntropyLoss()

num_epochs = 50
init_lr = 0.1

optimizer = optim.Adam(net.parameters(), lr=init_lr, weight_decay = 0.1)
print(optimizer)

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
print(input, '\n', target)

for epoch in range(1, num_epochs+1):
    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    train_l_sum, n, start = 0.0, 0, time.time()
    # adjust_lr(optimizer, epoch, init_lr)
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x).to(device)
        l = loss(y_hat, y.T).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        train_acc_sum += (y_hat.argmax(dim = 1) == y).cpu().sum().item()
        n += y.shape[0]
        batch_count += 1
    print('epoch: %d,  loss: %.4f, time: %.3f sec' %(epoch, l.cpu().item(), time.time() - start))
    # if
    # if abs(l.cpu().item()) < 0.005:
    #     break

xx, yy = iter(test_iter).next()
xx = xx.to(device)
dif = loss(net(xx).cpu(), yy)
print(net(xx), '\n', yy)

# print(dif)
# # print(dif, '\n', torch.norm(dif), torch.norm(yy))
# pred_error = torch.empty(len(xx), 1)
# for i in range(len(xx)):
#     pred_error[i] = torch.dot(net(xx).cpu()[i,:], yy[i,:]) / (torch.norm(net(xx).cpu()[i,:]) * torch.norm(yy[i,:]))
# pred_error = pred_error / math.pi * 180
# print(pred_error)
# for name, param in net.named_parameters():
#     print(name, '\n', param)

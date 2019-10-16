import numpy as np
import torch
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py
import time
import math

x = np.array(range(1,1001)).reshape(1000, 1)
# print(x.shape)
x_1 = x
x_2 = 2 * x
x_3 = 3 * x
x_4 = 4 * x

y_1 = np.tile(np.array([0]), (len(x),1))
# print(y_1.shape)
y_2 = np.tile(np.array([1]), (len(x),1))
y_3 = np.tile(np.array([2]), (len(x),1))
y_4 = np.tile(np.array([3]), (len(x),1))

data1 = np.stack([x, x_1, y_1], axis = 1)
# print(data1.shape)
data2 = np.stack([x, x_2, y_2], axis = 1)
data3 = np.stack([x, x_3, y_3], axis = 1)
data4 = np.stack([x, x_4, y_4], axis = 1)

data = np.vstack([data1, data2, data3, data4]).squeeze(2)
# print(data.shape)
np.random.shuffle(data)
# print(data[0:10])

in_feature = 2
out_feature = 4
hid_feature1 = 30
hid_feature2 = 60
hid_feature3 = 90
hid_feature4 = 60
hid_feature5 = 30

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
                    )
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay = 0.1)

features = torch.tensor(data[:, 0:2], dtype = torch.float)
labels = torch.tensor(data[:, -1], dtype = torch.long)
train_set = Data.TensorDataset(features[0:3000,:], labels[0:3000])
test_set = Data.TensorDataset(features[3000:-1,:], labels[3000:-1])

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

num_epochs = 50

for epoch in range(1, num_epochs+1):
    print('training on cpu')
    batch_count = 0
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
    # adjust_lr(optimizer, epoch, init_lr)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
        n += y.shape[0]
        batch_count += 1
    print('epoch: %d,  loss: %.4f, train acc: %.4f time: %.3f sec' %(epoch, l.item()/n, train_acc_sum/n, time.time() - start))

xx, yy = iter(test_iter).next()
yy_hat = net(xx)
test_acc = (yy_hat.argmax(dim=1) == yy).sum() / len(yy)
print('test_acc: ', test_acc)
pred = yy_hat.argmax(dim=1)
print(pred[0:10], '\n', yy[0:10])

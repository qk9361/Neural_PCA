import numpy as np
import torch
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py
import time
import math

def adjust_lr(optimizer, epoch, init_lr):
    lr = init_lr ** (epoch // 10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

ang = np.array(range(180))
print(ang.shape)
ang_reg = ang / 180 * math.pi
basis = np.array(range(1, 101))

x = locals()
y = locals()
data = np.stack([basis * math.cos(ang_reg[0]), basis * math.sin(ang_reg[0]), np.zeros(len(basis))], axis = 1)
print(data)

for i in ang[1:180]:
    tmp = []
    x['x'+str(i)] = basis * math.cos(ang_reg[i])
    y['y'+str(i)] = basis * math.sin(ang_reg[i])
    tmp = np.stack([x['x'+str(i)], y['y'+str(i)], np.ones(len(basis))*i ], axis = 1)
    data = np.append(data, tmp, axis = 0)

# data = np.array(data, dtype = np.float)
print(data.shape, data.dtype)
# data = data.reshape(-1, 3)
# print(data[450:460])

np.random.shuffle(data)
# print(data[0:10])

in_feature = 2
out_feature = 180
hid_feature1 = 200
hid_feature2 = 400
hid_feature3 = 600
hid_feature4 = 400
hid_feature5 = 200

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

init_lr = 0.01
optimizer = optim.Adam(net.parameters(), lr = init_lr, weight_decay = 0.1)

features = torch.tensor(data[:, 0:2], dtype = torch.float)
labels = torch.tensor(data[:, -1], dtype = torch.long)
train_set = Data.TensorDataset(features[0:15000,:], labels[0:15000])
test_set = Data.TensorDataset(features[15000:-1,:], labels[15000:-1])

batch_size = 200
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
    init.normal_(param, mean = 0, std = 0.1)

num_epochs = 100

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

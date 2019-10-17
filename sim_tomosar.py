import numpy as np
import torch
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py
import time
import math
import cmath
from scipy import io
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_lr(optimizer, epoch, init_lr):
    lr = init_lr ** (epoch // 10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

data = io.loadmat('kpca_nn_simulation.mat')
for key in data.keys():
    print(key)

# load simulated SAR signals as observations/features
g = data['G'].T
g_r = np.real(g)
g_i = np.imag(g)
g_ravel = np.hstack([g_r, g_i])
print(g_ravel.shape)
# print(g.shape, '\n', g_r[0, :], g_i[0, :], g_r.dtype, g_i.dtype)

# load simulated steering vectors as labels
r1 = data['R'][:,:,0].T#.squeeze(2)
r2 = data['R'][:,:,1].T#.squeeze(2)
# print(r1.shape, r2.shape)

r1_r = np.real(r1)
r1_i = np.imag(r1)
r1_ravel = np.hstack([r1_r, r1_i])
r2_r = np.real(r2)
r2_i = np.imag(r2)
r2_ravel = np.hstack([r2_r, r2_i])
print(r1_ravel.shape, r2_ravel.shape)

in_features = 13
out_features = 13
hid_feature1 = 1000
hid_feature2 = 2000
hid_feature3 = 3000
hid_feature4 = 2000
hid_feature5 = 1000

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(in_features, hid_feature1),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature1),
                                    nn.Linear(hid_feature1, hid_feature2),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature2),
                                    nn.Linear(hid_feature2, hid_feature3),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature3),
                                    nn.Linear(hid_feature3, hid_feature4),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature4),
                                    nn.Linear(hid_feature4, hid_feature5),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature5),
                                    nn.Linear(hid_feature5, out_features),
                                    )
        self.dense2 = nn.Sequential(nn.Linear(in_features, hid_feature1),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature1),
                                    nn.Linear(hid_feature1, hid_feature2),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature2),
                                    nn.Linear(hid_feature2, hid_feature3),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature3),
                                    nn.Linear(hid_feature3, hid_feature4),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature4),
                                    nn.Linear(hid_feature4, hid_feature5),
                                    nn.Sigmoid(),
                                    # nn.BatchNorm1d(hid_feature5),
                                    nn.Linear(hid_feature5, out_features),
                                    )
    def forward(self, x1, x2):
        y1 = self.dense1(x1)
        y2 = self.dense2(x2)
        y = torch.cat((y1, y2), dim = 1)
        return y

net = my_net()

g_ravel = torch.tensor(g_ravel, dtype = torch.float)
r1_ravel = torch.tensor(r1_ravel, dtype = torch.float)
r2_ravel = torch.tensor(r2_ravel, dtype = torch.float)

# train_set = Data.TensorDataset(g_ravel[0:900,:], r1_ravel[0:900,:])
train_set = Data.TensorDataset(g_ravel, r1_ravel)
test_set = Data.TensorDataset(g_ravel[900:-1,:], r1_ravel[900:-1,:])

batch_size = 20
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

init_lr = 0.1
optimizer = optim.Adam(net.parameters(), lr = init_lr, weight_decay = 0.1)
loss = nn.MSELoss()
num_epochs = 100

for epoch in range(1, num_epochs+1):
    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
    adjust_lr(optimizer, epoch, init_lr)
    for x, y in train_iter:
        y = y.to(device)
        x1 = x[:,0:13].to(device)
        x2 = x[:,13:26].to(device)
        # x = F.normalize(x, dim = 0)
        # print(x)
        y_hat = net(x1,x2)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        # train_acc_sum += (y_hat.argmax(dim = 1).unsqueeze(1) == y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
    print('epoch: %d,  loss: %.4f, time: %.3f sec' %(epoch, l.cpu().item(), time.time() - start))

# xx, yy = iter(test_iter).next()
# xx, yy = iter(train_iter).next()
dataset = Data.TensorDataset(g_ravel, r1_ravel)
data_eval = Data.DataLoader(
                            dataset = dataset,
                            batch_size = 1000,
                            shuffle = True,
                            num_workers = num_workers,
                            )
xx,yy = iter(data_eval).next()
xx1, xx2 = xx[:,0:13], xx[:,13:26]
net.to('cpu')
yy_hat = net(xx1, xx2)
yy = yy.detach().numpy()
yy_hat = yy_hat.detach().numpy()

yy_r = yy[:,0:13]
yy_i = yy[:,13:26]
yy_hat_r = yy_hat[:,0:13]
yy_hat_i = yy_hat[:,13:26]

yyy = yy_r + yy_i * cmath.sqrt(-1)
yyy_hat = yy_hat_r + yy_hat_i * cmath.sqrt(-1)

angbias =  np.imag( np.log( np.sum((yyy_hat * np.conj(yyy)), axis=1) / np.sum((abs(yyy_hat) * abs(yyy)), axis=1) ) ) / math.pi * 180
print(angbias, angbias.shape)

plt.figure()
plt.plot(range(1,1001), angbias)
plt.xlabel('simulated samples')
plt.ylabel('angbias for the first steering vector r1')
plt.show()
# angbias1 = []
# for i in angbias:
#     angbias1.append(cmath.acos(i))
#
# angbias1 = np.array(angbias1)
# print(angbias1)
# angbias2 = angbias / cmath.pi * 180
# print(angbias2)

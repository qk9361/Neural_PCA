import torch
import time
import numpy as np
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_feature = 100
out_feature = 100
hid_feature1 = 1000
hid_feature2 = 600
hid_feature3 = 200

# class test_net(nn.Module):
#     def __init__(self):
#         super(test_net, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 6, 3, padding = 1, stride = 1), # in_channels, out_channels, kernel_size
#             nn.Sigmoid(),
#             nn.BatchNorm2d(6),
#             # nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 3, padding = 1, stride = 1),
#             nn.BatchNorm2d(16),
#             nn.Sigmoid(),
#             # nn.MaxPool2d(2, 2)
#             nn.Conv2d(16, 32, 3, padding = 1, stride = 1),
#             nn.BatchNorm2d(32),
#             nn.Sigmoid(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(32*10*10, 800),
#             nn.Sigmoid(),
#             nn.Linear(800, 400),
#             nn.Sigmoid(),
#             nn.Linear(400, 100)
#         )
#
#     def forward(self, x):
#         feature = self.conv(x)
#         output = self.fc(feature.view(x.shape[0], -1))
#         return output
# net = test_net()

# f = h5py.File('cov_mat.h5', 'r+')
# features = torch.tensor(f['cov'], dtype = torch.float)
# labels = torch.tensor(f['eig_mat_ravel'], dtype = torch.float)
# train_set = Data.TensorDataset(features[0:8000,:,:,:], labels[0:8000,:])
# test_set = Data.TensorDataset(features[8000:10000,:,:,:], labels[8000:10000,:])

net = nn.Sequential(nn.Linear(in_feature, hid_feature1),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature1),
                    nn.Linear(hid_feature1, hid_feature2),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature2),
                    nn.Linear(hid_feature2, hid_feature3),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(hid_feature3),
                    nn.Linear(hid_feature3, out_feature)
                    )

print(net)

f = h5py.File('cov_mat.h5', 'r+')
features = torch.tensor(f['cov_ravel'], dtype = torch.float)
labels = torch.tensor(f['eig_mat_ravel'], dtype = torch.float)
train_set = Data.TensorDataset(features[0:8000,:], labels[0:8000,:])
test_set = Data.TensorDataset(features[8000:10000,:], labels[8000:10000,:])

batch_size = 100
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

optimizer = optim.Adam(net.parameters(), lr=0.01)
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

test_l_sum, n = 0.0, 0
for x, y in test_iter:
    x = x.to(device)
    y = y.to(device)
    print(x.shape, y.shape)
    print(y[10,:], '\n', net(x)[10,:])
    print(y[10,:] - net(x)[10,:])
    print(torch.norm(y[10,:]))
    print(torch.norm(y[10,:] - net(x)[10,:]))
    break

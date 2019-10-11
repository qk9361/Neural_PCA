from sklearn.decomposition import PCA
import torch
import time
import numpy as np
from torch import nn, optim
import random
import torch.utils.data as Data
from torch.nn import init
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = np.empty([10, 10])
n = 1
for i in range(10):
    x[i, :] = np.random.normal(n, 0.05, 10)
    n += 1

y = x

x = x.reshape(-1,1)
y = y.reshape(-1,1)

data = np.stack([x,y], axis = 1)[:,:,0]
print(data, type(data), data.shape, x.shape, y.shape)

ori_data = data

pca = PCA(n_components = 1)
new_data = pca.fit_transform(data)
# print(new_data)


in_feature = 2
out_feature = 1
hid_feature1 = 200

net = nn.Sequential(
                    nn.Linear(in_feature, hid_feature1),
                    nn.Sigmoid(),
                    nn.Linear(hid_feature1, out_feature)
                    )
features = torch.tensor(ori_data, dtype = torch.float)
label = torch.tensor(new_data, dtype = torch.float)
dataset = Data.TensorDataset(features, label)

batch_size = 10
num_workers = 4

data_iter = Data.DataLoader(
                            dataset = dataset,
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
    for x, y in data_iter:
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

xx, yy = iter(data_iter).next()
xx = xx.to(device)
print(net(xx).T, '\n', yy.T)

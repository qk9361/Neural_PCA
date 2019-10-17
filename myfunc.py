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

in_features = 13
y_features = 13
out_features = 13
hid_feature1 = 1000
hid_feature2 = 2000
hid_feature3 = 4000
hid_feature4 = 2000
hid_feature5 = 1000

class auto_net(nn.Module):
    def __init__(self):
        super(auto_net, self).__init__()
        self.dense1_r = nn.Linear(in_features, hid_feature1)
        self.dense1_i = nn.Linear(in_features, hid_feature1)

        self.dense2_r = nn.Linear(hid_feature1, hid_feature2)
        self.dense2_i = nn.Linear(hid_feature1, hid_feature2)

        self.dense3_r = nn.Linear(hid_feature2, hid_feature3)
        self.dense3_i = nn.Linear(hid_feature2, hid_feature3)

        self.dense4_r = nn.Linear(hid_feature3, hid_feature4)
        self.dense4_i = nn.Linear(hid_feature3, hid_feature4)

        self.dense5_r = nn.Linear(hid_feature4, hid_feature5)
        self.dense5_i = nn.Linear(hid_feature4, hid_feature5)

        self.dense6_1_r = nn.Linear(hid_feature5, y_features)
        self.dense6_1_i = nn.Linear(hid_feature5, y_features)
        self.dense6_2_r = nn.Linear(hid_feature5, y_features)
        self.dense6_2_i = nn.Linear(hid_feature5, y_features)

        # if v is simulated, build network in this way
        # self.dense6_3_r = nn.Linear(hid_feature5, out_features3)
        # self.dense6_3_i = nn.Linear(hid_feature5, out_features3)
        # self.dense6_4_r = nn.Linear(hid_feature5, out_features4)
        # self.dense6_4_i = nn.Linear(hid_feature5, out_features4)

        # if v is not given, use this architechture
        self.dense7_1_r = nn.Linear(y_features, out_features)
        self.dense7_1_i = nn.Linear(y_features, out_features)
        self.dense7_2_r = nn.Linear(y_features, out_features)
        self.dense7_2_i = nn.Linear(y_features, out_features)
        # self.dense8_1_r = nn.Linear(y_features, y_hid_features1)
        # self.dense8_1_i = nn.Linear(y_features, y_hid_features1)
        # self.dense8_2_r = nn.Linear(y_hid_features1, out_features)
        # self.dense8_2_i = nn.Linear(y_hid_features1, out_features)

    def forward(self, x_r, x_i):
        # fully connected layer 1
        h1_rr = self.dense1_r(x_r)
        h1_ii = self.dense1_i(x_i)
        h1_ri = self.dense1_r(x_i)
        h1_ir = self.dense1_i(x_r)
        h1_r = torch.sigmoid(h1_rr - h1_ii)
        h1_i = torch.sigmoid(h1_ri + h1_ir)
        # print('a', h1_r.shape, h1_i.shape)
        # fully connected layer 2
        h2_rr = self.dense2_r(h1_r)
        h2_ii = self.dense2_i(h1_i)
        h2_ri = self.dense2_r(h1_i)
        h2_ir = self.dense2_i(h1_r)
        h2_r = torch.sigmoid(h2_rr - h2_ii)
        h2_i = torch.sigmoid(h2_ri + h2_ir)
        # print('b', h2_r.shape, h2_i.shape)
        # fully connected layer 3
        h3_rr = self.dense3_r(h2_r)
        h3_ii = self.dense3_i(h2_i)
        h3_ri = self.dense3_r(h2_i)
        h3_ir = self.dense3_i(h2_r)
        h3_r = torch.sigmoid(h3_rr - h3_ii)
        h3_i = torch.sigmoid(h3_ri + h3_ir)
        # print('c', h3_r.shape, h3_i.shape)
        # fully connected layer 4
        h4_rr = self.dense4_r(h3_r)
        h4_ii = self.dense4_i(h3_i)
        h4_ri = self.dense4_r(h3_i)
        h4_ir = self.dense4_i(h3_r)
        h4_r = torch.sigmoid(h4_rr - h4_ii)
        h4_i = torch.sigmoid(h4_ri + h4_ir)
        # print('d', h4_r.shape, h4_i.shape)
        # fully connected layer 5
        h5_rr = self.dense5_r(h4_r)
        h5_ii = self.dense5_i(h4_i)
        h5_ri = self.dense5_r(h4_i)
        h5_ir = self.dense5_i(h4_r)
        h5_r = torch.sigmoid(h5_rr - h5_ii)
        h5_i = torch.sigmoid(h5_ri + h5_ir)
        # print('e')
        # fully connected layer 6
        y1_rr = self.dense6_1_r(h5_r)
        y1_ii = self.dense6_1_i(h5_i)
        y1_ri = self.dense6_1_r(h5_i)
        y1_ir = self.dense6_1_i(h5_r)
        y1_r = torch.sigmoid(y1_rr - y1_ii)
        y1_i = torch.sigmoid(y1_ri + y1_ir)
        # print(y1_r.shape, y1_i.shape)
        y1 = torch.cat((y1_r, y1_i), dim=1)
        # print(y1.shape)

        y2_rr = self.dense6_2_r(h5_r)
        y2_ii = self.dense6_2_i(h5_i)
        y2_ri = self.dense6_2_r(h5_i)
        y2_ir = self.dense6_2_i(h5_r)
        y2_r = torch.sigmoid(y2_rr - y2_ii)
        y2_i = torch.sigmoid(y2_ri + y2_ir)
        y2 = torch.cat((y2_r, y2_i), dim=1)

        # y = torch.cat((y1, y2), dim=0)

        # for network with known v
        # v1_rr = self.dense6_3_r(h5_r)
        # v1_ii = self.dense6_3_i(h5_i)
        # v1_ri = self.dense6_3_r(h5_i)
        # v1_ir = self.dense6_3_i(h5_r)
        # v1_r = F.sigmoid(v1_rr - v1_ii)
        # v1_i = F.sigmoid(v1_ri + v1_ir)
        # v1 = torch.cat((v1_r, v1_i), dim=1)
        #
        # v2_rr = self.dense6_4_r(h5_r)
        # v2_ii = self.dense6_4_i(h5_i)
        # v2_ri = self.dense6_4_r(h5_i)
        # v2_ir = self.dense6_4_i(h5_r)
        # v2_r = F.sigmoid(v2_rr - v2_ii)
        # v2_i = F.sigmoid(v2_ri + v2_ir)
        # v2 = torch.cat((v2_r, v2_i), dim=1)
        #
        # r1v1_r = (y1_r * v1_r) - (y1_i * v1_i)
        # r1v1_i = (y1_r * v1_i) + (y1_i * v1_r)
        #
        # r2v2_r = (y2_r * v2_r) - (y2_i * v2_i)
        # r2v2_i = (y2_r * v2_i) + (y2_i * v2_r)
        #
        # out_r = r1v1_r + r2v2_r
        # out_i = r1v1_i + r2v2_i
        # out = torch.cat((out_r, out_i), dim=1)

        # for network with unknown v
        y1_h1_rr = self.dense7_1_r(y1_r)
        y1_h1_ii = self.dense7_1_i(y1_i)
        y1_h1_ri = self.dense7_1_r(y1_i)
        y1_h1_ir = self.dense7_1_i(y1_r)
        y1_h1_r = y1_h1_rr - y1_h1_ii
        y1_h1_i = y1_h1_ri + y1_h1_ir

        y2_h1_rr = self.dense7_2_r(y2_r)
        y2_h1_ii = self.dense7_2_i(y2_i)
        y2_h1_ri = self.dense7_2_r(y2_i)
        y2_h1_ir = self.dense7_2_i(y2_r)
        y2_h1_r = y2_h1_rr - y2_h1_ii
        y2_h1_i = y2_h1_ri + y2_h1_ir

        out_r = y1_h1_r + y2_h1_r
        out_i = y1_h1_i + y2_h1_i
        out = torch.cat((out_r, out_i), dim=1)

        return y1, y2, out

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
        self.dense3 = nn.Sequential(nn.Linear(in_features, hid_feature1),
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
        self.dense4 = nn.Sequential(nn.Linear(in_features, hid_feature1),
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
        y11 = self.dense1(x1)
        y12 = self.dense2(x2)
        y1 = torch.cat((y11, y12), dim = 1)
        y21 = self.dense3(x1)
        y22 = self.dense4(x2)
        y2 = torch.cat((y21, y22), dim = 1)
        y = torch.cat((y1, y2), dim = 1)
        return y

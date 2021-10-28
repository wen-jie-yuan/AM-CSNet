# -*-coding:utf-8-*-
# ! /usr/bin/env python
"""
Author And Time : ywj 2021/10/25 8:52
Desc: cal_flops_parms
"""
import time

import torch
import torch.nn as nn
from thop import profile
from torchstat import stat
from torchsummary import summary


class Net(nn.Module):
    pass


# 精确计算模型的参数量
def cal_flops_params_1(Net, input_size=(1, 256, 256)):
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    summary(model, input_size=input_size)
    end = time.time()
    print(end - start, "s")


# 精确计算模型的Flops
def cal_flops_params_2(Net, input_size=(1, 256, 256)):
    start = time.time()
    model = Net()
    stat(model, input_size)
    end = time.time()
    print(end - start, "s")


# 精确计算模型的Flops
def cal_flops_params_3(Net, input_size=(1, 256, 256)):
    start = time.time()
    img = torch.randn(input_size)
    net = Net()
    flops, params = profile(net, (img,))
    print('flops: ', flops, 'params: ', params)
    end = time.time()
    print(end - start, "s")

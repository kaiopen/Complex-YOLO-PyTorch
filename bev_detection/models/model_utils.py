# -*- encoding: utf-8 -*-
'''
@File    :   model_utils.py
@Time    :   2019/05/27 16:28:26
@Author  :   Painter
@Contact :   painter9509@126.com

'''
from torch import nn


def make_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                batch_norm=True, activation_func="relu"):
    layers = []
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    if batch_norm:
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    if activation_func == "relu":
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# -*- encoding: utf-8 -*-
'''
@File    :   complex_yolo.py
@Time    :   2019/05/27 15:22:52
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import torch
from torch import nn

from bev_detection.models.basic_model import BasicModule
from bev_detection.models.model_utils import make_conv2d


class ComplexYOLO(BasicModule):
    '''Docstring for ComplexYOLO'''
    def __init__(self):
        super(ComplexYOLO, self).__init__()
        self.__layers1 = nn.Sequential(
            # 0
            make_conv2d(in_channels=3, out_channels=24),  # 800
            nn.MaxPool2d(2),  # 400

            # 2
            make_conv2d(in_channels=24, out_channels=48),  # 400
            nn.MaxPool2d(2),  # 200

            # 4
            make_conv2d(in_channels=48, out_channels=64),  # 200
            make_conv2d(
                in_channels=64, out_channels=32, kernel_size=1, padding=0),
            make_conv2d(in_channels=32, out_channels=64),
            nn.MaxPool2d(2),  # 100

            # 8
            make_conv2d(in_channels=64, out_channels=128),
            make_conv2d(in_channels=128, out_channels=64),
            make_conv2d(in_channels=64, out_channels=128),
            nn.MaxPool2d(2),  # 50

            # 12
            make_conv2d(in_channels=128, out_channels=256))

        self.__layers2 = nn.Sequential(
            # 13
            make_conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=0),
            make_conv2d(in_channels=256, out_channels=512),
            nn.MaxPool2d(2),  # 25

            # 16
            make_conv2d(in_channels=512, out_channels=512),
            make_conv2d(
                in_channels=512, out_channels=512, kernel_size=1, padding=0),
            make_conv2d(in_channels=512, out_channels=1024),
            make_conv2d(in_channels=1024, out_channels=1024),
            make_conv2d(in_channels=1024, out_channels=1024))

        self.__layer22 = make_conv2d(in_channels=2048, out_channels=1024)
        self.__layer23 = nn.Conv2d(
            in_channels=1024, out_channels=75, kernel_size=1, padding=0)
        self.__dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = self.__layers1(x)
        reorg_x = self.reorg(x)
        x = self.__layers2(x)
        x = torch.cat((reorg_x, x), 1)
        x = self.__layer22(x)
        x = self.__layer23(x)
        x = self.__dropout(x)
        return x

    @staticmethod
    def reorg(x):
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)

        w_stride = h_stride = 2

        x = x.reshape(
            B, C,
            int(H / h_stride), h_stride,
            int(W / w_stride), w_stride).transpose(3, 4)
        x = x.reshape(
            B, C,
            int(H / h_stride * W / w_stride),
            h_stride * w_stride).transpose(2, 3)
        x = x.reshape(
            B, C,
            h_stride * w_stride,
            int(H / h_stride),
            int(W / w_stride)).transpose(1, 2)
        x = x.reshape(
            B,
            h_stride * w_stride * C,
            int(H / h_stride),
            int(W / w_stride))
        return x

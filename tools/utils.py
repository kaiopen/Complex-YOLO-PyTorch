# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/07/07 10:10:47
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import numpy as np
import matplotlib.pyplot as plt
import torch

from bev_detection.config import cfg


def get_prediction_fmaps(output):
    '''Get prediction prejected to feature maps.

    '''
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)
    anchors = cfg.get_anchors().type_as(output)
    n_anchors = anchors.size(0)
    n_classes = cfg.get_n_classes()

    output = output.reshape(
        B, n_anchors, (7 + n_classes), H, W).permute(0, 1, 3, 4, 2)
    t_x = torch.sigmoid(output[..., 0])
    t_y = torch.sigmoid(output[..., 1])
    t_l = output[..., 2]
    t_w = output[..., 3]

    c_x = torch.arange(H).type_as(output).repeat(W, 1).t()
    c_x = c_x.reshape((1, 1, H, W))
    c_y = torch.arange(W).type_as(output).repeat(H, 1)
    c_y = c_y.reshape((1, 1, H, W))
    p_l = anchors[:, 0].reshape((1, n_anchors, 1, 1))
    p_w = anchors[:, 1].reshape((1, n_anchors, 1, 1))

    prediction_fmaps = output.new_empty(output[..., 0: 8].shape)
    prediction_fmaps[..., 0] = t_x + c_x  # x
    prediction_fmaps[..., 1] = t_y + c_y  # y
    prediction_fmaps[..., 2] = torch.exp(t_l) * p_l  # l
    prediction_fmaps[..., 3] = torch.exp(t_w) * p_w  # w
    prediction_fmaps[..., 4] = output[..., 4]  # im
    prediction_fmaps[..., 5] = output[..., 5]  # re
    prediction_fmaps[..., 6] = torch.sigmoid(output[..., 6])  # conf
    prediction_fmaps[..., 7] = torch.argmax(torch.sigmoid(output[..., 7:]), -1)
    return prediction_fmaps


def get_prediction_bev(output, bev_size):
    fmap_H = output.size(2)
    fmap_W = output.size(3)
    bev_H, bev_W = bev_size

    prediction_fmap = get_prediction_fmaps(output)
    prediction_fmap[..., 0] = (prediction_fmap[..., 0] / fmap_H) * bev_H
    prediction_fmap[..., 1] = (prediction_fmap[..., 1] / fmap_W) * bev_W
    prediction_fmap[..., 2] = (prediction_fmap[..., 2] / fmap_H) * bev_H
    prediction_fmap[..., 3] = (prediction_fmap[..., 3] / fmap_W) * bev_W
    return prediction_fmap


def get_prediction_real(output):
    fmap_H = output.size(2)
    fmap_W = output.size(3)
    real_W, real_H = cfg.get_valid_boundary_size()

    prediction_fmap = get_prediction_fmaps(output)
    prediction_fmap[..., 0] = (prediction_fmap[..., 0] / fmap_H) * real_H
    prediction_fmap[..., 1] = (prediction_fmap[..., 1] / fmap_W) * real_W - \
        real_W / 2
    prediction_fmap[..., 2] = (prediction_fmap[..., 2] / fmap_H) * real_H
    prediction_fmap[..., 3] = (prediction_fmap[..., 3] / fmap_W) * real_W
    return prediction_fmap


def draw_box(box, color='r'):
    l2 = box[2] / 2
    w2 = box[3] / 2
    origin_box = np.array([[-l2, w2], [l2, w2], [l2, -w2], [-l2, -w2]])
    sina = box[4]
    cosa = box[5]
    rotation_mat = np.array([
        [cosa, sina],
        [-sina, cosa]])
    rotated_box = np.dot(rotation_mat, origin_box.T).T
    rotated_box[:, 0] = rotated_box[:, 0] + box[0]
    rotated_box[:, 1] = rotated_box[:, 1] + box[1]

    xs = rotated_box[:, 0]
    ys = rotated_box[:, 1]
    plt.plot(ys, xs, linewidth=1, color=color)
    plt.plot(ys[[0, -1]], xs[[0, -1]], linewidth=1, color=color)
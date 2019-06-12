# -*- encoding: utf-8 -*-
'''
@File    :   show_bev.py
@Time    :   2019/06/08 18:08:21
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import numpy as np
import matplotlib.pyplot as plt

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.config import cfg


def draw_box(box, H, W, color='r'):
    l2 = box[2] * H / 2
    w2 = box[3] * W / 2
    origin_box = np.array([[-l2, w2], [l2, w2], [l2, -w2], [-l2, -w2]])
    sina = box[4]
    cosa = box[5]
    rotation_mat = np.array([
        [cosa, sina],
        [-sina, cosa]])
    rotated_box = np.dot(rotation_mat, origin_box.T).T
    rotated_box[:, 0] = rotated_box[:, 0] + box[0] * H
    rotated_box[:, 1] = rotated_box[:, 1] + box[1] * W

    xs = rotated_box[:, 0]
    ys = rotated_box[:, 1]
    plt.plot(ys, xs, linewidth=1, color=color)
    plt.plot(ys[[0, -1]], xs[[0, -1]], linewidth=1, color=color)


if __name__ == "__main__":
    dataset = BEVKitti("train")
    ids = dataset.get_ids()
    width, height = cfg.get_valid_boundary_size()
    voxel_acc = cfg.get_voxel_acc()
    W = width / voxel_acc["rphi_acc"]
    H = height / voxel_acc["rphi_acc"]
    for idx in ids:
        idx = 5119
        bev = dataset.get_bev(idx)
        bev = bev.transpose((1, 2, 0))
        plt.imshow(bev)

        target = dataset.get_target(idx)
        target = target[target[:, 0] != 0]
        for obj in target:
            draw_box(obj[1: 7], H, W)
            # l2 = obj[3] * H / 2
            # w2 = obj[4] * W / 2
            # box = np.array([[-l2, w2], [l2, w2], [l2, -w2], [-l2, -w2]])
            # sina = obj[5]
            # cosa = obj[6]
            # rotation_mat = np.array([
            #     [cosa, -sina],
            #     [sina, cosa]])
            # rotated_box = np.dot(rotation_mat, box.T).T
            # rotated_box[:, 0] = rotated_box[:, 0] + obj[1] * H
            # rotated_box[:, 1] = rotated_box[:, 1] + obj[2] * W

            # xs = rotated_box[:, 0]
            # ys = rotated_box[:, 1]
            # plt.plot(ys, xs, linewidth=1, color='r')
            # plt.plot(ys[[0, -1]], xs[[0, -1]], linewidth=1, color='r')

        plt.show()
        break

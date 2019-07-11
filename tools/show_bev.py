# -*- encoding: utf-8 -*-
'''
@File    :   show_bev.py
@Time    :   2019/06/08 18:08:21
@Author  :   Painter
@Contact :   painter9509@126.com

Show BEV maps according to index/name of data with the ground truth.
'''

import random

import matplotlib.pyplot as plt

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.config import cfg
from tools.utils import draw_box


if __name__ == "__main__":
    dataset = BEVKitti("train")
    ids = dataset.get_ids()
    width, height = cfg.get_valid_boundary_size()
    voxel_acc = cfg.get_voxel_acc()
    W = width / voxel_acc["rphi_acc"]
    H = height / voxel_acc["rphi_acc"]

    idx = 9000  # NOTE: Invalid index/name of data.
    if idx not in ids:
        ids = [str(idx) for idx in ids]
        raise Exception(
            "Invalid index/name of data. `{}` etc. are available.".format(
                "`, `".join(random.sample(ids, 5))))

    bev = dataset.get_bev(idx)
    bev_H, bev_W = bev.hspae[1, 2]

    bev = bev.transpose((1, 2, 0))
    plt.imshow(bev)

    target = dataset.get_target(idx)
    target = target[target[:, 0] != 0]

    target[:, 1] = target[:, 1] * bev_H
    target[:, 2] = target[:, 2] * bev_W
    target[:, 3] = target[:, 3] * bev_H
    target[:, 4] = target[:, 4] * bev_W
    for obj in target:
        draw_box(obj[1: 7])

    plt.show()

# -*- encoding: utf-8 -*-
'''
@File    :   bev_kitti.py
@Time    :   2019/05/24 13:33:16
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os
import math

import numpy as np

import torch

from bev_detection.data.datasets.kitti.basic_kitti import BasicKitti
from bev_detection.config import cfg


class BEVKitti(BasicKitti):
    '''Docstring for BEVKitti'''
    __bev_path = os.path.join(
        cfg.get_datasets_cache_root(), "Kitti/bev/{:0>6}.npy")
    __bev_objs_path = os.path.join(
        cfg.get_datasets_cache_root(), "Kitti/bev_objs/{:0>6}.bin")

    def __init__(self, split):
        super(BEVKitti, self).__init__(split)

    def __getitem__(self, i):
        idx = self._ids[i]
        bev = self.get_bev(idx)
        bev = torch.tensor(bev, dtype=torch.float32)
        target = self.get_target(idx)
        target = torch.tensor(target, dtype=torch.float32)
        return bev, target

    @classmethod
    def get_bev(cls, idx):
        '''Docstring for get_bev'''
        bev_path = cls.__bev_path.format(idx)
        bev = np.load(bev_path)
        bev = bev.astype(np.float32)
        return bev

    @classmethod
    def save_bev(cls, bev, idx):
        '''Docstring for save_bev'''
        bev_dir = os.path.dirname(cls.__bev_path)
        if not os.path.exists(bev_dir):
            os.makedirs(bev_dir)
        bev_path = cls.__bev_path.format(idx)
        np.save(bev_path, bev)

    @classmethod
    def get_target(cls, idx):
        '''Docstring for get_target'''
        valid_pts_boundary = cfg.get_valid_pts_lidar_boundary()
        max_x = valid_pts_boundary["max_x"]
        min_x = valid_pts_boundary["min_x"]
        max_y = valid_pts_boundary["max_y"]
        min_y = valid_pts_boundary["min_y"]
        width, height = cfg.get_valid_boundary_size()

        # The maximum number of object in one data is `24`.
        # cls_id, x, y, w, h, im, re
        target = np.zeros((24, 7), dtype=np.float32)
        objs = cls.get_objs(idx)
        calib = cls.get_calib(idx)
        i = 0
        for obj in objs:
            cls_id = obj.get_cls_id()
            if cls_id in (0, 8, 9) or obj.get_level() not in (1, 2, 3):
                continue

            obj_pos = obj.get_obj_pos()
            obj_pos = calib.rect2lidar(obj_pos)
            center_x = obj_pos[0, 0]
            center_y = obj_pos[0, 1]

            if center_x >= min_x and center_x < max_x and \
                    center_y >= min_y and center_y < max_y:
                target[i, 0] = cls_id

                target[i, 1] = center_x / height
                target[i, 2] = (center_y + width / 2) / width

                obj_size = obj.get_obj_size()
                target[i, 3] = obj_size[2] / height  # l
                target[i, 4] = obj_size[1] / width  # w

                angle = cls.__get_angle(obj.get_rotation_y())
                im = math.sin(angle)
                re = math.cos(angle)
                target[i, 5] = im
                target[i, 6] = re
                i += 1
        return target

    @staticmethod
    def __get_angle(rotation_y):
        angle = rotation_y - math.pi / 2
        if angle < -math.pi:
            angle = 2 * math.pi + angle
        return angle

# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2019/05/20 15:54:37
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os

import torch


BASICPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        # Parameters for preprocessing and dataset
        self.__split = "train"
        self.__datasets_root = os.path.join(BASICPATH, "datasets")
        self.__datasets_cache_root = os.path.join(BASICPATH, "datasets_cache")
        self.__dataset = "Kitti"
        self.__valid_pts_lidar_boundary = {
            # The valid boundary of point cloud in lidar coordinate
            "min_x": 0,
            "max_x": 80,
            "min_y": -40,
            "max_y": 40,
            "min_z": -2,
            "max_z": 1.25
        }
        self.__valid_pts_rect_boundary = {
            # The valid boundary of point cloud in rectified camera 0
            # coordinate
            "min_l": 0,
            "max_l": 80,
            "min_w": -40,
            "max_w": 40,
            "min_h": -1.25,
            "max_h": 2
        }
        self.__voxel_acc = {
            "rphi_acc": 0.1,  # in meter
            "theta_acc": 0.1  # in degree
        }
        self.__valid_boundary_height = \
            self.__valid_pts_lidar_boundary["max_x"] - \
            self.__valid_pts_lidar_boundary["min_x"]
        self.__valid_boundary_width = \
            self.__valid_pts_lidar_boundary["max_y"] - \
            self.__valid_pts_lidar_boundary["min_y"]

        # Parameters for training
        self.__batch_size = 12
        self.__num_workers = 8
        self.__pin_memory = False
        self.__n_epochs = 1000
        self.__checkpoints_root = os.path.join(BASICPATH, "checkpoints")
        self.__initial_lr = 1e-5
        self.__momentum = 0.9
        self.__weight_decay = 0.0005
        self.__conf_threshold = 0.5

        # Parameters for validating
        self.__val_batch_size = 1
        self.__val_conf_threshold = 0.8

        # Parameters for others
        self.__device = None
        self.__anchors = torch.tensor([
            (1.19, 1.08),
            (4.41, 3.42),
            (11.38, 6.63),
            (5.11, 9.42),
            (16.62, 10.52)], dtype=torch.float32)
        self.__n_classes = 8
        self.__ignore_threshold = 0.6

    def get_datasets_root(self):
        return self.__datasets_root

    def get_datasets_cache_root(self):
        return self.__datasets_cache_root

    def get_valid_pts_lidar_boundary(self):
        return self.__valid_pts_lidar_boundary

    def get_valid_pts_rect_boundary(self):
        return self.__valid_pts_rect_boundary

    def get_voxel_acc(self):
        return self.__voxel_acc

    def get_valid_boundary_size(self):
        return self.__valid_boundary_width, self.__valid_boundary_height

    def get_batch_size(self):
        if self.__split == "train":
            return self.__batch_size
        else:
            return self.__val_batch_size

    def get_num_workers(self):
        return self.__num_workers

    def get_pin_memory(self):
        return self.__pin_memory

    def get_n_epochs(self):
        return self.__n_epochs

    def get_checkpoints_root(self):
        return self.__checkpoints_root

    def get_initial_lr(self):
        return self.__initial_lr

    def get_momentum(self):
        return self.__momentum

    def get_weight_decay(self):
        return self.__weight_decay

    def get_device(self):
        return self.__device

    def get_anchors(self):
        return self.__anchors

    def get_n_classes(self):
        return self.__n_classes

    def get_ignore_threshold(self):
        return self.__ignore_threshold

    def get_conf_threshold(self):
        if self.__split == "train":
            return self.__conf_threshold
        else:
            return self.__val_conf_threshold

    def update(self, *new_cfg):
        state_dict = vars(self)
        for kwargs in new_cfg:
            for k, v in kwargs.items():
                k = "_{}__{}".format(self.__class__.__name__, k)
                if k not in state_dict:
                    continue
                setattr(self, k, v)

        print("====== user config ======")
        for k, v in vars(self).items():
            print("{}: {}".format(k, v))
        print("======     end     ======\n")


cfg = Config()

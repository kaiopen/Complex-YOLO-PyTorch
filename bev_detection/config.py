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
        self.__output_root = os.path.join(BASICPATH, "outputs")
        self.__active_cls_str = "Car"
        self.__device = None
        self.__split = "train"

        # Parameters for preprocessing and dataset
        self.__dataset = "Kitti"
        self.__datasets_cache_root = os.path.join(
            self.__output_root, "datasets_cache")
        self.__datasets_root = os.path.join(BASICPATH, "datasets")
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
        self.__valid_boundary_height = \
            self.__valid_pts_lidar_boundary["max_x"] - \
            self.__valid_pts_lidar_boundary["min_x"]
        self.__valid_boundary_width = \
            self.__valid_pts_lidar_boundary["max_y"] - \
            self.__valid_pts_lidar_boundary["min_y"]
        self.__voxel_acc = {
            "rphi_acc": 0.1,  # in meter
            "theta_acc": 0.1  # in degree
        }

        # Parameters for model
        self.__anchors = torch.tensor([
            (1.19, 1.08),
            (4.41, 3.42),
            (11.38, 6.63),
            (5.11, 9.42),
            (16.62, 10.52)], dtype=torch.float32)
        self.__n_classes = 8

        # Parameters for training
        self.__batch_size = 12
        self.__num_workers = 8
        self.__pin_memory = False
        self.__n_epochs = 1000
        self.__initial_lr = 1e-5
        self.__momentum = 0.9
        self.__weight_decay = 0.0005
        self.__conf_threshold = 0.5

        # Parameters for evaluating
        self.__eval_batch_size = 1
        self.__eval_conf_threshold = 0.1

        # Parameters for others
        self.__ignore_threshold = 0.6

    # Sorted by name of methods.
    def get_active_cls_str(self):
        return self.__active_cls_str

    def get_anchors(self):
        return self.__anchors

    def get_batch_size(self):
        if self.__split == "active_train":
            return self.__batch_size
        else:
            return self.__val_batch_size

    def get_checkpoints_root(self):
        return os.path.join(
            self.__output_root, self.__active_cls_str.lower(), "checkpoints")

    def get_conf_threshold(self):
        if self.__split == "active_train":
            return self.__conf_threshold
        else:
            return self.__val_conf_threshold

    def get_datasets_cache_root(self):
        return self.__datasets_cache_root

    def get_datasets_root(self):
        return self.__datasets_root

    def get_device(self):
        return self.__device

    def get_eval_root(self):
        return os.path.join(
            self.__output_root, self.__active_cls_str.lower(), "eval")

    def get_ignore_threshold(self):
        return self.__ignore_threshold

    def get_initial_lr(self):
        return self.__initial_lr

    def get_kitti_native_eval_root(self):
        eval_root = self.get_eval_root()
        return os.path.join(eval_root, "kitti_native_eval")

    def get_momentum(self):
        return self.__momentum

    def get_num_workers(self):
        return self.__num_workers

    def get_n_classes(self):
        return self.__n_classes

    def get_n_epochs(self):
        return self.__n_epochs

    def get_pin_memory(self):
        return self.__pin_memory

    def get_split(self):
        return self.__split

    def get_valid_pts_lidar_boundary(self):
        return self.__valid_pts_lidar_boundary

    def get_valid_pts_rect_boundary(self):
        return self.__valid_pts_rect_boundary

    def get_valid_boundary_size(self):
        return self.__valid_boundary_width, self.__valid_boundary_height

    def get_voxel_acc(self):
        return self.__voxel_acc

    def get_weight_decay(self):
        return self.__weight_decay

    def update(self, *new_cfg):
        state_dict = vars(self)
        for kwargs in new_cfg:
            for k, v in kwargs.items():
                k = "_{}__{}".format(self.__class__.__name__, k)
                if k not in state_dict:
                    continue
                setattr(self, k, v)


cfg = Config()

# -*- encoding: utf-8 -*-
'''
@File    :   basic_kitti.py
@Time    :   2019/05/14 14:32:36
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os
from PIL import Image

import numpy as np
from torch.utils.data import Dataset

from bev_detection.utils.utils import cls_str2id
from bev_detection.config import cfg


class BasicKitti(Dataset):
    '''Basic Kitti dataset

    '''
    __dataset_root = os.path.join(cfg.get_datasets_root(), "Kitti")
    __pts_path = os.path.join(__dataset_root, "training/velodyne/{:0>6}.bin")
    __img_path = os.path.join(__dataset_root, "training/image_2/{:0>6}.png")
    __calib_path = os.path.join(__dataset_root, "training/calib/{:0>6}.txt")
    __label_path = os.path.join(__dataset_root, "training/label_2/{:0>6}.txt")
    __plane_path = os.path.join(__dataset_root, "training/planes/{:0>6}.txt")

    def __init__(self):
        super(BasicKitti, self).__init__()
        split = cfg.get_split()
        if split in ("train", "val", "trainval"):
            split_path = os.path.join(self.__dataset_root, split + ".txt")
        else:
            split_path = os.path.join(
                cfg.get_datasets_cache_root(), "Kitti", split + ".txt")
        with open(split_path, 'r') as f:
            self._ids = [int(idx.strip()) for idx in f.readlines()]

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self._ids)

    def get_ids(self):
        return self._ids

    @classmethod
    def get_pts_lidar(cls, idx):
        '''Get lidar data in lidar coordinate in one scan accordingly.

        Args:
            idx (int): The index of data.

        Returns:
            numpy.ndarray: Lidar data in shape `(N, 4)` where are x, y, z and
                intensity in lidar coordinate.

        '''
        pts_path = cls.__pts_path.format(idx)
        return np.fromfile(pts_path, dtype=np.float32).reshape((-1, 4))

    @classmethod
    def get_pts_rect(cls, idx):
        '''Get lidar data in camera 0 coordinate in one scan accordingly.

        Args:
            idx (int): The index of data.

        Returns:
            numpy.ndarray: Lidar data with shape `(N, 3)`
                where are x, y, z in camera 0 coordinate.
            numpy.ndarray: The intensity of lidar points with shape `(N, 1)`.

        '''
        pts_lidar = cls.get_pts_lidar(idx)
        pts_lidar = pts_lidar.reshape((-1, 4))
        calib = cls.get_calib(idx)
        pts_rect, pts_intensity = calib.lidar2rect(pts_lidar)
        return pts_rect, pts_intensity

    @classmethod
    def get_valid_pts_lidar(cls, idx, img_size=None):
        pts_lidar = cls.get_pts_lidar(idx)
        valid_pts_boundary = cfg.get_valid_pts_lidar_boundary()
        mask = np.where(
            (pts_lidar[:, 0] >= valid_pts_boundary["min_x"]) &
            (pts_lidar[:, 0] < valid_pts_boundary["max_x"]) &
            (pts_lidar[:, 1] >= valid_pts_boundary["min_y"]) &
            (pts_lidar[:, 1] < valid_pts_boundary["max_y"]) &
            (pts_lidar[:, 2] >= valid_pts_boundary["min_z"]) &
            (pts_lidar[:, 2] < valid_pts_boundary["max_z"]))
        pts_lidar = pts_lidar[mask]

        if img_size is not None:
            # Remove points out of image.
            calib = cls.get_calib(idx)
            pts_img, _, _ = calib.lidar2img(pts_lidar)
            mask = np.where(
                (pts_img[:, 0] >= 0) &
                (pts_img[:, 0] < img_size[0]) &
                (pts_img[:, 1] >= 0) &
                (pts_img[:, 1] < img_size[1]))
            pts_lidar = pts_lidar[mask]

        return pts_lidar

    @classmethod
    def get_valid_pts_rect(cls, idx, img_size=None):
        pts_rect, pts_intensity = cls.get_pts_rect(idx)
        valid_pts_boundary = cfg.get_valid_pts_rect_boundary()
        mask = np.where(
            (pts_rect[:, 2] >= valid_pts_boundary["min_l"]) &
            (pts_rect[:, 2] < valid_pts_boundary["max_l"]) &
            (pts_rect[:, 0] >= valid_pts_boundary["min_w"]) &
            (pts_rect[:, 0] < valid_pts_boundary["max_w"]) &
            (pts_rect[:, 1] >= valid_pts_boundary["min_h"]) &
            (pts_rect[:, 1] < valid_pts_boundary["max_h"]))
        pts_rect = pts_rect[mask]
        pts_intensity = pts_intensity[mask]

        if img_size is not None:
            # Remove points out of image.
            calib = cls.get_calib(idx)
            pts_img, _ = calib.rect2img(pts_rect)
            mask = np.where(
                (pts_img[:, 0] >= 0) &
                (pts_img[:, 0] < img_size[0]) &
                (pts_img[:, 1] >= 0) &
                (pts_img[:, 1] < img_size[1]))
            pts_rect = pts_rect[mask]
            pts_intensity = pts_intensity[mask]

        return pts_rect, pts_intensity

    @classmethod
    def get_img(cls, idx, return_size=False):
        '''Get an image accordingly.

        Args:
            idx (int): The index of data.
            return_info (bool, optional): If `True`, the function will return
                the information about image including height and width.
                Default is `False`.

        Returns:
            PIL.Image: An image.
            tuple of int: The width and height of image.

        '''
        img_path = cls.__img_path.format(idx)
        img = Image.open(img_path)
        if return_size:
            w, h = img.size
            return img, (w, h)
        return img

    @classmethod
    def get_calib(cls, idx):
        '''Get the calibration matrix accordingly.

        Args:
            idx (int): The index of data.

        Returns:
            object: An implementation of class `Calib`.

        '''
        calib_path = cls.__calib_path.format(idx)
        return Calib(calib_path)

    @classmethod
    def get_objs(cls, idx):
        '''Get ground truth accordingly.

        Args:
            idx (int): The index of data.

        Returns:
            list of object: A list of objects, the implementations of class
                `Obj`.

        '''
        label_path = cls.__label_path.format(idx)
        with open(label_path, 'r') as f:
            labels = f.readlines()
        objs = [Obj(label=label) for label in labels]
        return objs

    @classmethod
    def get_plane(cls, idx):
        plane_path = cls.__plane_path.format(idx)
        plane = np.genfromtxt(plane_path, skip_header=3, dtype=np.float32)

        # `plane` is in the rectified camera coordinate.
        # Ensure normal is always facing up.
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0: 3])
        return plane / norm


class Calib(object):
    def __init__(self, calib_path):
        super(Calib, self).__init__()
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        self.__P2 = np.array(
            lines[2].strip().split()[1:], dtype=np.float32).reshape((3, 4))
        self.__P3 = np.array(
            lines[3].strip().split()[1:], dtype=np.float32).reshape((3, 4))

        R0 = np.array(
            lines[4].strip().split()[1:], dtype=np.float32).reshape((3, 3))

        L2C = np.array(
            lines[5].strip().split()[1:], dtype=np.float32).reshape((3, 4))

        self.__L2R = np.dot(R0, L2C).T  # `(4, 3)`

    def lidar2rect(self, pts_lidar):
        '''Project points in lidar coordinate to rectified camera 0 coordinate.

        In Kitti's official instruction, it is recommended to expand `R0` and
        `L2C` to 4*4 matrix and calculate by formula as follows which preserves
        intensity values of points.
        `y = R0 * Tr_velo_to_cam * x`
        Here, we calculate as follows due to lower error.

        '''
        pts_intensity = pts_lidar[:, -1].copy()
        pts_lidar[:, -1] = 1  # important for lower error
        pts_rect = np.dot(pts_lidar, self.__L2R)  # `(N, 3)`
        return pts_rect, pts_intensity

    def rect2lidar(self, pts_rect):
        '''Project points in rectified camera 0 coordinate to lidar coordinate.

        Args:
            pts_rect (numpy.ndarray): The points in rectified camera 0
                coordinate.

        Returns:
            numpy.ndarray: The points in lidar coordinate with shape `(N, 3)`.
                Note that intensity values are not available here.

        '''
        if pts_rect.ndim == 1:
            pts_rect = pts_rect.reshape((1, 3))
        d = self.__L2R[-1].reshape((1, 3))
        A = self.__L2R[: -1].T

        b = pts_rect - d
        b = b.T
        x = np.linalg.solve(A, b)
        pts_lidar = x.T
        return pts_lidar

    def rect2img(self, pts_rect):
        # Expand `pts_rect` with one column that is made up with `1`.
        # (N, 4)
        pts_rect = np.hstack(
            (pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)))
        pts_img = np.dot(pts_rect, self.__P2.T)  # (N, 3)
        # Depth in rectified camera 0 coordinate
        pts_rect_depth = pts_img[:, 2] - self.__P2.T[3, 2]
        pts_img = (pts_img[:, :2].T / pts_rect[:, 2]).T  # (N, 2)
        return pts_img, pts_rect_depth

    def lidar2img(self, pts_lidar):
        pts_rect, pts_intensity = self.lidar2rect(pts_lidar)
        pts_img, pts_rect_depth = self.rect2img(pts_rect)
        return pts_img, pts_rect_depth, pts_intensity


class Obj(object):
    def __init__(self, label):
        super(Obj, self).__init__()
        info = label.strip().split()
        self.__cls_str = info[0]
        self.__cls_id = cls_str2id(self.__cls_str)
        self.__alpha = float(info[3])
        self.__bbox_2d = np.array(  # in camera 2 coordinate
            [float(info[i]) for i in range(4, 8)], dtype=np.float32)
        # in rectified camera 0 coordinate
        self.__obj_size = [float(info[i]) for i in range(8, 11)]
        self.__obj_pos = np.array(
            [float(info[i]) for i in range(11, 14)], dtype=np.float32)
        self.__rotation_y = float(info[14])
        self.__score = float(info[15]) if len(info) == 16 else -1.
        self.__level = self.__calculate_level(info)

    def get_cls_str(self):
        return self.__cls_str

    def get_cls_id(self):
        return self.__cls_id

    def get_alpha(self):
        return self.__alpha

    def get_bbox_2d(self):
        return self.__bbox_2d

    def get_obj_size(self):
        return self.__obj_size

    def get_obj_pos(self):
        return self.__obj_pos

    def get_rotation_y(self):
        return self.__rotation_y

    def get_score(self):
        return self.__score

    def get_level(self):
        return self.__level

    def __calculate_level(self, info):
        '''Evaluate the difficulty of detecting the object.

        Difficulties are defined as follows:
        1. Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully
        visible, Max. truncation: 15%.
        2. Moderate: Min. bounding box height: 25 Px, Max. occlusion level:
        Partly occluded, Max. truncation: 30%.
        3. Hard: Min. bounding box height: 25 Px, Max. occlusion level: Largely
        occluded, Max. truncation: 50%.
        Visit
        http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
        for more details.

        '''
        truncation = float(info[1])
        occlusion = float(info[2])
        # truncation 0: non-truncated 1: truncated
        # occlusion 0: fully visible 1: partly occluded
        # 2: largely occluded 3: unknown

        height = self.__bbox_2d[3] - self.__bbox_2d[1]
        if height >= 40 and occlusion <= 0 and truncation <= .15:
            return 1  # Easy
        elif height >= 25 and occlusion <= 1 and truncation <= .30:
            return 2  # Moderate
        elif height >= 25 and \
            occlusion <= 2 and \
                truncation <= .50:
            return 3  # Hard
        else:
            return 0  # Unknown

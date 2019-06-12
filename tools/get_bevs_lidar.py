# -*- encoding: utf-8 -*-
'''
@File    :   get_bevs_lidar.py
@Time    :   2019/06/01 16:32:12
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import numpy as np

import init_path
from bev_detection.data.datasets.kitti.basic_kitti import BasicKitti
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.config import cfg


def normalize(data):
    '''Normalize data to 0 ~ 1.

    '''
    max_data = np.max(data)
    min_data = np.min(data)
    return ((data - min_data) / (max_data - min_data))


def generate_BEV(pts_lidar):
    '''Generate bird's eye view feature maps.

    Args:
        pts_lidar (numpy.ndarray): Point cloud with shape `(N, 4)`.

    Returns:
        numpy.ndarray: BEV maps with shape `(C, H, W)` including one height
            map, one density map and one intensity map.

    NOTE: The BEV maps' coordinate is as follows.

                  ^ x
                  |
           lidar/ |
            BEV   |
        y <--------

    '''
    pts_voxel = np.copy(pts_lidar)

    width, height = cfg.get_valid_boundary_size()
    voxel_acc = cfg.get_voxel_acc()

    # Voxelising point cloud accroding `voxel_acc`.
    pts_voxel[:, 0] = np.floor(pts_voxel[:, 0] / voxel_acc["rphi_acc"])
    pts_voxel[:, 1] = np.floor(
        (pts_voxel[:, 1] + width / 2) / voxel_acc["rphi_acc"])

    width = int(width / voxel_acc["rphi_acc"])
    height = int(height / voxel_acc["rphi_acc"])

    # TODO: `pts_lidar`-`pts_voxel` mapping table

    # Sort points by height firstly in descending order.
    mask = np.lexsort((-pts_voxel[:, 2], pts_voxel[:, 1], pts_voxel[:, 0]))
    pts_voxel = pts_voxel[mask]

    _, ids, counts = np.unique(
        pts_voxel[:, :2], axis=0, return_index=True, return_counts=True)
    pts_voxel = pts_voxel[ids]

    grids_height = normalize(pts_voxel[:, 2])
    grids_density = np.minimum(1., np.log(counts + 1) / np.log(64))
    grids_intensity = normalize(pts_voxel[:, 3])

    # Generate maps with shape `(C, H, W)`.
    # 800*800
    height_map = np.zeros((height, width))
    density_map = np.zeros((height, width))
    intensity_map = np.zeros((height, width))

    xs = pts_voxel[:, 0].astype(np.int_)
    ys = pts_voxel[:, 1].astype(np.int_)
    height_map[xs, ys] = grids_height  # h*w
    density_map[xs, ys] = grids_density
    intensity_map[xs, ys] = grids_intensity

    bev = np.stack((height_map, density_map, intensity_map))

    return bev


# import matplotlib.pyplot as plt


# def show_bev(bev):
#     fig, ax = plt.subplots(ncols=3, nrows=1)
#     height_map = bev[0]
#     density_map = bev[1]
#     intensity_map = bev[2]

#     ax[0].imshow(height_map, cmap=plt.cm.gray)
#     ax[1].imshow(density_map, cmap=plt.cm.gray)
#     ax[2].imshow(intensity_map, cmap=plt.cm.gray)
#     plt.show()


if __name__ == "__main__":
    datasets_cache_root = cfg.get_datasets_cache_root()

    dataset = BasicKitti(split="trainval")

    ids = dataset.get_ids()

    for idx in ids:
        # idx = ids[0]
        print("Generating BEV maps {:0>6} ...".format(idx))
        pts_lidar = dataset.get_valid_pts_lidar(idx)
        bev = generate_BEV(pts_lidar)
        # show_bev(bev)
        # break

        BEVKitti.save_bev(bev, idx)
    print("Successfully save BEV to " + datasets_cache_root + "/Kitti/bev/")

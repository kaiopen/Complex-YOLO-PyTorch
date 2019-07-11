# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2019/07/08 09:41:16
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os
import shutil
from glob import glob
from multiprocessing import Process
import subprocess

import numpy as np
import torch

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.models.complex_yolo import ComplexYOLO
from bev_detection.utils.utils import cls_id2str
from bev_detection.config import cfg, BASICPATH
from tools.utils import get_prediction_real


def update_cfg():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='Testing Complex-YOLO.')

    parser.add_argument(
        "--use_cuda", action="store_true", default=False,
        help="Whether to use CUDA. Default is `False`.")
    args = parser.parse_args()

    kwargs = {}

    if torch.cuda.is_available():
        if not args.use_cuda:
            warning_str = "CUDA devices are available. " + \
                "It is recommended to run with `--use_cuda`."
            warnings.warn(warning_str)
    elif args.use_cuda:
        warnings.warn(
            "CUDA devices are excepted but not available. CPU will be used.")
        args.use_cuda = False
    kwargs["device"] = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs["split"] = "val"

    cfg.update(vars(args), kwargs)


def cp_kitti_native_eval_script():
    kitti_native_eval_root = cfg.get_kitti_native_eval_root()
    make_script_path = os.path.join(kitti_native_eval_root, "run_make.sh")
    if not os.path.exists(kitti_native_eval_root):
        # os.makedirs(kitti_native_eval_root)

        from_dir = os.path.join(BASICPATH, "tools/kitti_native_eval")
        shutil.copytree(from_dir, kitti_native_eval_root)
        print("Successfully copy KITTI evaulation codes to `{}`.".format(
            kitti_native_eval_root))

        subprocess.call([make_script_path, kitti_native_eval_root])
        print("Sucessfully make KITTI evaluation codes.")


def get_rys(ims, res):
    angles = np.arctan2(ims, res)
    rys = angles + np.pi / 2

    mask = rys > np.pi
    rys[mask] = rys[mask] - 2 * np.pi
    return rys


def get_bboxes_3d(prediction, calib, gt_objs):
    '''Get 3D bounding boxes in rectified camera 0 coordinate.

    Since the height information has not been estimated directly with
    regression, a predefined spatial height location and a predefined height
    are calculated from the mean over all ground truth objects in each
    data/image.

    Args:
        prediction (numpy.ndarray): The prediction results projected to real
            world in lidar coordinate with shape `(N, 8)` where are `x`, `y`,
            `l`, `w`, `im`, `re`, `conf` and `cls_id`.

        Returns:
            numpy.ndarray: The 3D bounding boxes in rectified camera 0
                coordinate with shape `(N, 7)` where are `x`, `y`, `z`, `h`,
                `w`, `l` and `ry` in formate of KITTI (as same as the label
                files).

    '''
    pts_lidar = np.zeros((prediction.shape[0], 4))  # The forth is intensity.
    pts_lidar[:, 0] = prediction[:, 0]  # `x` in lidar coordinate
    pts_lidar[:, 1] = prediction[:, 1]  # `y` in lidar coordinate
    #  `z = 0` in lidar coordinate
    #  Convert to rectified camera 0 coordinate.
    pts_rect, _ = calib.lidar2rect(pts_lidar)

    # Calculate a spatial height location and a height value in rectified
    # camera 0 coordinate.
    height = 0
    y = 0
    active_cls_str = cfg.get_active_cls_str()
    for obj in gt_objs:
        if obj.get_cls_str() == active_cls_str:
            height += obj.get_obj_size()[0]
            y += obj.get_obj_pos()[1]
    height = height / len(gt_objs)
    y = y / len(gt_objs)

    # Get 3D boxes in rectified camera 0 coordinate.
    bboxes = np.zeros((prediction.shape[0], 7))
    bboxes[:, 0] = pts_rect[:, 0]  # x
    bboxes[:, 1] = y  # y
    bboxes[:, 2] = pts_rect[:, 2]  # z
    bboxes[:, 3] = height  # h
    bboxes[:, 4] = prediction[:, 3]  # w
    bboxes[:, 5] = prediction[:, 2]  # l
    bboxes[:, 6] = get_rys(prediction[:, 4], prediction[:, 5])  # ry
    return bboxes


def get_bbox_3d_corners(bbox_3d):
    '''Get corner points of 3D bounding box in rectified camera 0 coordinate.

    Args:
        bbox_3d (numpy.ndarray):

    '''
    x, y, z, h, w, l, ry = bbox_3d

    # Get eight corners of 3D boxes.
    rotation_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]])

    corners_x = np.array(
        [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2])
    corners_y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    corners_z = np.array(
        [l / 2, -l / 2, l / 2, -l / 2, l / 2, -l / 2, l / 2, -l / 2])

    # (N, 3)
    corners_rect = np.dot(
        rotation_mat, np.array([corners_x, corners_y, corners_z])).T

    return corners_rect


def get_bbox_2d(bbox_3d, calib, is_truncate=False, img_size=None):
    corners = get_bbox_3d_corners(bbox_3d)
    # Project the eight corner points to image.
    corners_img, _ = calib.rect2img(corners)  # (N, 2)

    # Get 2D boxes.
    x1 = np.amin(corners_img[:, 0])
    y1 = np.amin(corners_img[:, 1])
    x2 = np.amax(corners_img[:, 0])
    y2 = np.amax(corners_img[:, 1])
    bbox_2d = np.array([x1, y1, x2, y2])

    # Truncation and discard.
    if is_truncate:
        if img_size is None:
            raise ValueError("Image size must be provided.")

        img_W, img_H = img_size
        # Discard invalid box outside the image.
        if bbox_2d[0] > img_W or \
            bbox_2d[1] > img_H or \
            bbox_2d[2] < 0 or \
                bbox_2d[3] < 0:
            return None

        # Discard box that is larger than 80% width or height of image.
        if (bbox_2d[2] - bbox_2d[0]) > (img_W * 0.8) or\
                (bbox_2d[3] - bbox_2d[1]) > (img_H * 0.8):
            return None

        # Truncate the box.
        if bbox_2d[0] < 0:
            bbox_2d[0] = 0
        if bbox_2d[1] < 0:
            bbox_2d[1] = 0
        if bbox_2d[2] > img_W:
            bbox_2d[2] = img_W
        if bbox_2d[3] > img_H:
            bbox_2d[3] = img_H
    return bbox_2d


def get_clss_str(clss_id):
    clss_str = []
    for cls_id in clss_id:
        cls_str = cls_id2str(int(cls_id))
        clss_str.append(cls_str)
    return clss_str


def save_prediction_in_kitti_format(
        prediction_kitti_dir, prediction, dataset, idx):
    '''Extend prediction with 3D information and convert it to kitti format.

    Since the height information has not been estimated directly with
    regression, a predefined spatial height location and a predefined height
    are calculated from the mean over all ground truth objects in each
    data/image.

     Args:
        prediction (numpy.ndarray): The prediction results projected to real
            world in lidar coordinate with shape `(N, 8)` where are `x`, `y`,
            `l`, `w`, `im`, `re`, `conf` and `cls_id`.

    '''
    prediction_kitti_path = os.path.join(
        prediction_kitti_dir, "{:0>6}.txt".format(idx))

    saved_str = \
        "\tSave prediction results ({:0>6}) in KITTI format at `{}`.".format(
            idx, prediction_kitti_path)
    if len(prediction) == 0:
        # No prediction
        np.savetxt(prediction_kitti_path, [])
        print(saved_str)
        return

    prediction = prediction.reshape((-1, 8))

    calib = dataset.get_calib(idx)
    gt_objs = dataset.get_objs(idx)
    _, img_size = dataset.get_img(idx, return_size=True)

    bboxes_3d = get_bboxes_3d(prediction, calib, gt_objs)

    # Get 2D boxes in camera 2 (image) coordinate.
    bboxes_2d = []
    mask = []
    for bbox_3d in bboxes_3d:
        bbox_2d = get_bbox_2d(
            bbox_3d, calib, is_truncate=True, img_size=img_size)

        if bbox_2d is None:
            mask.append(False)
            continue

        mask.append(True)
        bboxes_2d.append(bbox_2d)

    bboxes_2d = np.asarray(bboxes_2d)
    bboxes_3d = bboxes_3d[mask]
    prediction = prediction[mask]

    if len(prediction) == 0:
        np.savetxt(prediction_kitti_path, [])
        print(saved_str)
        return

    prediction_kitti = np.ones((len(prediction), 16))

    # Class
    clss_id = prediction[:, -1]
    clss_str = get_clss_str(clss_id)

    # Alpha
    prediction_kitti[:, 3] = -10 * prediction_kitti[:, 3]

    # 2D bounding box
    prediction_kitti[:, 4: 8] = bboxes_2d

    # Size (h, w, l)
    prediction_kitti[:, 8: 11] = bboxes_3d[:, 3: 6]

    # Position (x, y, z)
    prediction_kitti[:, 11: 14] = bboxes_3d[:, 0: 3]

    # Rotation y
    prediction_kitti[:, 14] = bboxes_3d[:, -1]

    # Score
    prediction_kitti[:, 15] = prediction[:, 6]

    prediction_kitti = np.round(prediction_kitti, 3)

    prediction_kitti_label = np.column_stack((
        clss_str,
        -1 * np.ones((prediction_kitti.shape[0], 2), dtype=np.int32),
        prediction_kitti[:, 3:]))
    np.savetxt(
        prediction_kitti_path, prediction_kitti_label,
        newline="\r\n", fmt='%s')
    print(saved_str)


def get_checkpoints_names(return_epochs=False):
    checkpoints_dir = cfg.get_checkpoints_root()

    checkpoints_names = [os.path.basename(f) for f in glob(
        os.path.join(checkpoints_dir, "*.pth"))]
    if return_epochs:
        epochs = [
            int(fname.split('.')[0].split('_')[-1])
            for fname in checkpoints_names]
        return checkpoints_names, epochs
    return checkpoints_names


def run_kitti_native__eval_script(epoch):
    kitti_native_eval_root = cfg.get_kitti_native_eval_root()
    kitti_native_eval_script_path = os.path.join(
        kitti_native_eval_root, "run_eval.sh")
    label_dir = os.path.join(cfg.get_datasets_root(), "Kitti/training/label_2")
    results_dir = os.path.join(kitti_native_eval_root, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    score_threshold = cfg.get_conf_threshold()
    active_cls = cfg.get_active_cls_str()

    results_fname = active_cls + "_" + str(score_threshold) + "_results.txt"
    results_path = os.path.join(results_dir, results_fname)

    subprocess.call([
        kitti_native_eval_script_path,
        kitti_native_eval_root,
        label_dir,
        str(score_threshold),
        str(epoch),
        results_path])


def kitti_native_eval(epoch):
    eval_process = Process(target=run_kitti_native__eval_script, args=(epoch))
    eval_process.start()


if __name__ == "__main__":
    update_cfg()

    device = cfg.get_device()
    score_threshold = cfg.get_conf_threshold()
    kitti_native_eval_root = cfg.get_kitti_native_eval_root()

    dataset = BEVKitti()
    ids = dataset.get_ids()
    model = ComplexYOLO().to(device)

    cp_kitti_native_eval_script()

    checkpoints_names, epochs = get_checkpoints_names(return_epochs=True)
    for checkpoint_name, epoch in zip(checkpoints_names, epochs):
        print(
            "\n\nEvaluate checkpoint: {}".format(checkpoint_name) +
            "\tEpoch: {}.\n".format(epoch))
        model.load(checkpoint_name)

        prediction_kitti_dir = os.path.join(
            kitti_native_eval_root, str(score_threshold), str(epoch), "data")
        if not os.path.exists(prediction_kitti_dir):
            os.makedirs(prediction_kitti_dir)

        model.eval()
        for idx in ids:
            with torch.no_grad():
                bev = dataset.get_bev(idx)
                bev = torch.tensor(bev, dtype=torch.float32)
                bev = bev.to(device)
                bev = bev.unsqueeze(0)
                output = model(bev)

                prediction_real = get_prediction_real(output)
                prediction_real = prediction_real[
                    prediction_real[..., 6] > score_threshold]

            prediction_real = prediction_real.cpu().numpy()
            save_prediction_in_kitti_format(
                prediction_kitti_dir, prediction_real, dataset, idx)

        kitti_native_eval(epoch)

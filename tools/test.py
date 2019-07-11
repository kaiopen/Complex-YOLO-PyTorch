# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2019/06/12 14:19:48
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import matplotlib.pyplot as plt
import torch

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.models.complex_yolo import ComplexYOLO
from bev_detection.config import cfg
from tools.utils import get_prediction_bev, draw_box


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
            warnings.warn(
                "CUDA devices are available. " +
                "It is recommended to run with `--use_cuda`.")
    elif args.use_cuda:
        warnings.warn(
            "CUDA devices are excepted but not available. CPU will be used.")
        args.use_cuda = False
    kwargs["device"] = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs["split"] = "val"

    cfg.update(vars(args), kwargs)


def show_result(bev, preds, target):
    bev_H = bev.shape[1]
    bev_W = bev.shape[2]

    bev = bev.transpose((1, 2, 0))
    plt.imshow(bev)
    # Draw target
    target = target[target[:, 0] != 0]
    for obj in target:
        draw_box(obj[1: 7], bev_H, bev_W, color='r')

    # Draw prediction
    for obj in preds:
        draw_box(obj[0: 6], bev_H, bev_W, color='g')

    plt.show()


if __name__ == "__main__":
    update_cfg()
    device = cfg.get_device()
    val_dataset = BEVKitti("val")
    model = ComplexYOLO().to(device)
    model.load()
    model.eval()
    for i, (bev, target) in enumerate(val_dataset):
        bev = bev.unsqueeze(0)
        bev = bev.to(device)
        output = model(bev)

        bev = bev.squeeze(0)
        bev = bev.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        prediction_bev = get_prediction_bev(output, bev.shape[1, 2])
        prediction_bev = prediction_bev.cpu().detach().numpy()

        show_result(bev, prediction_bev, target)
        if (i + 1) % 10 == 0:
            plt.close("all")

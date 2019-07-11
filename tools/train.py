# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2019/05/28 10:39:43
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import time
import os

import torch
from torch.utils.data import DataLoader
from torch import optim

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.models.complex_yolo import ComplexYOLO
from bev_detection.utils.region_loss import RegionLoss
from bev_detection.config import cfg


def update_cfg():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='Training Complex-YOLO.')

    parser.add_argument(
        "--use_cuda", action="store_true", default=False,
        help="Whether to use CUDA. Default is `False`.")
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help="Whether to use pinned memory. Default is `False`.")

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
    kwargs["split"] = "train"

    cfg.update(vars(args), kwargs)


def filter_dataset():
    '''Filter dataset according to active class.

    A new split file named `active_train.txt` will be generated.

    '''
    active_split_dir = os.path.join(cfg.get_datasets_cache_root(), "Kitti")
    if not os.path.exists(os.path.dirname(active_split_dir)):
        os.makedirs(active_split_dir)
    active_split_path = os.path.join(active_split_dir, "active_train.txt")

    dataset = BEVKitti()
    ids = dataset.get_ids()

    active_ids = []
    active_clss = cfg.get_active_cls_str()

    for idx in ids:
        objs = dataset.get_objs(idx)
        for obj in objs:
            if obj.get_cls_str() in active_clss:
                active_ids.append("{:0>6}\n".format(idx))
                break
    active_ids[-1] = active_ids[-1][: -1]
    with open(active_split_path, 'w') as f:
        f.writelines(active_ids)

    kwargs = {"split": "active_train"}
    cfg.update(kwargs)


if __name__ == "__main__":
    update_cfg()

    filter_dataset()

    dataset = BEVKitti()

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.get_batch_size(),
        shuffle=True,
        num_workers=cfg.get_num_workers(),
        pin_memory=cfg.get_pin_memory())

    model = ComplexYOLO().to(cfg.get_device())
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get_initial_lr(),
        momentum=cfg.get_momentum(),
        weight_decay=cfg.get_weight_decay())
    region_loss = RegionLoss()

    dataset_size = len(dataset)
    for epoch in range(cfg.get_n_epochs()):
        for group in optimizer.param_groups:
            if epoch >= 4 and epoch < 80:
                group["lr"] = 1e-4
            elif epoch >= 80 and epoch < 160:
                group["lr"] = 1e-5
            elif epoch >= 160:
                group["lr"] = 1e-6

        model.train()
        n_trained = 0
        total_loss = 0
        start_time = time.time()
        for b, (bevs, targets) in enumerate(loader):
            n_trained += bevs.size(0)

            optimizer.zero_grad()
            bevs = bevs.to(cfg.get_device())
            # targets = targets.to(cfg.get_device())
            output = model(bevs)
            loss = region_loss(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if b % 10 == 0:
                print("Train epoch {} [{}/{} {:.2f}%]\tLoss: {:.3f}".format(
                    epoch, n_trained, dataset_size,
                    100. * n_trained/dataset_size, loss.item()))

        end_time = time.time()
        print("Train epoch {}\tAverage Loss: {:.3f}\tTime: {:.3f}s.".format(
            epoch, total_loss / len(loader), end_time-start_time))

        if epoch % 10 == 0:
            model.save(epoch=epoch)

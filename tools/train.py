# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2019/05/28 10:39:43
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import torch
from torch.utils.data import DataLoader
from torch import optim

import init_path
from bev_detection.data.datasets.kitti.bev_kitti import BEVKitti
from bev_detection.models.complex_yolo import ComplexYOLO
from bev_detection.utils.region_loss import RegionLoss
from bev_detection.config import cfg
from tools.evaluate import evaluate


def update_cfg():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='Training Complex-YOLO.')

    # Parameters for precessing and dataset
    # parser.add_argument(
    #     "--dataset", type=str, choices=["Kitti"],
    #     help="Dataset. `Kitti` is available. Default is `Kitti`.")

    # Parameters for dataloader
    parser.add_argument(
        "--batch_size", type=int, default=12,
        help='Batch size. Default is `12`.')

    # Parameters for training
    # parser.add_argument(
    #     '--n_epochs', type=int, default=1000,
    #     help='The number of epochs. Default is `1000`.')
    # parser.add_argument(
    #     "--initial_lr", type=float, default=1e-5,
    #     help="Initial learning rate. Default is `1e-5`.")
    # parser.add_argument(
    #     "--momentum", type=float, default=0.9,
    #     help="Momentum coefficient. Default is `0.9`.")
    # parser.add_argument(
    #     "--weight_decay", type=float, default=0.0005,
    #     help="Weight decay coefficient. Default is `0.0005`.")

    parser.add_argument(
        "--use_cuda", action="store_true", default=False,
        help="Whether to use CUDA. Default is `False`.")
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="The number of workers/processes to loader data. Default is `8`.")
    parser.add_argument(
        "--pin_memory", action="store_true", default=False,
        help="Whether to use pinned memory. Default is `False`.")

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

    cfg.update(vars(args), kwargs)


if __name__ == "__main__":
    update_cfg()

    train_dataset = BEVKitti("train")
    val_dataset = BEVKitti("val")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.get_batch_size(),
        shuffle=True,
        num_workers=cfg.get_num_workers(),
        pin_memory=cfg.get_pin_memory())

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.get_batch_size(),
        num_workers=cfg.get_num_workers(),
        pin_memory=cfg.get_pin_memory())

    model = ComplexYOLO().to(cfg.get_device())
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get_initial_lr(),
        momentum=cfg.get_momentum(),
        weight_decay=cfg.get_weight_decay())
    region_loss = RegionLoss()

    dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    for epoch in range(1, cfg.get_n_epochs() + 1):
        for group in optimizer.param_groups:
            if epoch >= 4 and epoch < 80:
                group["lr"] = 1e-4
            elif epoch >= 80 and epoch < 160:
                group["lr"] = 1e-5
            elif epoch >= 160:
                group["lr"] = 1e-6

        model.train()
        n_trained = 0
        cumul_n_gt = 0
        cumul_n_correct = 0
        for b, (bevs, targets) in enumerate(train_loader):
            n_trained += bevs.size(0)

            optimizer.zero_grad()
            bevs = bevs.to(cfg.get_device())
            targets = targets.to(cfg.get_device())
            output = model(bevs)
            loss, statistics = region_loss(output, targets)
            loss.backward()
            optimizer.step()

            n_gt = statistics["n_gt"]
            n_correct = statistics["n_correct"]
            cumul_n_gt += n_gt
            cumul_n_correct += n_correct
            if (b + 1) % 10 == 0:
                print("Train epoch {} [{}/{} {:.2f}%]\tLoss: {:.3f}".format(
                    epoch, n_trained, dataset_size,
                    100. * n_trained/dataset_size, loss.item()))
        accuracy = float(cumul_n_correct / cumul_n_gt) if n_gt else 1
        print("Train epoch {}\tAccuracy: {:.3f}.".format(epoch, accuracy))
        print("\n")

        evaluate(model, val_loader, region_loss, val_dataset_size, epoch)
        print("\n")

        if epoch % 50 == 0:
            model.save()

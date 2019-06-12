# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2019/06/06 11:42:40
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import init_path
from bev_detection.config import cfg


def evaluate(model, loader, region_loss, dataset_size, epoch):
    model.eval()
    device = cfg.get_device()
    n_evaled = 0
    cumul_n_gt = 0
    cumul_n_correct = 0
    for b, (bevs, targets) in enumerate(loader):
        n_evaled += bevs.size(0)
        bevs = bevs.to(device)
        targets = targets.to(device)
        output = model(bevs)
        loss, statistics = region_loss(output, targets)

        n_gt = statistics["n_gt"]
        n_correct = statistics["n_correct"]
        cumul_n_gt += n_gt
        cumul_n_correct += n_correct
        if (b + 1) % 10 == 0:
            print("Eval epoch {} [{}/{} {:.2f}%]\tLoss: {:.3f}".format(
                epoch, n_evaled, dataset_size,
                100. * n_evaled/dataset_size, loss.item()))
    accuracy = float(cumul_n_correct / cumul_n_gt) if n_gt else 1
    print("Eval epoch {}\tAccuracy: {:.3f}.".format(epoch, accuracy))
    print("\n")

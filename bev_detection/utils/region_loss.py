# -*- encoding: utf-8 -*-
'''
@File    :   region_loss.py
@Time    :   2019/05/30 09:54:20
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import math

import torch
from torch import nn

from bev_detection.config import cfg


def get_fmap_prediction(output):
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)
    anchors = cfg.get_anchors().type_as(output)
    n_anchors = anchors.size(0)
    n_classes = cfg.get_n_classes()

    output = output.reshape(
        B, n_anchors, (7 + n_classes), H, W).permute(0, 1, 3, 4, 2)
    t_x = torch.sigmoid(output[..., 0])
    t_y = torch.sigmoid(output[..., 1])
    t_l = output[..., 2]
    t_w = output[..., 3]

    c_x = torch.arange(H).type_as(output).repeat(W, 1).t()
    c_x = c_x.reshape((1, 1, H, W))
    c_y = torch.arange(W).type_as(output).repeat(H, 1)
    c_y = c_y.reshape((1, 1, H, W))
    p_l = anchors[:, 0].reshape((1, n_anchors, 1, 1))
    p_w = anchors[:, 1].reshape((1, n_anchors, 1, 1))

    fmap_preds = output.new_empty(output[..., 0: 8].shape)
    fmap_preds[..., 0] = t_x + c_x
    fmap_preds[..., 1] = t_y + c_y
    fmap_preds[..., 2] = torch.exp(t_l) * p_l
    fmap_preds[..., 3] = torch.exp(t_w) * p_w
    fmap_preds[..., 4] = output[..., 4]
    fmap_preds[..., 5] = output[..., 5]
    fmap_preds[..., 6] = torch.sigmoid(output[..., 6])
    fmap_preds[..., 7] = torch.argmax(torch.sigmoid(output[..., 7:]))
    return fmap_preds


def build_targets(targets, shape, fmap_preds):
    B = shape[0]
    n_anchors = shape[1]
    H = shape[2]
    W = shape[3]
    fmap_preds = fmap_preds.cpu()
    targets = targets.cpu()

    anchors = cfg.get_anchors().reshape((-1, 2)).type(torch.float32)
    anchor_boxes = torch.cat((torch.zeros(n_anchors, 2), anchors), dim=1)

    mask = torch.zeros(B, n_anchors, H, W).type(torch.uint8)
    conf_mask = torch.ones(B, n_anchors, H, W).type(torch.uint8)
    ts = torch.zeros(shape, dtype=torch.float32)
    gt_boxes = torch.zeros((B, n_anchors, H, W, 6), dtype=torch.float32)

    n_gt = 0
    n_correct = 0
    conf_threshold = cfg.get_conf_threshold()
    for b in range(B):
        for o in range(targets.size(1)):
            if targets[b, o, 0] == 0:
                break

            n_gt += 1
            # Shift ground truth boxes to feature map with size `W` * `H`.
            gt_x = targets[b, o, 1] * H
            gt_y = targets[b, o, 2] * W
            gt_l = targets[b, o, 3] * H
            gt_w = targets[b, o, 4] * W
            gt_im = targets[b, o, 5]
            gt_re = targets[b, o, 6]

            gt_box = torch.tensor(
                [0, 0, gt_l, gt_w], dtype=torch.float32)
            ious = get_ious(gt_box, anchor_boxes)
            actived_anchor = torch.argmax(ious)
            actived_x = int(gt_x)
            actived_y = int(gt_y)

            gt_boxes[b, actived_anchor, actived_x, actived_y, 0] = gt_x
            gt_boxes[b, actived_anchor, actived_x, actived_y, 1] = gt_y
            gt_boxes[b, actived_anchor, actived_x, actived_y, 2] = gt_l
            gt_boxes[b, actived_anchor, actived_x, actived_y, 3] = gt_w
            gt_boxes[b, actived_anchor, actived_x, actived_y, 4] = gt_im
            gt_boxes[b, actived_anchor, actived_x, actived_y, 5] = gt_re

            # Build target.
            ts[b, actived_anchor, actived_x, actived_y, 0] = gt_x - actived_x
            ts[b, actived_anchor, actived_x, actived_y, 1] = gt_y - actived_y
            ts[b, actived_anchor, actived_x, actived_y, 2] = math.log(
                gt_l / anchors[actived_anchor, 0])
            ts[b, actived_anchor, actived_x, actived_y, 3] = math.log(
                gt_w / anchors[actived_anchor, 1])
            ts[b, actived_anchor, actived_x, actived_y, 4] = gt_im
            ts[b, actived_anchor, actived_x, actived_y, 5] = gt_re
            ts[b, actived_anchor, actived_x, actived_y, 6] = 1
            obj_cls_id = int(targets[b, o, 0])
            ts[b, actived_anchor, actived_x, actived_y, 7 + obj_cls_id] = 1

            ignore_threshold = cfg.get_ignore_threshold()
            conf_mask[b, ious > ignore_threshold, actived_x, actived_y] = 0
            conf_mask[b, actived_anchor, actived_x, actived_y] = 1
            mask[b, actived_anchor, actived_x, actived_y] = 1

            # Do statistics.
            pred = fmap_preds[b, actived_anchor, actived_x, actived_y]
            pred_box = pred[:4].reshape((-1, 4))
            gt_box = torch.tensor(
                [gt_x, gt_y, gt_l, gt_w], dtype=torch.float32)
            iou = get_ious(gt_box, pred_box)
            conf = pred[6]
            # cls_id = pred[7]
            if iou > 0.5 and conf > conf_threshold:
                n_correct += 1

    statistics = {
        "n_gt": n_gt,
        "n_correct": n_correct}
    true_mask = mask
    false_mask = conf_mask - mask
    return ts, gt_boxes, true_mask, false_mask, statistics


def get_ious(gt, boxes):
    boxes = boxes.reshape((-1, 4))
    gt_l = gt[2]
    gt_w = gt[3]
    boxes_l = boxes[:, 2]
    boxes_w = boxes[:, 3]
    min_x = torch.min(
        gt[0] - gt_l / 2., boxes[:, 0] - boxes_l / 2.)
    max_x = torch.max(
        gt[0] + gt_l / 2., boxes[:, 0] + boxes_l / 2.)
    min_y = torch.min(
        gt[1] - gt_w / 2., boxes[:, 1] - boxes_w / 2.)
    max_y = torch.max(
        gt[1] + gt_w / 2., boxes[:, 1] + boxes_w / 2.)
    union_w = max_y - min_y
    union_h = max_x - min_x
    inter_w = gt_w + boxes_w - union_w
    inter_l = gt_l + boxes_l - union_h
    mask = ((inter_w <= 0) + (inter_l <= 0) > 0)
    gt_area = gt_w * gt_l
    boxes_areas = boxes_w * boxes_l
    inter_areas = inter_w * inter_l
    inter_areas[mask] = 0
    union_areas = gt_area + boxes_areas - inter_areas
    return inter_areas / union_areas


class RegionLoss(nn.Module):
    '''Docstring for RegionLoss'''
    def __init__(self):
        super(RegionLoss, self).__init__()
        device = cfg.get_device()
        self.__mse_loss = nn.MSELoss(reduction="mean").to(device)
        self.__bce_loss = nn.BCELoss(reduction="mean").to(device)
        self.__ce_loss = nn.CrossEntropyLoss().to(device)

    def forward(self, output, targets):
        '''
        Args:
            output (torch.tensor): Feature maps with shape
                `(batch_size, n_anchors * (7 + n_classes), h, w)`.

        '''
        LongTensor = torch.cuda.LongTensor \
            if output.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor \
            if output.is_cuda else torch.ByteTensor

        B = output.size(0)
        H = output.size(2)
        W = output.size(3)
        anchors = cfg.get_anchors().type_as(output)
        n_anchors = anchors.size(0)
        n_classes = cfg.get_n_classes()

        fmap_preds = get_fmap_prediction(output)

        output = output.reshape(
            B, n_anchors, (7 + n_classes), H, W).permute(0, 1, 3, 4, 2)
        o_x = torch.sigmoid(output[..., 0])
        o_y = torch.sigmoid(output[..., 1])
        o_l = output[..., 2]
        o_w = output[..., 3]
        o_im = output[..., 4]
        o_re = output[..., 5]
        o_conf = torch.sigmoid(output[..., 6])
        o_cls = torch.sigmoid(output[..., 7:])

        shape = (B, n_anchors, H, W, 7 + n_classes)
        ts, gt_boxes, true_mask, false_mask, statistics = build_targets(
            targets.cpu(), shape, fmap_preds.cpu())

        # loss
        true_mask = true_mask.type(ByteTensor)
        false_mask = false_mask.type(ByteTensor)

        t_x = ts[..., 0].type_as(output).requires_grad_(False)
        t_y = ts[..., 1].type_as(output).requires_grad_(False)
        t_l = ts[..., 2].type_as(output).requires_grad_(False)
        t_w = ts[..., 3].type_as(output).requires_grad_(False)
        t_im = ts[..., 4].type_as(output).requires_grad_(False)
        t_re = ts[..., 5].type_as(output).requires_grad_(False)
        t_conf = ts[..., 6].type_as(output).requires_grad_(False)
        t_cls = ts[..., 7:].type(LongTensor).requires_grad_(False)

        loss_x = self.__mse_loss(o_x[true_mask], t_x[true_mask])
        loss_y = self.__mse_loss(o_y[true_mask], t_y[true_mask])
        loss_l = self.__mse_loss(o_l[true_mask], t_l[true_mask])
        loss_w = self.__mse_loss(o_w[true_mask], t_w[true_mask])
        loss_im = self.__mse_loss(o_im[true_mask], t_im[true_mask])
        loss_re = self.__mse_loss(o_re[true_mask], t_re[true_mask])
        loss_conf1 = self.__bce_loss(o_conf[false_mask], t_conf[false_mask])
        loss_conf2 = self.__bce_loss(o_conf[true_mask], t_conf[true_mask])
        loss_conf = loss_conf1 + loss_conf2
        loss_cls = (1 / B) * self.__ce_loss(
            o_cls[true_mask], torch.argmax(t_cls[true_mask], 1))
        loss = loss_x + loss_y + loss_w + loss_l + \
            loss_im + loss_re + loss_conf + loss_cls
        return loss, statistics

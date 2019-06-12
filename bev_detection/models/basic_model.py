# -*- encoding: utf-8 -*-
'''
@File    :   basic_model.py
@Time    :   2019/05/27 09:54:36
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os
import time
from glob import glob

import torch
from torch import nn

from bev_detection.config import cfg


class BasicModule(nn.Module):
    __checkpoints_dir = cfg.get_checkpoints_root()

    def __init__(self):
        super(BasicModule, self).__init__()
        self.__model_name = self.__class__.__name__

    def load(self, fname=None):
        if fname is None:
            fnames = [f.split('_')[-1] for f in glob(
                os.path.join(self.__checkpoints_dir, "*.pth"))]
            fname = self.__model_name + '_' + max(fnames)
        elif fname.split('.')[-1] != ".pth":
            fname = fname + ".pth"

        checkpoint_path = os.path.join(self.__checkpoints_dir, fname)
        if cfg.get_device().type == "cpu":
            self.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"))
        else:
            self.load_state_dict(torch.load(checkpoint_path))



    def save(self, fname=None):
        if not os.path.exists(self.__checkpoints_dir):
            os.makedirs(self.__checkpoints_dir)

        if fname is None:
            fname = time.strftime(self.__model_name + "_%Y%m%d%H%M%S.pth")
        elif fname.split('.')[-1] != ".pth":
            fname = fname + ".pth"

        checkpoint_path = os.path.join(self.__checkpoints_dir, fname)
        torch.save(self.state_dict(), checkpoint_path)

# -*- encoding: utf-8 -*-
'''
@File    :   init_path.py
@Time    :   2019/05/18 12:02:49
@Author  :   Painter
@Contact :   painter9509@126.com

'''

import os, sys

BASICPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASICPATH)
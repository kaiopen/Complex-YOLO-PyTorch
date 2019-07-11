# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/07/08 19:05:31
@Author  :   Painter
@Contact :   painter9509@126.com

'''

clss_str = (
    "__background__",
    "Car", "Van", "Truck",
    "Tram",
    "Pedestrian", "Person_sitting", "Cyclist",
    "Misc", "DontCare")


def cls_str2id(cls_str):
    return clss_str.index(cls_str)


def cls_id2str(cls_id):
    return clss_str[cls_id]

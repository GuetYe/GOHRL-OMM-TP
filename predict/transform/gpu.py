#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2024/12/05 20:10:35
@File: dataset.py
@Decs:*********************
"""

# 是否使用GPU进行加速
import torch


def gpu():
    """
    GPU加速指令
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 检查是否可用GPU进行加速
    return device

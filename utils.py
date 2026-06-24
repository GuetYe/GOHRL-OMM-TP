#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2025/1/6 15:37
@File: utils.py
@Decs:*********************
"""
import copy
import numpy as np
import torch
import os

def compute_advantage(gamma, lmbda, td_delta):
    """
    广义优势估计，实现优势函数（Advantage Function）的计算
    :param gamma: 折扣因子γ
    :param lmbda: 平滑参数𝜆
    :param td_delta: 时序差分误差
    :return: 每个时间步的优势值
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []  # 存储计算得到的优势估计值
    advantage = 0.0
    # 反向遍历时累积计算优势估计
    for delta in td_delta[::-1]:
        # 使用 GAE 公式递推计算每个时间步的优势估计：  At = δt + γ*λ*A(t+1)
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()  # 反向计算的，需要将元素顺序反转
    advantage_array = np.array(advantage_list)
    return torch.tensor(advantage_array, dtype=torch.float32)

def save_model(model, save_path, model_name):
    model_file = os.path.join(save_path, model_name)
    torch.save(model.state_dict(), model_file)

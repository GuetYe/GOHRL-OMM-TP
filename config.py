# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2026/1/14 14:33:45
@Author  :   LFY 
@Contact :   2545039870@qq.com
'''

import torch
import numpy as np
from pathlib import Path

# ------------------数据类型dtype--------------------------#
NUMPY_TYPE = np.float32
TORCH_TYPE = torch.float32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径
WORK_DIR = Path.cwd().parent  # D:\Desktop


# 服务器路径
# LINKS_INFO = WORK_DIR / "pycharm_project_521/DataSet/node14/links_info.xml"  # 链路信息的xml文件路径
# PICKLE_PATH = WORK_DIR / "pycharm_project_521/DataSet/node14/weight_pickle"  # 数据路径
LINKS_INFO = WORK_DIR / "Overlay-Multicast-predict-RL/DataSet/node21/links_info.xml"  # 链路信息的xml文件路径
PICKLE_PATH = WORK_DIR / "Overlay-Multicast-predict-RL/DataSet/node21/weight_pickle"  # 数据路径


# 奖励折扣率
REWARD_DISCOUNT = 0.95

# 动作和状态的数量
STATE_NUM = 4  # 采用的QoS指标数量
ACTION_NUM = 21  # 节点的个数，图节点不同们需要修改，14个节点为14；目前21节点

# 奖励函数：free_bw, delay, loss, used_bw, pkt_err, pkt_drop, distance
PARAM_WEIGHT = [0.7, 0.4, 0.1]

#  执行动作后的状态[end, part, loop]，[5, 1, -2, -0.1]
# 第一个点
# 10节点：[3,  -0.25, -1]
# 14节点：[2,  -0.25, -1]
# # 21节点：[1.5,  -0.25, -1]

#第二个点
INTERNAL_REWARD = [5,  1, -3]


# 下层反馈上层的奖励 1，-1
EXTERNAL_REWARD = [-0.1, -3]

# 源目的节点，真是节点1-14
# 14节点：src_node = 3，dst_lst = [4, 7, 14]
# 21节点：src_node = 1，dst_lst = [16, 11, 19]
src_node = 1
dst_lst = [16, 11, 19]

# 训练参数
episodes = 1000  # 训练周期
# 图数据的开始，终止，步长4.14，从picture_end：110 -》210（10节点跟21节点使用）
# 10节点修改未110
picture_start = 5
picture_end = 50
picture_step = 5
picture_count = len(range(picture_start, picture_end, picture_step))  # 图片的数量


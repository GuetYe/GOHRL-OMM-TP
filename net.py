#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2024/12/12 16:44
@File: net.py
@Decs:*********************
"""
import torch
import torch.nn as nn
import config
import torch.nn.functional as F

from env import *


# 外层PPO策略网络
class PPOPolicyNet(nn.Module):
    def __init__(self, input_shape=(config.STATE_NUM, config.ACTION_NUM), action_num=config.ACTION_NUM):
        super(PPOPolicyNet, self).__init__()
        self.input_shape = input_shape  # 节点个数 * 节点个数 * QoS指标数 ，14节点：14x14x4；21节点：21x21x4
        self.action_num = action_num  # 动作的数量，用于选择动作
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=config.STATE_NUM, out_channels=32, kernel_size=3, stride=1, padding=0)  # 14节点输出： 32x12x12；21节点输出： 32x19x19
        # 14节点
        # self.fc1 = nn.Linear(32 * 12 * 12, 32)
        # 21节点
        self.fc1 = nn.Linear(32 * 19 * 19, 32)
        self.action_head = nn.Linear(32, self.action_num)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.action_head(x)
        return x


# 内层PPO价值网络
class PPOValueNet(nn.Module):
    def __init__(self):
        super(PPOValueNet, self).__init__()
        self.conv1 = nn.Conv2d(config.STATE_NUM, 32, kernel_size=(1, 1))  # 输出： 32x12x12
        self.fc1 = nn.Linear(32 * config.ACTION_NUM * config.ACTION_NUM, 32)
        self.fc2 = nn.Linear(32, 16)
        self.state_value = nn.Linear(16, 1)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q_values = self.state_value(x)   # 输出动作 Q 值
        return q_values

# 内层AC策略网络
class ACPolicyNet(nn.Module):
    def __init__(self):
        super(ACPolicyNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(config.STATE_NUM, out_channels=32, kernel_size=3, stride=1, padding=0)  # 输出： 128x12x12
        # 14节点
        # self.fc1 = nn.Linear(32 * 12 * 12, 32)
        # 21节点
        self.fc1 = nn.Linear(32 * 19 * 19, 32)
        self.action_head = nn.Linear(32, config.ACTION_NUM)  # 输出动作 Q 值

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # 输出形状因该与节点数量相同
        x = F.leaky_relu(self.fc1(x))
        x = self.action_head(x)
        return x

# 内层AC价值网络
class ACValueNet(nn.Module):
    def __init__(self):
        super(ACValueNet, self).__init__()
        self.conv1 = nn.Conv2d(config.STATE_NUM, 32, kernel_size=(1, 1))  # 输出： 32x12x12
        self.fc1 = nn.Linear(32 * config.ACTION_NUM * config.ACTION_NUM, 32)
        self.fc2 = nn.Linear(32, 16)
        self.state_value = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q_value = self.state_value(x)   # 输出动作 Q 值
        return q_value


if __name__ == '__main__':
    # # 假设输入维度为 (batch_size, 14, 14, 7)
    # input_tensor = torch.randn(32, 14, 14, 7)  # Batch size 为 32


    #model = PPOPolicyNet()
    #model = PPOValueNet()
    #model = ACPolicyNet()
    model = ACValueNet()
    # output = model.forward(input_tensor)
    # print("输出Q值维度:", output, output.shape)  # 输出应为 (batch_size, num_actions)
    env = Environment()
    for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=13, n=14, step=2)):
        env.update_pkl_graph(read_pkl(pkl_path))
        env.reset()

        # a = env.combined_normal_matrix[0, 0, :, :]
        #print(env.combined_normal_matrix.shape, env.combined_normal_matrix[0, 0, :, :])
        # # 获取对角线上值为 -1 和 1 的索引
        # indices_minus_1 = (a == -1).nonzero(as_tuple=True)[0]
        # indices_1 = (a == 1).nonzero(as_tuple=True)[0]
        # print(indices_minus_1[0], indices_minus_1[1], indices_minus_1[2], indices_1[0])
        # print(indices_minus_1, indices_1)
        # a = torch.tensor(env.combined_normal_matrix)

        # print('lfylfylfylfylfy', a, a.shape)

        out = model.forward(env.combined_normal_matrix)
        print(out, out.shape)

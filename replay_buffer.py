#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2024/12/16 15:38
@File: replay_buffer.py
@Decs:*********************
"""

from collections import deque, namedtuple
import random
import torch



class UpperAcReplayBuffer():
    ''' 上层AC经验回放池 '''

    def __init__(self, buffer_size, batch_size):
        # 创建一个双端队列(先进先出)，设置最大长度为capacity
        self.memory = deque(maxlen=buffer_size)  # 队列,先进先出
        self.batch_size = batch_size  # 采样的批量大小
        # 当前状态，目标，奖励， 下一刻状态，终止标志位
        self.experience = namedtuple("Experience", field_names=["state", "subgoal", "reward", "next_state"])

    def add(self, state, goal, reward, next_state):
        """
        向缓冲区添加新的经验元组
        :param state: 当前状态，tensor([6])	tensor([5, 7, 8])
        :param goal: 当前目标， [tensor([6]), tensor([8])]
        :param reward: 当前奖励， int(num)
        :param next_state: 下一刻状态,tensor([6, 8])	tensor([5, 7])
        """
        transitions = self.experience(state, goal, reward, next_state)

        self.memory.append(transitions)  # 经验元组添加到缓冲区中

    def sample(self):
        """
        从memory中随机采样batch_size数量的经验
        :return: 状态、目标、奖励、下一个状态
        """
        # 如果当前缓冲区中的数据不足 batch_size，直接返回所有样本
        if self.batch_size > len(self.memory):
            return self.return_all_samples()
        else:  # 从memory中随机抽取batch_size个经验

            transitions = random.sample(self.memory, self.batch_size)  # 从buffer中随机抽取batch_size个经验
            state, goal, reward, next_state = zip(*transitions)  # 将采样经验解压成状态、动作、奖励、下一个状态的元组
            return state, goal, reward, next_state

    def return_all_samples(self):
        """
        返回memory中的所有的经验
        :return:
        """
        all_transitions = self.size()
        transitions = random.sample(self.memory, all_transitions)  # 从buffer中随机抽取全部经验
        state, goal, reward, next_state, = zip(*transitions)  # 将采样经验解压成状态、动作、奖励、下一个状态的元组
        return state, goal, reward, next_state

    def size(self):
        # 返回当前buffer中存储的经验数量
        return len(self.memory)

    def clear(self):
        self.memory.clear()


class UnderAcReplayBuffer():
    ''' 下层AC经验回放池 '''

    def __init__(self, buffer_size, batch_size):
        # 创建一个双端队列(先进先出)，设置最大长度为capacity
        self.memory = deque(maxlen=buffer_size)  # 队列,先进先出
        self.batch_size = batch_size  # 采样的批量大小
        # 当前状态，目标，奖励， 下一刻状态，终止标志位
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        :param state: 当前状态矩阵
        :param action: 选择的动作
        :param reward: 当前动作对应的奖励
        :param next_state: 下一刻状态
        :param done: 当前动作标志
        :return:
        """
        transitions = self.experience(state, action, reward, next_state, done)
        #print("经验添加成功")
        self.memory.append(transitions)  # 经验元组添加到缓冲区中

    def sample(self):
        """
        从memory中随机采样batch_size数量的经验
        :return: 状态、目标、奖励、下一个状态
        """
        # 如果当前缓冲区中的数据不足 batch_size，直接返回所有样本
        if self.batch_size > len(self.memory):
            return self.return_all_samples()
        else:  # 从memory中随机抽取batch_size个经验
            transitions = random.sample(self.memory, self.batch_size)  # 从buffer中随机抽取batch_size个经验
            state, action, reward, next_state, done = zip(*transitions)  # 将采样经验解压成状态、动作、奖励、下一个状态的元组`
            return state, action, reward, next_state, done

    def return_all_samples(self):
        """
        返回memory中的所有的经验
        :return:
        """
        all_transitions = self.size()
        transitions = random.sample(self.memory, all_transitions)  # 从buffer中随机抽取全部经验
        state, action, reward, next_state, done = zip(*transitions)  # 将采样经验解压成状态、动作、奖励、下一个状态的元组
        return state, action, reward, next_state, done

    def size(self):
        # 返回当前buffer中存储的经验数量
        # print("当前buffer中存储的经验数量: ", len(self.memory))
        return len(self.memory)

    def clear(self):
        self.memory.clear()

# 用于测试
def generate_data(num_samples, max_value=10):
    data = []
    for _ in range(num_samples):
        state = [torch.randint(0, max_value, (2,)), torch.randint(0, max_value, (2,))]  # 随机生成一个2x2的Tensor
        goal = [torch.randint(0, max_value, (1,)), torch.randint(0, max_value, (1,))]  # 随机生成一个2x1的Tensor
        reward = torch.tensor(random.randint(0, 2))  # 随机生成一个奖励值，0或1
        next_state = [torch.randint(0, max_value, (2,)), torch.randint(0, max_value, (2,))]
        data.append((state, goal, reward, next_state))
    return data


if __name__ == '__main__':
    # ---------------------------------test---------------------------------#
    replay_buffer = DqnReplayBuffer(54, 64)
    state = [torch.tensor([6]), torch.tensor([5, 7, 8])]
    reward = torch.tensor(1)
    subgoal = [torch.tensor([6]), torch.tensor([8])]
    next_state = [torch.tensor([6, 8]), torch.tensor([5, 7])]

    replay_buffer.add(state, subgoal, torch.tensor(1), next_state)
    # 生成1000条数据
    data = generate_data(1000)
    for i in data:
        state, subgoal, reward, next_state = i[0], i[1], i[2], i[3]
        replay_buffer.add(state, subgoal, reward, next_state)

    replay_buffer.size()
    state, goal, reward, next_state = replay_buffer.sample()
    print(state, goal, reward, next_state, sep='\n')
    print(len(state), len(goal), len(reward), len(next_state))
    replay_buffer.size()

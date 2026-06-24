#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/1/14 13:09
@File: UpperAC.py
@Decs:*********************
"""

import copy

import torch

from parse_config import parse_config
from net import PPOPolicyNet, PPOValueNet
from config import *

from replay_buffer import UpperAcReplayBuffer
from utils import compute_advantage
from parse_config import get_save_path, save_train_parmeter
from dataset import get_node_neighbors, get_all_pairs_dijkstra_path_length
import torch.nn.functional as F

from env import *

# 设置打印选项，保证不省略任何元素
torch.set_printoptions(threshold=float('inf'))

class UpperActorCritic:
    def __init__(self, external_config: Path = None):
        config = parse_config(external_config)
        self.policy_lr = float(config['policy_lr'])
        self.value_lr = float(config['value_lr'])

        self.gamma = config['reward_decay']  # 折扣因子
        self.batch_size = config['batch_size']  # 每次从经验池中采样的样本数量

        self.actor = PPOPolicyNet().to(DEVICE)  # 策略网络
        self.critic = PPOValueNet().to(DEVICE)  # 价值网络

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.value_lr)
        # 学习率调度器，当优化器的 step 达到 2500 时，调整学习率

        # 定义经验回放池
        self.memory_capacity = int(config['memory_capacity'])
        self.memory_pool = UpperAcReplayBuffer(buffer_size=self.memory_capacity, batch_size=self.batch_size)

        # 邻居节点，{node：[neighbors_node]}：
        # {1: [3, 4, 5, 11], 2: [4, 5, 6], 3: [1], 4: [1, 2, 5, 6, 9, 11, 13], 5: [1, 2, 4, 6, 7, 8],
        # 6: [2, 4, 5], 7: [5, 9], 8: [5, 9, 14], 9: [4, 7, 8, 10, 13], 10: [9, 13], 11: [1, 4, 14], 12: [13, 14],
        # 13: [4, 9, 10, 12], 14: [8, 11, 12]}
        self.neighbors = get_node_neighbors()  # 邻居节点，{node：[neighbors_node]}  在动作选择时选择邻居节点
        self.all_hops = get_all_pairs_dijkstra_path_length()  # 返回两节点之间的最短距离字典{src：{dst：distance，。。。}}
        self.nodes = len(self.neighbors)  # 节点个数

    def take_action(self, state, src_lst, dst_lst):
        """
        动作选择，
        :param state: [bw，delay，loss，distance，location]
        :param src_lst: 源节点列表
        :param dst_lst: 目的节点列表
        :return: 源目的节点对，位置索引:[0,节点数量-1], 离散[源，目的]；当前状态；下一个状态
        """
        #print(f"每次选择子目标的src_lst = {src_lst}, dst_lst = {dst_lst}")
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(DEVICE)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        next_state_matrix = copy.deepcopy(state)

        probs = self.actor(state)

        # 做one-hot编码，源目的节点位置索引为1
        src_mask = torch.zeros(self.nodes, dtype=torch.float32, device=DEVICE)
        src_mask[[s - 1 for s in src_lst]] = 1.0
        dst_mask = torch.zeros(self.nodes, dtype=torch.float32, device=DEVICE)
        dst_mask[[d - 1 for d in dst_lst]] = 1.0

        src_mask = src_mask.unsqueeze(0)  # [1, 14]
        dst_mask = dst_mask.unsqueeze(0)  # [1, 14]

        # ---------- 4. mask 非法动作 ----------
        src_logits = probs.masked_fill(src_mask == 0, -1e9)
        dst_logits = probs.masked_fill(dst_mask == 0, -1e9)

        dist_src = torch.distributions.Categorical(logits=src_logits)
        dist_dst = torch.distributions.Categorical(logits=dst_logits)

        src_idx = dist_src.sample()
        dst_idx = dist_dst.sample()

        #  ---------- 5. 更新下一刻状态 ----------
        next_state_matrix[0, 0, dst_idx, dst_idx] = 1.0

        return [src_idx.item(), dst_idx.item()], state, next_state_matrix

    def store(self, state, subgoal, reward, next_state):
        """
        将一个 (state, subgoal, reward, next_state) 元组存储到 经验回放池 (memory_pool) 中，供后续采样和训练
        :param state: 当前状态，状态矩阵
        :param subgoal: 当前目标， [tensor([6]), tensor([8])]
        :param reward: 当前奖励， int(num)
        :param next_state: 下一刻状态,状态矩阵
        :return: None
        """
        self.memory_pool.add(state, subgoal, reward, next_state)

    def _get_batch_vars(self):
        """
        从经验池中随机采样一批数据(小批量样本)，用于训练
        :return:
        """
        state, subgoals, reward, next_state = self.memory_pool.sample()
        return state, subgoals, reward, next_state

    def update(self):
        states, subgoals, rewards, next_states = self._get_batch_vars()
        states_batchs = torch.cat(states).to(DEVICE)
        actions_batchs = torch.tensor(subgoals, dtype=torch.long).view(-1,2).to(DEVICE)
        #actions_batchs = torch.cat(subgoals).view(-1, 1).to(DEVICE)
        rewards_batchs = torch.tensor(rewards, dtype=TORCH_TYPE).unsqueeze(1).to(DEVICE)  # 调整好形状： torch.Size([50, 1])
        next_states_batchs = torch.cat(next_states).to(DEVICE)

        # 1. 计算 TD 误差
        td_target = rewards_batchs + self.gamma * self.critic(next_states_batchs)
        td_delta = td_target - self.critic(states_batchs)

        # 拆解子目标
        src = actions_batchs[:, 0].view(-1, 1)
        dst = actions_batchs[:, 1].view(-1, 1)
        probs = torch.softmax(self.actor(states_batchs), dim=1)
        log_probs = torch.log(probs.gather(1, src) + 1e-7) + torch.log(probs.gather(1, dst) + 1e-7)

        # 计算策略网络的损失 = -log(当前策略选择动作概率) * 时序差分A(st, at)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 价值网络的损失函数，均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states_batchs), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()



if __name__ == '__main__':
    upper_agent = UpperActorCritic('./config/external_config.yaml')
    env = Environment()
    for episode in range(1):
        ep_reward = 0  # 一个episode的所有图的奖励和
        # for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=config.picture_start, n=config.picture_end,
        #                                                  step=config.picture_step)):  # 前一个(PICKLE_PATH, s=3, n=6, step=1))
        for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=5, n=6, step=1)):  # 前一个(PICKLE_PATH, s=3, n=6, step=1))
            # 在线强化学习
            transition_dict = {'states': [], 'subgoal': [], 'next_states': [], 'rewards': [], 'dones': []}
            env.update_pkl_graph(read_pkl(pkl_path))  # 读取一幅图的信息，并重置环境
            external_route_matrix = env.reset()  # 获取
            subgoal, state, next_state = upper_agent.take_action(env.combined_normal_matrix, env.src_lst, env.dst_lst)

            #print("subgoal ：",subgoal, "state : ", state[0, 0], "next_state : ", next_state[0, 0])

            #print("Mt:",env.Mt.shape, env.Mt)
            #print("internal_route_matrix", external_route_matrix)
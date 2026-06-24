#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/1/14 15:45
@File: UnderAC.py
@Decs:*********************
"""
import copy
import sys

import torch

import config
from torch.utils.data import  DataLoader
from net import ACPolicyNet,ACValueNet
from config import *
from replay_buffer import UnderAcReplayBuffer
from parse_config import get_save_path, save_train_parmeter, parse_config
from dataset import get_node_neighbors, get_all_pairs_dijkstra_path_length

from env import *
import torch.nn.functional as F
from UpperAC import UpperActorCritic
from predict.transform.transformer_model import Transformer
from dataset_seq import read_file_to_datasets, prepare_data, normalize_value
import draw_tools

class UnderActorCritic:
    def __init__(self, internal_config: Path = None):
        config = parse_config(internal_config)
        self.policy_lr = float(config['policy_lr'])
        self.value_lr = float(config['value_lr'])

        self.gamma = config['reward_decay']  # 折扣因子
        self.batch_size = config['batch_size']  # 每次从经验池中采样的样本数量

        self.actor = ACPolicyNet().to(DEVICE)  # 策略网络
        self.critic = ACValueNet().to(DEVICE)  # 价值网络

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.value_lr)

        # 定义经验回放池
        self.memory_capacity = int(config['memory_capacity'])
        self.memory_pool = UnderAcReplayBuffer(buffer_size=self.memory_capacity, batch_size=self.batch_size)

        self.neighbors = get_node_neighbors()  # 邻居节点，{node：[neighbors_node]}  在动作选择时选择邻居节点
        self.all_hops = get_all_pairs_dijkstra_path_length()  # 返回两节点之间的最短距离字典{src：{dst：distance，。。。}}
        self.nodes = len(self.neighbors)  # 节点个数

    def take_action(self, state, src_node, dst_node):
        """
        邻居动作选择
        :param state: 当前状态，位置矩阵
        :param src_node: 源节点
        :param dst_node: 目的节点
        :return: 动作，节点的索引：0-nodes
        """
        internal_state_matrix = copy.deepcopy(env.Mt)
        internal_state_matrix[0, 0, :, :] = state
        #print( self.neighbors)
        src_neighbor = self.neighbors.get(src_node)
        #print("src_neighbor = ", src_neighbor)
        src_neighbor = [neighbor_node for neighbor_node in src_neighbor if
                        self.all_hops[src_node][dst_node] > self.all_hops[neighbor_node][dst_node]]

        #print("src_neighbor = ", src_neighbor)
        action_prob = self.actor.forward(internal_state_matrix.to(DEVICE))  # 网络输出
        # # 构造邻居节点矩阵
        # mask = torch.zeros((self.nodes, len(src_neighbor)), dtype=config.TORCH_TYPE, device=DEVICE)
        # for col, idx in enumerate(src_neighbor):
        #     mask[idx - 1, col] = 1
        #
        # act_local = torch.matmul(action_prob, mask) + 1e-8  # 返回邻居节点的 Q值，形状[1, len(src_neighbor)]
        # action_prob = torch.matmul(act_local, mask.T)  # 返回原形状的 邻居节点的 Q值，形状[1, self.nodes]
        # action_dist = torch.distributions.Categorical(action_prob)
        # action = action_dist.sample()  # 0-13
        # return action.item()

        # 构造 mask
        mask = torch.ones_like(action_prob) * (-1e9)

        for idx in src_neighbor:
            mask[0, idx - 1] = 0

        masked_logits = action_prob + mask

        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()

        return action.item()


    def store(self, state, action, reward, next_state, done):
        """
        将一个 (state, action, reward, next_state, done) 元组存储到 经验回放池 (memory_pool) 中，供后续采样和训练
        :param state: 当前状态，状态矩阵
        :param action: 动作
        :param reward: 当前奖励， int(num)
        :param next_state: 下一刻状态,状态矩阵
        :param done: 标志位
        :return: None
        """
        self.memory_pool.add(state, action, reward, next_state, done)

    def _get_batch_vars(self):
        """
        从经验池中随机采样一批数据(小批量样本)，用于训练
        :return:
        """
        state, action, reward, next_state, done = self.memory_pool.sample()
        return state, action, reward, next_state, done

    def _calculate_loss_with_learn(self):
        """
        通过经验回放和计算损失来更新策略/价值网络
        :return: None
        """
        states, actions, rewards, next_states, dones = self._get_batch_vars()
        states_batchs = torch.cat(states).to(DEVICE)
        actions_batchs = torch.tensor(actions).view(-1, 1).to(DEVICE)
        rewards_batchs = torch.tensor(rewards, dtype=TORCH_TYPE).unsqueeze(1).to(DEVICE)  # 调整好形状： torch.Size([50, 1])
        next_states_batchs = torch.cat(next_states).to(DEVICE)
        dones_batchs = torch.tensor(dones, dtype=TORCH_TYPE).unsqueeze(1).to(DEVICE)

        # 1. 计算 TD 误差
        td_target = rewards_batchs + self.gamma * self.critic(next_states_batchs) * (1 - dones_batchs)
        td_delta = td_target - self.critic(states_batchs)

        log_probs = torch.log(self.actor(states_batchs).gather(1, actions_batchs) + 1e-7)
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
    ppo_agent = UpperActorCritic('./config/external_config.yaml')
    ac_agent = UnderActorCritic('./config/internal_config.yaml')
    env = Environment()
    episode_reward = []  # 初始化每个周期的总奖励列表
    episode_average_reward = []  # 初始化每个周期的平均奖励列表
    subgoal_reward_lst = []  # 存每幅图的单 目标奖励
    all_path_info = {}  # 存储节点所有的路径
    save_path = get_save_path()  # 获取存储路径位置

    batch_size = 1
    input_dim = 29
    output_dim = 29
    hidden_dim = 256
    windows = 10
    model = Transformer(n_encoder_inputs=input_dim, n_decoder_inputs=input_dim, output_dim=output_dim).to(DEVICE)
    model.load_state_dict(torch.load('predict/transform/model_weight/2026_05_26_10_43_03/50_transformer_model.pth'))
    DATASET_PATH = PICKLE_PATH  # 数据集的路径  predict\data\pkl\topo21
    # 加载数据集
    datasets = read_file_to_datasets(DATASET_PATH, windows, 0, 450)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    for episode in range(1000):
        ep_reward = 0  # 一个episode的所有图的奖励和
        ep_picture_reward = 0  # 一个episode的所有图的奖励和
        episode_picture_reward = []  # 初始化每个张图的总奖励列表
        #for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=9, n=11, step=1)):  # 前一个(PICKLE_PATH, s=3, n=6, step=1))
        for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=config.picture_start, n=config.picture_end, step=config.picture_step)):  # 前一个(PICKLE_PATH, s=3, n=6, step=1))

            total_reward = 0  # 每幅图src_lst -- > dst_lst的奖励
            total_reward_lst = []  # 每幅图的单目标奖励
            total_path_info = {}  # 一幅图的所有子目标路径
            env.update_pkl_graph(read_pkl(pkl_path))  # 读取一幅图的信息，并重置环境
            external_route_matrix = env.reset()  # 获取
            flag = True  # 每幅图完成的标志，True/False：未完成/完成

            # 预测
            inputs, targets = seq_data = next(data_iter)  # 这就是第一个batch的数据
            input_normalized_data = normalize_value(inputs).squeeze(-1).to(DEVICE)
            targets_normalized_data = normalize_value(targets).squeeze(-1).to(DEVICE)
            outputs = model(input_normalized_data, input_normalized_data)
            outputs = outputs.squeeze(-1).squeeze(0)
            env.update_bw(outputs)

            while(flag):
                # ep_subgoal_task += 1
                subgoal_reward = 0  # 每个subgoal奖励
                subgoal, ppo_state, ppo_next_state = ppo_agent.take_action(env.combined_normal_matrix, env.src_lst, env.dst_lst)

                #print("subgoal ：", subgoal)

                # 映射给下层当前状态，self.src_node, self.dst_node
                env.subgoal_map_inter_state(subgoal)  # 还要映射到self.src_node, self.dst_node
                #print("env.src_node ： ", env.src_node, " ,env.dst_node : ", env.dst_node, " ,env.internal_state_matrix ： ", env.internal_state_matrix)

                for i in range(12):  # 这里限制单条路径长度小于10
                    action = ac_agent.take_action(env.internal_state_matrix, env.new_src_node, env.dst_node)
                    ac_next_state, reward, done, judge_flag, path_info = env.step(action)
                    #print("env_step：", next_state, reward, done, judge_flag, path_info, env.src_node, env.dst_node)
                    next_state_matrix = copy.deepcopy(env.Mt)
                    next_state_matrix[0, 0, :, :] = ac_next_state
                    ac_agent.store(env.Mt, action, reward, next_state_matrix, done)
                    subgoal_reward += reward  # 取单目标的奖励
                    if ac_agent.memory_pool.size() > ac_agent.batch_size:  # AC计算损失
                        ac_agent._calculate_loss_with_learn()

                    if done == 1:  # 完成一个子目标更新状态
                        # 更新源，目的节点列表
                        env.src_lst.append(int(subgoal[-1]) + 1)
                        env.dst_lst.remove(int(subgoal[-1]) + 1)
                        env.combined_normal_matrix[0, 0, :, :][int(subgoal[-1]), int(subgoal[-1])] = 1
                        #print(f"完成子目标，单个子目标奖励为{subgoal_reward}, 更新的源目的节点列表为env.src_lst = {env.src_lst}, env.dst_lst = {env.dst_lst}, 路径为")
                        break

                total_reward_lst.append(subgoal_reward)
                total_path_info.setdefault((subgoal[0]+1, subgoal[1]+1), []).extend(path_info['path'])  # 存规划的路径

                #print(f"存规划的路径subgoal , path_info['path'] = {total_path_info}")
                total_reward += subgoal_reward  # 计算子目标奖励和
                len_a = env.path_info["step_num"]

                if len(env.dst_lst) == 0:
                    print("该图的任务完成")
                    flag = False
                # 一幅图最多20次完成任务
                if env.path_info["step_num"] > 50 and len(env.dst_lst) != 0:
                    print("该图的任务未完成，超过20步，跳出训练")
                    flag = False
                #print(f"完成子一个目标后的路径{env.path_info},源节点为{env.dst_lst},目的节点为{env.src_lst}")

                # upper_ac_reward = total_reward + (config.EXTERNAL_REWARD[0] if done else config.EXTERNAL_REWARD[1]) * total_reward  # 确保反馈给上层的奖励 dqn_reward 保留梯度
                # 3月12好修改
                upper_ac_reward = subgoal_reward + (config.EXTERNAL_REWARD[0] if done else config.EXTERNAL_REWARD[1])

                #print(f"完成子目标，反馈给ppo单个子目标奖励为{ppo_reward}")

                # 这里
                ppo_agent.store(ppo_state, subgoal, upper_ac_reward, ppo_next_state)
                ppo_agent.update()


                # 添加一个
            # 计算一张图的所有子目标奖励
            ep_picture_reward = sum(total_reward_lst) / len(total_reward_lst) if total_reward_lst else 0
            episode_picture_reward.append(ep_picture_reward)
            all_path_info.setdefault(episode, total_path_info)



        #计算每个周期的奖励
        ep_reward = sum(episode_picture_reward) / len(episode_picture_reward) if episode_picture_reward else 0
        ep_average_reward = ep_reward / config.picture_count

        episode_reward.append(ep_reward)
        episode_average_reward.append(ep_average_reward)

        print(f"------------------------------第{episode}回合训练完成，总奖励为：{ep_reward}，平均奖励为：{ep_average_reward}------------------------------")
    # print(f"训练上层的奖励：{episode_reward}, config.picture_count: {config.picture_count}")
    # print(f"训练的平均奖励：{episode_average_reward}")
    # print(f"规划的路径：{all_path_info}")

    draw_tools.draw_and_save(save_path, episode_reward, episode_average_reward, all_path_info, "rewards", "average_reward", "all_paths")

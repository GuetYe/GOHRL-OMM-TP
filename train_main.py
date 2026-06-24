# #!/usr/bin/env python
# # -*- coding:utf-8 -*-
# """
# @Auther:LFY
# @Contact:2545039870@qq.com
# @Time: 2025/3/3 10:40
# @File: train_main.py
# @Decs:*********************
# """
# import copy
# import os
# import utils
# import datetime
# from pathlib import Path
# from env import *
# from UpperAC import PPO
# from UnderAC import ActorCritic
#
#
#
# # 设置打印选项，保证不省略任何元素
# torch.set_printoptions(threshold=float('inf'))+
#
# ppo_agent = PPO('./config/external_config.yaml')
# ac_agent = ActorCritic('./config/internal_config.yaml')
# env = Environment()
# episode_reward = []  # 初始化每个周期的总奖励列表
# episode_average_reward = []  # 初始化每个周期的平均奖励列表
# all_path_info = {}  # 存储节点所有的路径
# picture_count = len(range(3, 50, 3))  # 获取每次训练的图片数量
# for episode in range(500):
#     ep_reward = 0  # 一个episode的所有图的奖励和
#     for index, pkl_path in enumerate(
#             file_path_yield(PICKLE_PATH, s=3, n=50, step=3)):  # 前一个(PICKLE_PATH, s=3, n=6, step=1))
#         total_reward = 0  # 每幅图src_lst -- > dst_lst的奖励
#         # 读取一幅图的信息，并重置环境
#         env.update_pkl_graph(read_pkl(pkl_path))
#         route_matrix, internal_route_matrix = env.reset()
#         print(f"这幅图的源节点列表为：{env.src_lst}, 目的节点列表为：{env.dst_lst}")
#         flag = True  # 每幅图完成的标志，True/False：未完成/完成
#         while (flag):
#             # 目标为节点在矩阵中位置的索引
#             subgoal, state_lst, next_state_lst = ppo_agent.take_action(env.combined_normal_matrix, episode)
#             # print(f"第{index}图的子目标为：{subgoal}")
#             env.subgoal_map_inter_state(subgoal)  # 还要映射到self.src_node, self.dst_node
#             # print("-----------------------------------测试下层---------------------------------")
#             for _ in range(10):
#                 action = ppo_agent.take_action(env.internal_state_matrix)
#                 next_state, reward, done, judge_flag, path_info = env.step(action)
#                 # 将 reward 转换为 PyTorch 张量，并设置 requires_grad=True
#                 # print(f"reward= {reward,}", f"done = {done,}", f"judge_flag = {judge_flag,}", f"path_info = {path_info}")
#                 next_state_matrix = copy.deepcopy(env.Mt)
#                 next_state_matrix[0, 0, :, :] = next_state
#                 ppo_agent.store(env.Mt, action, reward, next_state_matrix, done)
#                 env.internal_state_matrix = copy.deepcopy(next_state)
#
#                 total_reward += reward
#                 all_path_info[path_info['src_dst']] = path_info['path']  # 存规划的路径
#                 if ppo_agent.memory_pool.size() > 64:  # PPO计算损失
#                     ppo_agent._calculate_loss_with_learn()
#                 if done == 1:  # 完成一个子目标更新状态
#                     # print("-----------------------完成一个子目标，更新状态---------------------------")
#                     # 源节点列表{env.src_lst}：1-14， 目的节点列表{env.dst_lst}， 子目标{subgoal}：0-13
#                     env.src_lst.append(int(subgoal[-1]) + 1)
#                     env.dst_lst.remove(int(subgoal[-1]) + 1)
#                     env.combined_normal_matrix[0, 0, :, :][int(subgoal[-1]), int(subgoal[-1])] = 1
#                     # print(f"源节点列表{env.src_lst}， 目的节点列表{env.dst_lst}， 子目标{subgoal}， 位置矩阵{env.combined_normal_matrix[0, 0, : , :]}")
#                     # print(f"len(env.dst_lst == {len(env.dst_lst)}")
#                     break
#             len_a = env.path_info["step_num"]
#             # print(f"第{index}图的子目标为：{subgoal},子目标奖励为：{total_reward},步长为：{len_a},done={done}")
#             if len(env.dst_lst) == 0:
#                 print("该图的任务完成")
#                 flag = False
#             if env.path_info["step_num"] > 90 and len(env.dst_lst) != 0:  # 从60步修改成
#                 print("该图的任务未完成，超过90步，跳出训练")
#                 flag = False
#
#     #         dqn_reward = total_reward + (2 if done else -1) * torch.ones_like(
#     #             total_reward)  # 确保反馈给上层的奖励 dqn_reward 保留梯度
#     #         dqn_agent.store(state_lst, subgoal, dqn_reward, next_state_lst)
#     #         ep_reward += dqn_reward
#     #
#     #         # 3.18修改的DQN学习
#     #         if (episode % 10 == 0) and dqn_agent.memory_pool.size() > 32:  # 64--》 32
#     #             dqn_agent.learn()
#     #
#     # if episode % 1 == 0:
#     #     print(f"{episode} % 1 == 0次的规划的路径：{all_path_info}")
#     # average_reward = ep_reward / picture_count
#     # print(
#     #     f"-----------------------------------------------第{episode}回合训练完成，总奖励为：{ep_reward}，平均奖励为：{average_reward}-----------------------------------------")
#     # episode_reward.append(ep_reward.detach().cpu().numpy())
#     # episode_average_reward.append(average_reward.detach().cpu().numpy())
# # print(f"训练的奖励：{episode_reward}")
# # print(f"训练的平均奖励：{episode_average_reward}")
# # print(f"规划的路径：{all_path_info}")
# #
# # # 获取存储路径
# # current_directory = Path.cwd()
# # mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
# # save_path = os.path.join(current_directory, 'train_data', mkfile_time)
# # # 创建文件夹（exist_ok=True 避免已存在时报错）
# # os.makedirs(save_path, exist_ok=True)
# #
# # draw_tools.draw_and_save(save_path, episode_reward, episode_average_reward, all_path_info, "rewards", "average_reward",
# #                          "all_paths")
# # utils.save_model(dqn_agent.policy_net, save_path, "dqn_policy_net")
# # utils.save_model(dqn_agent.target_net, save_path, "dqn_target_net")
# # # 保存PPO目标网络和价值网络的模型
# # utils.save_model(ppo_agent.actor, save_path, "ppo_actor_net")
# # utils.save_model(ppo_agent.critic, save_path, "ppo_critic_net")
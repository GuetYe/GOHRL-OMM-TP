#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2024/12/18 10:30
@File: parse_config.py
@Decs:*********************
"""
import json
import yaml
import os
import datetime
from pathlib import Path
import config

def parse_config(config_path: str or Path):
    """
    解析yaml文件
    :param config_path: yaml config path, type of pathlib.Path
    :return: EasyDict: a dict of config
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)
    assert Path.exists(config_path), f"{config_path} is not exists"

    with open(config_path, 'r', encoding='utf-8') as file:
        result = yaml.load_all(file.read(), yaml.FullLoader)

        result = list(result)[0]
        print(result)

    assert isinstance(result, dict), f"{result} is not a dict type"
    return result

def get_save_path():
    """
    返回训练数据存储路径
    :return:
    """
    current_directory = Path.cwd()
    mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    save_path = os.path.join(current_directory, 'train_data', 'node21', mkfile_time)
    # 创建文件夹（exist_ok=True 避免已存在时报错）
    os.makedirs(save_path, exist_ok=True)
    return save_path

def save_train_parmeter(save_path, dqn_agent, ppo_agent):
    """
    存储训练的参数
    :param save_path: 存储路径
    :param dqn_agent: dqn参数
    :param ppo_agent: ppo参数
    :return:
    """
    params = {
        "dqn_agent": {
            "lr": dqn_agent.lr,
            "batch_size": dqn_agent.batch_size,
            "memory_capacity": dqn_agent.memory_capacity,
            "update_target_net_frequency": dqn_agent.update_target_net_frequency
        },
        "ppo_agent": {
            "policy_lr": ppo_agent.policy_lr,
            "value_lr": ppo_agent.value_lr,
            "batch_size": ppo_agent.batch_size,
            "memory_capacity": ppo_agent.memory_capacity
        },
        "train_data":{
            "reward_param_weight": config.PARAM_WEIGHT,
            "internal_reward": config.INTERNAL_REWARD,
            "external_reward": config.EXTERNAL_REWARD,
            "episodes": config.episodes,
            "picture_start": config.picture_start,
            "picture_end": config.picture_end,
            "picture_step": config.picture_step,
            "picture_count": config.picture_count
        },
        "train_srd_to_dst":{
            "src_node":config.src_node,
            "dst_lst":config.dst_lst
    }
    }
    # 保存奖励数据到 CSV
    json_file = os.path.join(save_path, "train_parmeter.json")
    with open(json_file, "w") as f:
        json.dump(params, f, indent=4)

if __name__ == '__main__':
    result = parse_config('./config/external_config.yaml')
    print(result)

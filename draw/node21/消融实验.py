#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/4/8 15:40
@File: 消融实验.py
@Decs:*********************
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# --------------------------
# 设置中文字体
# --------------------------
# 1. 自动查找系统可用黑体（SimHei）路径
simhei_path = None
for font in font_manager.findSystemFonts():
    if 'SimHei' in font or 'simhei' in font.lower():
        simhei_path = font
        break

if simhei_path:
    my_font = font_manager.FontProperties(fname=simhei_path)
else:
    # 找不到 SimHei 就使用默认字体
    my_font = None
    print("警告：未找到 SimHei 字体，中文可能无法正常显示。")

# 设置显示负号
plt.rcParams['axes.unicode_minus'] = False

def load_reward(file_path):
    """读取 rewards.csv 并返回一维 numpy 数组"""
    df = pd.read_csv(file_path)
    # 自动识别 reward 列，或只有一列时取第一列
    if 'Reward' in df.columns:
        reward = df['Reward']
    elif df.shape[1] == 1:
        reward = df.iloc[:, 0]
    else:
        raise ValueError(f"{file_path} 中未找到 reward 列，请检查 CSV 格式")
    reward = pd.to_numeric(reward, errors='coerce').dropna()
    return reward.to_numpy()


def draw_compare_reward_line(file1, file2, xyt_label, line_labels=None, smooth_window=1):
    """
    读取两个 rewards.csv 文件并画折线图
    smooth_window: 平滑窗口，1表示不平滑
    """
    # 读取数据
    data1 = load_reward(file1)
    data2 = load_reward(file2)

    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))

    # plt.style.use("seaborn-whitegrid")
    # plt.rcParams['font.family'] = 'Times New Roman'
    #plt.rcParams['axes.unicode_minus'] = False

    #fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 6))

    # 默认折线标签
    if line_labels is None:
        line_labels = ["Method1", "Method2"]

    episodes1 = np.arange(len(data1))
    episodes2 = np.arange(len(data2))

    # 折线颜色
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    # 绘制折线
    ax.plot(episodes1, data1, label=line_labels[0], color=colors[0], linewidth=2)
    ax.plot(episodes2, data2, label=line_labels[1], color=colors[1], linewidth=2)

    # 设置中文字体
    if my_font:
        ax.set_xlabel("Episode", fontsize=12, fontproperties=my_font)
        ax.set_ylabel(xyt_label, fontsize=12, fontproperties=my_font)
        #ax.set_title(f"{xyt_label} Comparison", fontsize=14, fontweight='bold', fontproperties=my_font)
        ax.legend(loc='best', prop=my_font)
    else:
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(xyt_label, fontsize=12)
        #ax.set_title(f"{xyt_label} Comparison", fontsize=14, fontweight='bold')
        ax.legend(loc='best')


    # ax.legend(loc='best', )
    # ax.grid(True, linestyle='--', alpha=0.6)

    #保存图片
    #fig.savefig(f"./picture/{xyt_label}.png", dpi=400)
    #fig.savefig(f"./picture/{xyt_label}.svg", format="svg", dpi=400, bbox_inches="tight")

    plt.show()


# ==========================
# 调用示例
# ==========================
# =========================
# 1. 文件路径
# =========================
file1 = r'./2026_03_14_13_07_36_最优/rewards.csv'
file2 = r'./2026_03_14_15_04_31_nice/rewards.csv'
# Without Transformer-based Traffic Prediction
# With Transformer-based Traffic Prediction
#draw_compare_reward_line(file1, file2, xyt_label="Reward", line_labels=["使用Transformer流量预测机制", "未使用流量预测机制"], smooth_window=20)

draw_compare_reward_line(file1, file2, xyt_label="Reward", line_labels=["Without Transformer-based Traffic Prediction",
                                                                        "With Transformer-based Traffic Prediction"], smooth_window=20)


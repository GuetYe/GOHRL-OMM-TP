#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/4/8 13:42
@File: draw.py
@Decs:*********************
"""
import numpy as np
import matplotlib.pyplot as plt


def draw_line_chat(x_data, param1_data, param2_data, param3_data, param4_data, save_path):
    """
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data、parameter3_data  中
    # 假设还有 x 轴上的数据，存储在 x_data 中

    LSTM Predicted Value     /     GRU Predicted Value     /     Transformer Predicted Value


    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.figure(facecolor='white')  # 设置整个图形的背景色
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置坐标轴区域的背景色

    confidence_interval = 0.01  # 置信水平
    upper_bound = []
    lower_bound = []
    # 计算置信区间的上限和下限   B22222
    for i in range(len(param2_data)):
        upper_bound.append(param2_data[i] + confidence_interval)
        lower_bound.append(param2_data[i] - confidence_interval)
    # 绘制折线图
    # 我的
    #bar_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    line1, = plt.plot(x_data, param1_data, linestyle='-', linewidth=1, color='#d62728', label='Real Value')

    line2, = plt.plot(x_data, param2_data, linestyle='-', linewidth=1, color='#1f77b4',
                      label='LSTM Predicted Value')

    line3, = plt.plot(x_data, param3_data, linestyle='-', linewidth=1, color='#2ca02c',
                      label='GRU Predicted Value')

    line4, = plt.plot(x_data, param4_data, linestyle='-', linewidth=1, color='#9467bd',
                      label='Transformer Predicted Value')

    # plt.fill_between(x_data, lower_bound, upper_bound, color='c', alpha=0.3, label='90% Confidence Interval')

    # # 设置Y轴范围
    # plt.ylim(0, 30)

    # 添加标签和标题
    plt.xlabel('时间')
    plt.ylabel('网络流量预测值 Mbps')

    # 添加图例，并设置图例标记样式
    plt.legend()

    # 添加网格线
    # plt.grid(True)

    # 保存信息
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.2)
    # 显示图形
    plt.show()


def draw_error_line_chat(x_data, param2_data, param3_data, param4_data, error_type, save_path):
    """
    # 假设有两个参数的数据，分别存储在 parameter1_data 和 parameter2_data、parameter3_data  中
    # 假设还有 x 轴上的数据，存储在 x_data 中

    LSTM Predicted Value     /     GRU Predicted Value     /     Transformer Predicted Value


    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.figure(facecolor='white')  # 设置整个图形的背景色
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置坐标轴区域的背景色

    confidence_interval = 0.01  # 置信水平
    upper_bound = []
    lower_bound = []
    # 计算置信区间的上限和下限
    for i in range(len(param2_data)):
        upper_bound.append(param2_data[i] + confidence_interval)
        lower_bound.append(param2_data[i] - confidence_interval)
    # 绘制折线图
    # 我的
    # bar_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    line2, = plt.plot(x_data, param2_data, linestyle='-', linewidth=1, color='#d62728',
                      label='LSTM Predicted Value')

    line3, = plt.plot(x_data, param3_data, linestyle='-', linewidth=1, color='#1f77b4',
                      label='GRU Predicted Value')

    line4, = plt.plot(x_data, param4_data, linestyle='-', linewidth=1, color='#2ca02c',
                      label='Transformer Predicted Value')

    # plt.fill_between(x_data, lower_bound, upper_bound, color='c', alpha=0.3, label='90% Confidence Interval')

    # 添加标签和标题
    plt.xlabel('时间')
    plt.ylabel(error_type)

    # 添加图例，并设置图例标记样式
    plt.legend()

    # 添加网格线
    # plt.grid(True)

    # 保存信息
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.2)
    # 显示图形
    plt.show()


def draw_bar_chat(y1, y2, y3, save_path):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.figure(facecolor='white')  # 设置整个图形的背景色
    ax = plt.gca()
    ax.set_facecolor('white')  # 设置坐标轴区域的背景色

    # 柱形的宽度
    bar_width = 0.3
    x = np.arange(3)
    # bar_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    bars1 = plt.bar(x, y1, width=bar_width, color='#2ca02c', label='Transformer')
    bars2 = plt.bar(x + bar_width, y2, width=bar_width, color='#d62728', label='LSTM')
    bars3 = plt.bar(x + bar_width * 2, y3, width=bar_width, color='#1f77b4', label='GRU')

    # 柱子上添加数值
    plt.text(x[0], round(y1[0], 3), round(y1[0], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[1], round(y1[1], 3), round(y1[1], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[2], round(y1[2], 3), round(y1[2], 3), ha='center', va='bottom', fontsize=10, zorder=10)

    plt.text(x[0] + bar_width, round(y2[0], 3), round(y2[0], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[1] + bar_width, round(y2[1], 3), round(y2[1], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[2] + bar_width, round(y2[2], 3), round(y2[2], 3), ha='center', va='bottom', fontsize=10, zorder=10)

    plt.text(x[0] + bar_width * 2, round(y3[0], 3), round(y3[0], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[1] + bar_width * 2, round(y3[1], 3), round(y3[1], 3), ha='center', va='bottom', fontsize=10, zorder=10)
    plt.text(x[2] + bar_width * 2, round(y3[2], 3), round(y3[2], 3), ha='center', va='bottom', fontsize=10, zorder=10)

    # 设置x轴标签
    tick_label = ['MAE', 'MSE', 'MAPE']
    plt.xticks(x + bar_width, tick_label)  # 设置X轴标签位置和文本
    # 去除X轴刻度线
    plt.tick_params(axis='x', which='both', length=0)

    # 添加标签和标题
    plt.xlabel('指标')
    plt.ylabel('指标值')

    # 添加图例，并设置图例标记样式
    plt.legend()

    # 保存信息
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.2)
    plt.show()



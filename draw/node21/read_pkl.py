#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2025/6/8 22:41
@File: read_pkl.py
@Decs:*********************
"""
import os
import json
import pickle
import sys

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xml.etree.ElementTree as ET  # 解析xml树形结构
import networkx as nx
from pathlib import Path


font_cn = font_manager.FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)
font_en = font_manager.FontProperties(fname="C:/Windows/Fonts/times.ttf", size=12)
plt.rcParams["axes.unicode_minus"] = False     # 解决负号显示问题

def parse_topo_links_info():
    """ 解析topo生成的xml，为设置env的state和action """
    my_graph = nx.Graph()
    parser = ET.parse(LINKS_INFO)
    root = parser.getroot()

    def _str_tuple2int_list(s: str):
        s = s.strip()
        assert s.startswith('(') and s.endswith(')')
        _s = s[1:-1].split(', ')
        # print(_s)
        return [int(i) for i in _s]

    node1, node2, port1, port2, free_bw, delay, loss, used_bw, pkt_err, pkt_drop, distance = None, None, None, \
                                                                                             None, None, None, \
                                                                                             None, None, None, \
                                                                                             None, None
    for child in root.iter():
        if child.tag == 'links':
            node1, node2 = _str_tuple2int_list(child.text)
            # print(node1, node2)
        elif child.tag == 'ports':
            port1, port2 = _str_tuple2int_list(child.text)
        elif child.tag == 'free_bw':
            free_bw = float(child.text)
        elif child.tag == 'delay':
            delay = float(child.text[0: -2])
        elif child.tag == 'loss':
            loss = float(child.text)
        elif child.tag == 'used_bw':
            used_bw = float(child.text)
        elif child.tag == 'pkt_err':
            pkt_err = float(child.text)
        elif child.tag == 'pkt_drop':
            pkt_drop = float(child.text)
        elif child.tag == 'distance':
            distance = float(child.text)
        else:
            continue
        # print(node1, node2, port1, port2, free_bw, delay, loss, used_bw, pkt_err, pkt_drop, distance)
        my_graph.add_edge(node1, node2, port1=port1, port2=port2, free_bw=free_bw, delay=delay, loss=loss,
                          used_bw=used_bw, pkt_err=pkt_err, pkt_drop=pkt_drop, distance=distance)

    return my_graph


def get_all_pairs_dijkstra_path_length():
    """ 获取两节点直接的最短距离（跳数）"""
    my_graph = parse_topo_links_info()
    # all_hops = dict(nx.all_pairs_shortest_path_length(my_graph))
    all_hops = nx.shortest_path(my_graph, source=1, target=19)
    #all_hops = dict(nx.all_pairs_shortest_path(my_graph))
    #all_hops = dict(nx.single_source_shortest_path(my_graph, 2))

    print("all_hops：", all_hops)
    return all_hops


def get_node_neighbors():
    """ 获取节点的邻居节点 """
    my_graph = parse_topo_links_info()
    nodes = sorted(my_graph.nodes())
    neighbors = {}  # 存邻居节点
    # print(nodes)
    for node in nodes:
        neighbors[node] = list(my_graph.neighbors(node))
    return neighbors


def file_path_yield(file_dir, s=10, n=500, step=1):
    """
    按序号读取保存数据的文件
    :param file_dir: 数据目录
    :param s: 开始文件index
    :param n: 结束文件index
    :param step: 读取步长
    :return: 返回选择的pkl文件路径
    """
    _dir = os.listdir(file_dir)
    # print("_dir = ", _dir)
    assert n <= len(_dir), "n should small than len(_dir)"  # 判断是否超出数据文件总数
    file_names = sorted(_dir, key=lambda x: int(x.split('-')[0]))  # 按文件序号排序
    for name in file_names[s:n:step]:
        print(file_dir / name)
        yield file_dir / name


def read_pkl(pickle_path):
    """ 读取pickle并转化为graph """
    pkl_graph = nx.read_gpickle(pickle_path)
    # print(pkl_graph.edges.data())
    # # 初始化绘图
    #nx.draw(pkl_graph, with_labels=True)
    #plt.savefig(f"./picture_tijiao/node13.svg", format="svg", dpi=400, bbox_inches="tight")
    #plt.savefig(f"./picture/node14.png", dpi=400)
    #plt.show()
    return pkl_graph


def get_data(tree_path):
    date_lst = []
    for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=0, n=48, step=1)):  # s=5, n=53, step=4)
        data = {}
        pkl_graph = read_pkl(pkl_path)
        for key in tree_path.keys():
            value_len = len(tree_path.get(key)) - 1
            bw, delay, loss = [], [], []

            # print("key = ", key, "，value = ", tree_path.get(key), ", value_lst_len = ",  value_len)
            for index in range(value_len):
                src, dst = tree_path.get(key)[index], tree_path.get(key)[index + 1]
                edge_data = pkl_graph[src][dst]
                bw.append(edge_data["free_bw"])
                delay.append(edge_data["delay"])
                loss.append(edge_data["loss"])
            total_loss_rate = 1 - math.prod([1 - p for p in loss])  # 计算链路丢包率
            # print("每条路径的参数：", bw, delay, total_loss_rate)
            # print("参数的最大或最小值：", min(bw), max(delay), max(loss))

            data.setdefault(key)
            data[key] = {"bw": min(bw), "delay": sum(delay), "loss": total_loss_rate}

            bw.clear(), delay.clear(), loss.clear()
        # print("---------------data-----------------", data)

        # 分别提取三个指标
        bw_values = [v['bw'] for v in data.values()]
        delay_values = [v['delay'] for v in data.values()]
        loss_values = [v['loss'] for v in data.values()]

        # 计算平均值
        avg_bw = sum(bw_values) / len(bw_values)
        avg_delay = sum(delay_values) / len(delay_values)
        avg_loss = sum(loss_values) / len(loss_values)

        # 打印结果
        # print("------------------===！===---------------------")
        # print("平均带宽 (bw):", avg_bw)
        # print("平均时延 (delay):", avg_delay)
        # print("平均丢包率 (loss):", avg_loss)
        # print("------------------======---------------------")
        data.clear()

        date_lst.append({"bw": avg_bw, "delay": avg_delay, "loss": avg_loss})
        data.clear()
    #print(date_lst, len(date_lst))
    return date_lst


def draw_compare_exp_bar(tree_path_data, short_path_data, xyt_label):
    # 设置图像风格和字体
    plt.style.use("seaborn-whitegrid")  # 更干净的底图风格
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 主动创建 figure，使用 constrained_layout
    fig, ax = plt.subplots(constrained_layout=True)

    # 数据与参数设置
    data_groups = [tree_path_data, short_path_data]
    # 我的
    bar_labels = ["GOHRL-OMM-TP", "OSPF", "AC", "PPO"]
    bar_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    # cyy
    num_methods = len(data_groups)
    num_bars = len(tree_path_data)

    x = np.arange(num_bars)
    total_width = 0.8
    bar_width = total_width / num_methods
    # 调整x的位置，使柱子居中
    offset = (total_width - bar_width) / 2
    x_base = x - offset

    # 绘制每组柱子
    for i, data in enumerate(data_groups):
        ax.bar(x_base + i * bar_width, data,  # plt.bar(x_base + i * bar_width, data,
               width=bar_width, label=bar_labels[i], color=bar_colors[i])

    # x轴标签设置
    xtick_labels = ['1:00', '3:00', '5:00', '7:00', '9:00', '11:00', '13:00',
                    '15:00', '17:00', '19:00', '21:00', '23:00']
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels[:num_bars], rotation=0, ha='right', fontsize=10)
    #ax.set_yticklabels(ax.get_yticks(), fontsize=12)  # 纵轴刻度
    ax.set_xlabel("Time", fontproperties=font_en, fontsize=12)
    ax.set_ylabel(xyt_label, fontproperties=font_cn, fontsize=12)

    # 设置标题
    #ax.set_title("覆盖组播路径性能对比", fontsize=14, fontweight='bold')

    ax.legend(loc='best', prop=font_en)
    ax.grid(True, linestyle='--', alpha=0.6)
    # 如果数据差异过大，使用对数坐标


    #保存并显示
    fig.savefig(f"./picture/{xyt_label}.png", dpi=400)

    # fig.savefig("./picture_tijiao/瓶颈带宽.svg", format="svg", dpi=400, bbox_inches="tight")
    # fig.savefig(f"./picture_tijiao/{xyt_label}.svg", format = "svg", dpi = 400, bbox_inches = "tight")
    plt.show()

def group_average(data_list, group_size=2):
    """
    将列表按照指定大小分组，计算每组的平均值

    参数:
    data_list: 输入的数据列表
    group_size: 分组大小，默认为4

    返回:
    包含每组平均值的新列表
    """
    averaged_list = []
    for i in range(0, len(data_list), group_size):
        group = data_list[i:i + group_size]
        average = sum(group) / len(group)
        averaged_list.append(average)

    return averaged_list

def get_path_data(path_data):
    bw_list = [d['bw'] for d in path_data]
    delay_list = [d['delay'] for d in path_data]
    loss_list = [d['loss'] for d in path_data]

    tree_bw_list = group_average(bw_list, group_size=4)
    tree_delay_list = group_average(delay_list, group_size=4)
    tree_loss_list = group_average(loss_list, group_size=4)

    return tree_bw_list, tree_delay_list, tree_loss_list

# def draw_compare_exp_bar(tree_path_data, short_path_data, ac_path_data, ppo_path_data, xyt_label):
#     # 设置图像风格和字体
#     plt.style.use("seaborn-whitegrid")  # 更干净的底图风格
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#     size = 12
#     x = np.arange(size)
#
#     # 有a/b两种类型的数据，n设置为2·
#     total_width, n = 0.8, 4
#     # 每种类型的柱状图宽度，n为柱子的数量
#     width = total_width / n
#     bar_color = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
#     bar_label = ["GOHRL-OM", "OSPF", "AC", "PPO"]
#
#
#     # 重新设置x轴的坐标
#     x = x - (total_width - width) / n
#     # print(x)
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     # 画柱状图
#     plt.bar(x, tree_path_data, width=width, label=bar_label[0], color=bar_color[0])
#     plt.bar(x + width, short_path_data, width=width, label=bar_label[1], color=bar_color[1])
#     plt.bar(x + 2 * width, ac_path_data, width=width, label=bar_label[2], color=bar_color[2])
#     plt.bar(x + 3 * width, ppo_path_data, width=width, label=bar_label[3], color=bar_color[3])
#     plt.xticks(np.arange(size), ('1:00', '3:00', '5:00', '7:00', '9:00', '11:00', '13:00',
#                                  '15:00', '17:00', '19:00', '21:00', '23:00'))
#     # 显示图例
#     plt.legend(loc='best')
#     plt.xlabel("time")
#     plt.ylabel(xyt_label)
#     plt.savefig(f"./picture/{xyt_label}.png", dpi=400)
#     # 显示柱状图
#     plt.show()

# def get_path_data(path_data):
#     bw_list = [d['bw'] for d in path_data]
#     delay_list = [d['delay'] for d in path_data]
#     loss_list = [d['loss'] for d in path_data]
#     return bw_list, delay_list, loss_list


if __name__ == '__main__':
    WORK_DIR = Path.cwd().parent
    LINKS_INFO = WORK_DIR / "DataSet/21/links_info.xml"  # 链路信息的xml文件路径
    PICKLE_PATH = WORK_DIR / "DataSet/21/this"  # 数据路径 weight_pickle

    tree_path = {  # 我的结果
        (1, 11): [1, 4, 13, 11],
        (1, 16): [1, 5, 7, 16],
        (11, 19): [11, 13, 12, 14, 17, 19]
    }

    shor_path = {
        (1, 11):  [1, 3, 2, 11],
        (1, 16):  [1, 5, 7, 16],
        (1, 19): [1, 5, 7, 15, 19]
    }

    # get_data(tree_path)
    # get_all_pairs_dijkstra_path_length()

    print("----------------------tree_path--------------------------")
    tree_path_data = get_data(tree_path)
    tree_bw_list, tree_delay_list, tree_loss_list = get_path_data(tree_path_data)

    print(tree_bw_list, tree_delay_list, tree_loss_list)

    print("----------------------short_path--------------------------")
    short_path_data = get_data(shor_path)
    shor_bw_list, shor_delay_list, shor_loss_list = get_path_data(short_path_data)
    print(shor_bw_list, shor_delay_list, shor_loss_list)

    #xyt_label = ["瓶颈带宽mb/s", "链路时延ms", "链路丢包率%"]
    #xyt_label = ["瓶颈带宽(Mbits)", "时延(ms)", "丢包率(%)"]
    xyt_label = ["bandwith(Mbits)", "delay(ms)", "loss(%)"]
    draw_compare_exp_bar(tree_bw_list, shor_bw_list, xyt_label[0])
    draw_compare_exp_bar(tree_delay_list, shor_delay_list, xyt_label[1])
    draw_compare_exp_bar(tree_loss_list, shor_loss_list, xyt_label[2])

    resault = {}
    resault.setdefault("bw")
    resault.setdefault("delay")
    resault.setdefault("loss")
    resault["bw"] = [(sum(tree_bw_list) - sum(shor_bw_list)) / sum(shor_bw_list)]
    resault["delay"] = [(sum(shor_delay_list) - sum(tree_delay_list)) / sum(tree_delay_list)]
    resault["loss"] = [(sum(shor_loss_list) - sum(tree_loss_list)) / sum(tree_loss_list)]
    with open("./picture/test_resault.json", "w") as f:
        json.dump(resault, f)

    print(sum(tree_bw_list), sum(shor_bw_list))
    print(sum(tree_delay_list), sum(shor_delay_list))
    print(sum(tree_loss_list), sum(shor_loss_list))

    # 三个字典，分别存 bw、delay、loss
    bw_data = {
        "tree": tree_bw_list,
        "shor": shor_bw_list,
    }

    delay_data = {
        "tree": tree_delay_list,
        "shor": shor_delay_list,
    }

    loss_data = {
        "tree": tree_loss_list,
        "shor": shor_loss_list,
    }

    with open("./test_picture/bw.json", "w") as f:
        json.dump(bw_data, f, indent=4)

    with open("./test_picture/delay.png.json", "w") as f:
        json.dump(delay_data, f, indent=4)

    with open("./test_picture/loss.json", "w") as f:
        json.dump(loss_data, f, indent=4)

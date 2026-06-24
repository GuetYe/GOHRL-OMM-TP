#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2024/12/05 20:10:35
@File: dataset.py
@Decs:*********************
"""
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # 解析xml树形结构
import networkx as nx
from config import LINKS_INFO, PICKLE_PATH


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
    """
    获取两节点直接的最短距离（跳数）
        {   1: {1: 0, 11: 1, 3: 1, 4: 1, 5: 1, 2: 2, 6: 2, 7: 2, 8: 2, 9: 2, 13: 2, 14: 2, 10: 3, 12: 3},
            3: {3: 0, 1: 1, 11: 2, 4: 2, 5: 2, 2: 3, 6: 3, 7: 3, 8: 3, 9: 3, 13: 3, 14: 3, 10: 4, 12: 4},
            4: {4: 0, 1: 1, 2: 1, 5: 1, 6: 1, 9: 1, 11: 1, 13: 1, 3: 2, 7: 2, 8: 2, 10: 2, 12: 2, 14: 2},
            5: {5: 0, 1: 1, 2: 1, 4: 1, 6: 1, 7: 1, 8: 1, 3: 2, 9: 2, 11: 2, 13: 2, 14: 2, 10: 3, 12: 3},
            11: {11: 0, 1: 1, 4: 1, 14: 1, 2: 2, 3: 2, 5: 2, 6: 2, 8: 2, 9: 2, 12: 2, 13: 2, 7: 3, 10: 3},
            2: {2: 0, 4: 1, 5: 1, 6: 1, 1: 2, 7: 2, 8: 2, 9: 2, 11: 2, 13: 2, 3: 3, 10: 3, 12: 3, 14: 3},
            6: {6: 0, 2: 1, 4: 1, 5: 1, 1: 2, 7: 2, 8: 2, 9: 2, 11: 2, 13: 2, 3: 3, 10: 3, 12: 3, 14: 3},
            9: {9: 0, 4: 1, 7: 1, 8: 1, 10: 1, 13: 1, 1: 2, 2: 2, 5: 2, 6: 2, 11: 2, 12: 2, 14: 2, 3: 3},
            13: {13: 0, 9: 1, 10: 1, 4: 1, 12: 1, 1: 2, 2: 2, 5: 2, 6: 2, 7: 2, 8: 2, 11: 2, 14: 2, 3: 3},
            7: {7: 0, 9: 1, 5: 1, 1: 2, 2: 2, 4: 2, 6: 2, 8: 2, 10: 2, 13: 2, 3: 3, 11: 3, 12: 3, 14: 3},
            8: {8: 0, 9: 1, 5: 1, 14: 1, 1: 2, 2: 2, 4: 2, 6: 2, 7: 2, 10: 2, 11: 2, 12: 2, 13: 2, 3: 3},
            14: {14: 0, 8: 1, 11: 1, 12: 1, 1: 2, 4: 2, 5: 2, 9: 2, 13: 2, 2: 3, 3: 3, 6: 3, 7: 3, 10: 3},
            10: {10: 0, 9: 1, 13: 1, 4: 2, 7: 2, 8: 2, 12: 2, 1: 3, 2: 3, 5: 3, 6: 3, 11: 3, 14: 3, 3: 4},
            12: {12: 0, 13: 1, 14: 1, 4: 2, 8: 2, 9: 2, 10: 2, 11: 2, 1: 3, 2: 3, 5: 3, 6: 3, 7: 3, 3: 4}}
    """
    my_graph = parse_topo_links_info()
    all_hops = dict(nx.all_pairs_shortest_path_length(my_graph))
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

    assert n < len(_dir), "n should small than len(_dir)"  # 判断是否超出数据文件总数
    file_names = sorted(_dir, key=lambda x: int(x.split('-')[0]))  # 按文件序号排序
    for name in file_names[s:n:step]:
        print(file_dir / name)
        yield file_dir / name


def read_pkl(pickle_path):
    """ 读取pickle并转化为graph """
    pkl_graph = nx.read_gpickle(pickle_path)
    #print(pkl_graph.edges.data())
    # # 初始化绘图
    #nx.draw(pkl_graph, with_labels=True)
    #plt.show()
    return pkl_graph


def get_pkl_data():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)

    # 拼接文件路径
    pkl_file_path = os.path.join(script_dir, 'DataSet', 'weight_pickle14', '1-2022-11-17-15-41-19.pkl')

    # 检查文件是否存在
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            print(type(data))
            # 如果是 networkx 图，显示节点、边和属性
            if hasattr(data, 'nodes') and hasattr(data, 'edges'):
                print("\n节点及其属性:")
                for node, attr in data.nodes(data=True):
                    print(f"节点 {node}: {attr}")

                print("\n边及其权重:")
                for u, v, weight in data.edges(data=True):
                    print(f"边 ({u}, {v}): {weight}")
            else:
                print("该数据不是一个 networkx 图，内容如下：")
                print(data)
    else:
        print("文件不存在或路径错误")


if __name__ == '__main__':
    # 获取脚本所在目录的绝对路径
    # 拼接文件路径
    # pkl_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataSet', 'weight_pickle14')
    # print(pkl_file_path)
    # file_path_yield(PICKLE_PATH)
    # parse_topo_links_info()
    # get_node_neighbors()

    # ----------------------------------------------------------
    # # 获取脚本所在目录的绝对路径
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # print(script_dir)
    # #
    # # # 拼接文件路径
    # pkl_file_path = os.path.join(script_dir, 'DataSet', 'node14', 'weight_pickle', '11-2022-11-17-15-42-59.pkl')
    # read_pkl(pkl_file_path)
    #
    # print("--------------------------------------------------------------------------------")
    # graph = parse_topo_links_info()
    # neighbors = get_node_neighbors()
    # for index, pkl_path in enumerate(file_path_yield(PICKLE_PATH, s=3, n=53, step=1)):
    #     read_pkl(pkl_path)
    get_all_pairs_dijkstra_path_length()
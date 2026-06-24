from pathlib import Path

d_model = 8  # 模型中的隐藏状态将是8维向量
nhead = 8  # 该模型在多头注意力机制中使用了8个头。
num_layers = 8  # 变压器模型由16个相同的层组成。

# 数据集地址以及topo信息的地址
# WORK_DIR = Path.cwd()
WORK_DIR = Path.cwd().parent.parent  # 获取当前路径

DATASET_PATH = WORK_DIR / 'DataSet/pkl/topo21'  # 数据集的路径  predict\data\pkl\topo21

# -----------------从数据集中加载一些参数----------------------#
# dataset = DataSet(XML_TOPOLOGIES_PATH, DATASET_PATH)
# dataset.set_init_topology()

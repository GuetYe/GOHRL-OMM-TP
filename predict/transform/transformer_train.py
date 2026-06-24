import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *
from gpu import gpu
from dataset import file_path_yield, read_pickle, prepare_data, read_file_to_datasets
from transformer_model import Transformer
from config import DATASET_PATH

import time
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
import sys

# 初始化模型
batch_size = 1
input_dim = 29
output_dim = 29
hidden_dim = 256
windows = 10
model = Transformer(n_encoder_inputs=input_dim, n_decoder_inputs=input_dim, output_dim=output_dim).to(gpu())
# 加载数据集
datasets = read_file_to_datasets(DATASET_PATH,windows,0 ,300)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
pre_datasets = read_file_to_datasets(DATASET_PATH,windows,350 ,450)
dataloader_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 存放损失函数与精确度的路径
loss_graph_path, accuracy_graph_path, model_weight_path = create_graph_path()

# 训练模型
def train():
    # 训练模型
    loss_list = []
    epochs_lsit = []
    epochs = 50
    num = 1
    for epoch in range(1, epochs + 1):
        for index, seq_data in enumerate(dataloader):
            inputs, targets = seq_data  # 将每个批次的数据拆分为输入（inputs）和目标（targets）
            # 准备数据
            # 进行归一化处理
            input_normalized_data = normalize_value(inputs).squeeze(-1).to(gpu())
            targets_normalized_data = normalize_value(targets).squeeze(-1).to(gpu())
            optimizer.zero_grad()
            # 用模型进行预测
            outputs = model(input_normalized_data, input_normalized_data)
            # 计算损失
            loss = criterion(outputs, targets_normalized_data)
            loss.backward()
            optimizer.step()
            epochs_lsit.append(num)
            num = num +1 
            loss_list.append(loss.item())
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        torch.save(model.state_dict(), '{}/{}_transformer_model.pth'.format(model_weight_path, epoch))  # 保存模型权重
    plot_episode_data(loss_graph_path, epochs_lsit[0:1000], loss_list[0:1000], "episode", "loss", "loss")


if __name__ == "__main__":
    train()
    # pass

import torch
import torch.nn as nn
import torch.optim as optim

import config
import numpy as np
import torch.nn as nn
from utils import *
from gpu import gpu
from torch.utils.data import  DataLoader
from lstm_model import LSTMModel
from dataset import file_path_yield, read_pickle, prepare_data, read_file_to_datasets
from config import DATASET_PATH

import time
local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
# 存放损失函数与精确度的路径
loss_graph_path, accuracy_graph_path, model_weight_path = create_graph_path()


# 初始化模型
batch_size = 1
input_dim = 29
output_dim = 29
hidden_dim = 256
windows = 10
model = LSTMModel(input_dim, output_dim).to(gpu())
datasets = read_file_to_datasets(DATASET_PATH,windows,0 ,700)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

pre_datasets = read_file_to_datasets(DATASET_PATH,windows,680 ,940)
pe_dataloader = DataLoader(pre_datasets, batch_size=batch_size, shuffle=False)


# 验预测的数据
index_list = []
# 存放真实的数据
true_values = []
# 存放预测的数据
predicted_values = []
# 均方损失
elementwise_loss = ''
# 训练模型
def train():
    # 训练模型
    loss_list = []
    epochs_lsit = []
    epochs = 50
    num = 1
    for epoch in range(1, epochs + 1):
        for index, seq_data in enumerate(dataloader):
            inputs, targets = seq_data
            # 进行归一化处理
            input_normalized_data = normalize_value(inputs).squeeze(-1).to(gpu())
            targets_normalized_data = normalize_value(targets).squeeze(-1).to(gpu())
            optimizer.zero_grad()
            outputs = model(input_normalized_data)
            loss = criterion(outputs, targets_normalized_data)
            loss.backward()
            optimizer.step()
            epochs_lsit.append(num)
            num = num +1 
            loss_list.append(loss.item())
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')
        # if epoch % 10 == 0:
        torch.save(model.state_dict(), '{}/{}_transformer_model.pth'.format(model_weight_path, epoch))  # 保存模型权重
    plot_episode_data(loss_graph_path, epochs_lsit[0:1000], loss_list[0:1000], "episode", "loss", "loss")


def predict():
    index_list = []
    # 存放真实的数据
    true_values = []
    # 存放预测的数据
    predicted_values = []
    # 均方损失
    elementwise_loss = ''
    for index, seq_data in enumerate(pe_dataloader):
        inputs, targets = seq_data
        # 进行归一化处理
        input_normalized_data = normalize_value(inputs).squeeze(-1).to(gpu())
        targets_normalized_data = normalize_value(targets).squeeze(-1).to(gpu())
        min_vals = torch.min(targets)
        max_vals = torch.max(targets)
        with torch.no_grad():
            outputs = model(input_normalized_data)
        _predicted_values = unnormalize_value(outputs, min_vals, max_vals) # 反归一化的真实值

       # 将真实值和预测值存起来
        true_values.append(torch.sum(targets))
        predicted_values.append(torch.sum(_predicted_values))
        index_list.append(index)    
        # 均方损失
        true_values_tensor = torch.stack(true_values, dim=0).cpu()
        predicted_values_tensor = torch.stack(predicted_values, dim=0).cpu()
        elementwise_loss_tensor = (true_values_tensor - predicted_values_tensor) ** 2
        elementwise_loss = elementwise_loss_tensor.tolist()
    return index_list, true_values, predicted_values, elementwise_loss
         
        

if __name__ == "__main__":
    train()



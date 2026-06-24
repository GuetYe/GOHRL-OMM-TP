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

pre_datasets = read_file_to_datasets(DATASET_PATH,windows,690 ,940)
pe_dataloader = DataLoader(pre_datasets, batch_size=batch_size, shuffle=False)


def predict():
    index_list = []
    # 存放真实的数据
    true_values = []
    # 存放预测的数据
    predicted_values = []

    for index, seq_data in enumerate(pe_dataloader):
        inputs, targets = seq_data
        # 进行归一化处理
        input_normalized_data = normalize_value(inputs).squeeze(-1).to(gpu())
        targets_normalized_data = normalize_value(targets).squeeze(-1).to(gpu())
        min_vals = torch.min(targets)
        max_vals = torch.max(targets)
        with torch.no_grad():
            outputs = model(input_normalized_data)

        # _predicted_values = unnormalize_value(outputs, min_vals, max_vals) # 反归一化的真实值
        _predicted_values = outputs

        # 将真实值和预测值存起来
        # true_values.append(torch.sum(targets))
        true_values.append(torch.sum(targets_normalized_data))

        predicted_values.append(torch.sum(_predicted_values))
        index_list.append(index)    

    return index_list, true_values, predicted_values
         
        

if __name__ == "__main__":
    # train()



    model.load_state_dict(torch.load('./model_weight/2026_01_29_13_11_24/50_transformer_model.pth'))
    model.eval()  # Set the model to evaluation mode    
    index_list, true_values, predicted_values = predict()
        # 将数据存储到文件当中
    with open("./lfy_result.txt", "w+", encoding="utf-8") as file:
        for item1, item2, item3 in zip(index_list, true_values, predicted_values):
            file.write(f"{item1}\t{item2}\t{item3}\n")

    print(f"Lists have been written to file_path")
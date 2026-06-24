import sys

from transformer_model import Transformer
import torch
from torch.utils.data import  DataLoader
import numpy as np
import config
from utils import normalize_value,unnormalize_value
from gpu import gpu
from dataset import file_path_yield, read_pickle, prepare_data, read_file_to_datasets
from config import DATASET_PATH


# 初始化模型
batch_size = 1
input_dim = 29
output_dim = 29
hidden_dim = 256
windows = 10

# 加载数据集
# 训练集
datasets = read_file_to_datasets(DATASET_PATH,windows,0 ,300)
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
# 测试集
pre_datasets = read_file_to_datasets(DATASET_PATH,windows,310 ,450)
pe_dataloader = DataLoader(pre_datasets, batch_size=batch_size, shuffle=False)




# 加载模型
model = Transformer(n_encoder_inputs=input_dim, n_decoder_inputs=input_dim, output_dim=output_dim).to(gpu())
# model.load_state_dict(torch.load(
#     r'D:\Desktop\Overlay-Multicast-predict-RL\predict\transform\model_weight\2025_03_12_21_29_57\50_transformer_model.pth'
# ))
model.load_state_dict(torch.load\
                              ('D:/Desktop/Overlay-Multicast-predict-RL/predict/transform/model_weight/2026_04_14_10_36_01/50_transformer_model.pth'))
model.eval()  # Set the model to evaluation mode


# 输入测试的数据
def predict():
    index_list = []
    # 存放真实的数据
    true_values = []
    # 存放预测的数据
    predicted_values = []
    for index, seq_data in enumerate(pe_dataloader):
        inputs, targets = seq_data
        # 准备数据
        # 进行归一化处理
        input_normalized_data = normalize_value(inputs).squeeze(-1).to(gpu())
        targets_normalized_data = normalize_value(targets).squeeze(-1).to(gpu())
        min_vals = torch.min(targets)
        max_vals = torch.max(targets)
        with torch.no_grad():
            outputs = model(input_normalized_data, input_normalized_data)

        # _predicted_values = unnormalize_value(outputs, min_vals, max_vals) # 反归一化的真实值
        _predicted_values = outputs

        # 将真实值和预测值存起来
        # true_values.append(torch.sum(targets))
        true_values.append(torch.sum(targets_normalized_data))  # 对所有真实对所有边的带宽求和

        predicted_values.append(torch.sum(_predicted_values))   # 对所有预测边的带宽求和
        index_list.append(index)

        print(f"真实值：{input_normalized_data}, 预测值：{ _predicted_values}")
        sys.exit()

    return index_list, true_values, predicted_values

if __name__ == "__main__":

    # D:/Desktop/Overlay-Multicast-predict-RL/predict/transform/model_weight/2026_04_14_10_03_17/10_transformer_model.pth
    model.load_state_dict(torch.load('./model_weight/2026_04_14_10_36_01/50_transformer_model.pth'))
    model.eval()  # Set the model to evaluation mode    
    index_list, true_values, predicted_values = predict()


    #将数据存储到文件当中
    # with open("./lfy_result.txt", "w+", encoding="utf-8") as file:
    #     for item1, item2, item3 in zip(index_list, true_values, predicted_values):
    #         file.write(f"{item1}\t{item2}\t{item3}\n")
    #
    # print(f"Lists have been written to file_path")
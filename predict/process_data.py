#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/4/8 13:43
@File: process_data.py
@Decs:*********************
"""
import pandas as pd
from draw import draw_line_chat, draw_error_line_chat, draw_bar_chat






lstm_datapath = '.\\lstm\\lfy_result.txt'
gru_datapath = '.\\gru\\lfy_result.txt'
transformer_datapath = '.\\transform\\lfy_result.txt'

lstm_csv= pd.read_csv(lstm_datapath,sep='\s+',header=None, names=['col1', 'col2', 'col3'])
gru_csv = pd.read_csv(gru_datapath,sep='\s+',header=None, names=['col1', 'col2', 'col3'])
transforme_csv = pd.read_csv(transformer_datapath,sep='\s+',header=None, names=['col1', 'col2', 'col3'])

index = lstm_csv['col1'].to_numpy()
true_values = lstm_csv['col2'].to_numpy()
lstm_values = lstm_csv['col3'].to_numpy()
gru_values = gru_csv['col3'].to_numpy()
transformer_values = transforme_csv['col3'].to_numpy()

'''MAE'''
lstm_mae = (lstm_csv['col3'] - lstm_csv['col2']).abs().to_numpy()
gru_mae = (gru_csv['col3'] - gru_csv['col2']).abs().to_numpy()
transformer_mae = (transforme_csv['col3'] - transforme_csv['col2']).abs().to_numpy()

'''MSE'''
lstm_mse = ((lstm_csv['col3'] - lstm_csv['col2'])**2).to_numpy()
gru_mse = ((gru_csv['col3'] - gru_csv['col2'])**2).to_numpy()
transformer_mse = ((transforme_csv['col3'] - transforme_csv['col2'])**2).to_numpy()

'''MAPE'''
lstm_mape = ((lstm_csv['col3'] - lstm_csv['col2']).abs() / lstm_csv['col2'].replace(0, 1e-8).abs()).to_numpy()
gru_mape = ((gru_csv['col3'] - gru_csv['col2']).abs() / gru_csv['col2'].replace(0, 1e-8).abs()).to_numpy()
transformer_mape = ((transforme_csv['col3'] - transforme_csv['col2']).abs() / transforme_csv['col2'].replace(0, 1e-8).abs()).to_numpy()

# lstm_mape = (lstm_csv['col3'] - lstm_csv['col2']).abs()/lstm_csv['col2'].replace(0, 1e-8).abs().to_numpy()
# gru_mape = (gru_csv['col3'] - gru_csv['col2']).abs()/gru_csv['col2'].replace(0, 1e-8).abs().to_numpy()
# transformer_mape = ((transforme_csv['col3'] - transforme_csv['col2']).abs()/transforme_csv['col2'].replace(0, 1e-8).abs()).to_numpy()

'''Error'''
lstm_error=[lstm_mae.mean(), lstm_mse.mean(), lstm_mape.mean()]
gru_error=[gru_mae.mean(), gru_mse.mean(), gru_mape.mean()]
transformer_error=[transformer_mae.mean(), transformer_mse.mean(), transformer_mape.mean()]



# 预测数据对比图
draw_line_chat(index, true_values, lstm_values, gru_values, transformer_values, 'lfy_result.png')
#print("预测数据对比图： ",true_values, lstm_values, gru_values, transformer_values, sep="\n")


# 三种指标对比图
draw_error_line_chat(index, lstm_mae,  gru_mae, transformer_mae, '绝对误差', 'lfy_mae_result.png')
draw_error_line_chat(index, lstm_mse,  gru_mse, transformer_mse, '方差', 'lfy_mse_result.png')
draw_error_line_chat(index, lstm_mape,  gru_mape, transformer_mape, '绝对百分比误差', 'lfy_mape_result.png')

# 三种指标柱状图
draw_bar_chat(transformer_error, lstm_error, gru_error, 'lfy_compare_result.png')

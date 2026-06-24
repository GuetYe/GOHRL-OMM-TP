#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:LFY
@Contact:2545039870@qq.com
@Time: 2026/1/20 13:09
@File: lstm_model.py
@Decs:*********************
"""

import torch.nn as nn
import torch
import math
from gpu import gpu



class LSTMModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=128, n_layers=3, drop_prob=0.1):
        """
        LSTM初始化函数
        :param n_input:     输入数据的特征维度
        :param n_hidden:    隐藏层特征维度
        :param n_layers:    LSTM块的个数
        :param drop_prob:   dropout的概率
        """
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(n_input, n_hidden, n_layers, dropout=drop_prob)
        self.fc = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        _batch_size = x.size(0)
        _hidden = self.init_hidden(_batch_size, x.device)
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm(x, _hidden)
        # x = self.dropout(x)
        # out = x.contiguous().view(-1, _batch_size, self.n_hidden)
        # out = out.permute(1, 0, 2)
        # 只取最后一个时间步的输出
        # x 的形状为 (sequence_length, batch_size, n_hidden)
        # 取最后一个时间步的输出，形状为 (batch_size, n_hidden)
        out = x[-1, :, :]
        
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        """ Initialize hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))

        return hidden


def lstm_init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    """ for test """
    data = torch.rand((8 ,10, 58))  # [batch_size, seq_len, feature_size]


    model = LSTMModel(n_input=58, n_output=58)
    output = model(data)
    print(f'LSTM output shape:{output.shape}')

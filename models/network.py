import torch
import numpy as np
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=256):
        super(StateEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        return x


class LSTM(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=768, num_layer=1):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.in_dim, self.hidden_dim, self.num_layer)

    def init_hidden(self, batch_size):
        self.h = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()
        self.c = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()

    def forward(self, x):
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        return x


class GRUModel(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden


class Decoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=256):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=True)
        self.fc2 = nn.Linear(hidden_dim // 2, out_dim - 2, bias=True)
        self.fc_conct = nn.Linear(hidden_dim // 2, 2, bias=True)
        self.ac_sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        o1 = self.fc2(x)
        o2 = self.ac_sig(self.fc_conct(x))
        return o1, o2

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


class ControlEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=128, out_dim=64):
        super(ControlEncoder, self).__init__()
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
    def __init__(self, in_dim=128, hidden_dim=1024, num_layer=3):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.in_dim, self.hidden_dim, self.num_layer)

    def init_hidden(self, batch_size):
        self.h = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).normal_(std=0.01).cuda()
        self.c = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()

    def forward(self, x):
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        return x


class GRUModel(nn.Module):

    def __init__(self, input_num, h_size=1000):
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_size=input_num, hidden_size=h_size, num_layers=2, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, 1, h_size).normal_(std=0.01), requires_grad=True)

    def forward(self, x, h=None):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()
        x, h = self.rnn(x, h)
        return x, h


class Decoder(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=256):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=True)
        # self.fc2 = nn.Linear(hidden_dim // 2, out_dim - 2, bias=True)
        # self.fc_conct = nn.Linear(hidden_dim // 2, 2, bias=True)
        # self.ac_sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        # x = nn.functional.relu(x)
        # o1 = self.fc2(x)
        # o2 = self.ac_sig(self.fc_conct(x))
        # return o1, o2
        return x

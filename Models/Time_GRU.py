from settings import *
import torch
import torch.nn as nn

class Time_GRU(torch.nn.Module):
    def __init__(self, input_length, hidden_length):
        super(Time_GRU, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        # reset gate components (r)
        self.linear_reset_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_reset_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_reset = nn.Sigmoid()
        # update gate components (z)
        self.linear_update_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_update_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_update = nn.Sigmoid()
        # cell memory components (n)
        self.linear_cell_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell = nn.Tanh()
        # time gate components
        self.linear_time_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_time_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_time = nn.Sigmoid()
        self.sigmoid_time_inner = nn.Sigmoid()

    def reset_gate(self, x, h):
        x_temp = self.linear_reset_W_x(x).to(device)
        h_temp = self.linear_reset_W_h(h).to(device)
        r = self.sigmoid_reset(x_temp + h_temp).to(device)
        return r

    def update_gate(self, x, h):
        x_temp = self.linear_update_W_x(x).to(device)
        h_temp = self.linear_update_W_h(h).to(device)
        z = self.sigmoid_update(x_temp + h_temp).to(device)
        return z

    def time_gate(self, x, delt):
        x = self.linear_time_W_x(x).to(device)
        tt = self.linear_time_W_t(delt).to(device)
        t = self.sigmoid_time_inner(tt).to(device)
        T = self.sigmoid_time(x + t).to(device)
        return T

    def cell_gate(self, x, h, r, T):
        x_temp = self.linear_cell_W_x(x).to(device)
        h_temp = self.linear_cell_W_h(h).to(device)
        n = self.tanh_cell(x_temp + r * h_temp).to(device)
        return n*T

    def forward(self, x, h, delt):
        r = self.reset_gate(x, h).to(device)
        z = self.update_gate(x, h).to(device)
        T = self.time_gate(x, delt).to(device)
        n = self.cell_gate(x, h, r, T).to(device)
        h_next = (1 - z) * n + z * h
        return h_next.to(device)


class Sequence_Time_GRU(nn.Module):
    def __init__(self, device):
        super(Sequence_Time_GRU, self).__init__()
        self.rnn1 = Time_GRU(93, 93, device).to(device)
        self.rnn2 = Time_GRU(93, 93, device).to(device)
        self.linear = nn.Linear(93, 93).to(device)

    def forward(self, x, t, ft):
        outputs = []
        h_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        h_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        t_max = t.max()
        t = t / t_max
        ft_max = ft.max()
        ft = ft / ft_max
        for i in range(0, INPUT_LENGTH):
            h_1 = self.rnn1(x[i].float(), h_1, t[i].float())
            h_2 = self.rnn2(h_1, h_2, t[i].float())
        for i in range(0, OUTPUT_LENGTH):
            h_1 = self.rnn1(h_2.float(), h_1, ft[i].float())
            h_2 = self.rnn2(h_1, h_2, ft[i].float())
            output = self.linear(h_2).squeeze(0)
            outputs += [output]
        outputs = torch.stack(outputs, 0).to(device)
        return outputs
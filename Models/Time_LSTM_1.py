from settings import *
import torch
import torch.nn as nn




class Time_LSTM_1(torch.nn.Module):
    def __init__(self, input_length, hidden_length):
        super(Time_LSTM_1, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        # input gate components
        self.linear_input_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input = nn.Sigmoid()
        # forget gate components
        self.linear_forget_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()
        # time gate components
        self.linear_time_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_time_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_time = nn.Sigmoid()
        self.sigmoid_time_inner = nn.Sigmoid()
        # cell memory components
        self.linear_cell_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.tanh_cell = nn.Tanh()
        # out gate components
        self.linear_output_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_output_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False)
        self.sigmoid_out = nn.Sigmoid()
        self.activation_final = nn.Tanh()

    def input_gate(self, x, h, c_prev):
        x_temp = self.linear_input_W_x(x).to(device)
        h_temp = self.linear_input_W_h(h).to(device)
        c_temp = self.linear_input_w_c(c_prev).to(device)
        i = self.sigmoid_input(x_temp + h_temp + c_temp).to(device)
        return i

    def forget_gate(self, x, h, c_prev):
        x = self.linear_forget_W_x(x).to(device)
        h = self.linear_forget_W_h(h).to(device)
        c = self.linear_forget_w_c(c_prev).to(device)
        f = self.sigmoid_forget(x + h + c).to(device)
        return f

    def time_gate(self, x, delt):
        x = self.linear_time_W_x(x).to(device)
        tt = self.linear_time_W_t(delt).to(device)
        t = self.sigmoid_time_inner(tt).to(device)
        T = self.sigmoid_time(x + t).to(device)
        return T

    def cell_memory_gate(self, i, f, x, h, c_prev, T):
        x = self.linear_cell_W_x(x).to(device)
        h = self.linear_cell_W_h(h).to(device)
        k = self.tanh_cell(x + h).to(device)
        g = (i * T * k).to(device)
        c = (f * c_prev).to(device)
        c_next = g + c
        return c_next.to(device)

    def out_gate(self, x, h, c_prev, delt):
        x = self.linear_output_W_x(x).to(device)
        t = self.linear_output_W_t(delt).to(device)
        h = self.linear_output_W_h(h).to(device)
        c = self.linear_output_w_c(c_prev).to(device)
        o = self.sigmoid_out(x + t + h + c).to(device)
        return o

    def forward(self, x, tuple_in, delt):
        (h, c_prev) = tuple_in
        i = self.input_gate(x, h, c_prev).to(device)
        f = self.forget_gate(x, h, c_prev).to(device)
        T = self.time_gate(x, delt).to(device)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev, T).to(device)
        o = self.out_gate(x, h, c_prev, delt).to(device)
        h_next = o * self.activation_final(c_next).to(device)
        return h_next, c_next, T


class Sequence_Time_LSTM_1(nn.Module):
    def __init__(self, device):
        super(Sequence_Time_LSTM_1, self).__init__()
        self.rnn1 = Time_LSTM_1(93, 93, device).to(device)
        self.rnn2 = Time_LSTM_1(93, 93, device).to(device)
        self.linear = nn.Linear(93, 93).to(device)

    def forward(self, x, t, ft):
        outputs = []
        h_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        h_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        t_max = t.max()
        t = t / t_max
        ft_max = ft.max()
        ft = ft / ft_max
        for i in range(0, INPUT_LENGTH):
            h_1, c_1, time = self.rnn1(x[i].float(), (h_1, c_1), t[i].float())
            h_2, c_2, time = self.rnn2(h_1, (h_2, c_2), t[i].float())
        for i in range(0, OUTPUT_LENGTH):
            h_1, c_1, time = self.rnn1(h_2.float(), (h_1, c_1), ft[i].float())
            h_2, c_2, time = self.rnn2(h_1, (h_2, c_2), ft[i].float())
            output = self.linear(h_2).squeeze(0)
            outputs += [output]
        outputs = torch.stack(outputs, 0).to(device)
        return outputs



from settings import *
import torch
import torch.nn as nn


class Time_LSTM_2(torch.nn.Module):
    def __init__(self, input_length, hidden_length, device):
        super(Time_LSTM_2, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        # input gate components
        self.linear_input_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.sigmoid_input = nn.Sigmoid()
        # forget gate components
        self.linear_forget_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_forget_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.linear_forget_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.sigmoid_forget = nn.Sigmoid().to(device)
        # time gate 1 components
        self.linear_time_1_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_time_1_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False).to(device)
        self.sigmoid_time_1 = nn.Sigmoid().to(device)
        self.sigmoid_time_1_inner = nn.Sigmoid().to(device)
        # time gate 2 components
        self.linear_time_2_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_time_2_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False).to(device)
        self.sigmoid_time_2 = nn.Sigmoid().to(device)
        self.sigmoid_time_2_inner = nn.Sigmoid().to(device)
        # cell memory components
        self.linear_cell_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.tanh_cell = nn.Tanh().to(device)
        # cell curly components
        self.linear_cell_curly_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_cell_curly_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.tanh_cell_curly = nn.Tanh().to(device)
        # out gate components
        self.linear_output_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True).to(device)
        self.linear_output_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.linear_output_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False).to(device)
        self.linear_output_W_t = nn.Linear(TIME_EMBED_SIZE, self.hidden_length, bias=False).to(device)
        self.sigmoid_out = nn.Sigmoid().to(device)
        self.activation_final = nn.Tanh().to(device)

    def clamp_time_1_gate_weights(self):
        self.linear_time_1_W_t.weight.clamp(min=0)

    def input_gate(self, x, h, c_prev):
        x_temp = self.linear_input_W_x(x)
        h_temp = self.linear_input_W_h(h)
        c_temp = self.linear_input_w_c(c_prev)
        i = self.sigmoid_input(x_temp + h_temp + c_temp)
        return i

    def forget_gate(self, x, h, c_prev):
        x = self.linear_forget_W_x(x)
        h = self.linear_forget_W_h(h)
        c = self.linear_forget_w_c(c_prev)
        f = self.sigmoid_forget(x + h + c)
        return f

    def time_1_gate(self, x, delt):
        x = self.linear_time_1_W_x(x)
        tt = self.linear_time_1_W_t(delt)
        t = self.sigmoid_time_1_inner(tt)
        T = self.sigmoid_time_1(x + t)
        return T

    def time_2_gate(self, x, delt):
        x = self.linear_time_2_W_x(x)
        tt = self.linear_time_2_W_t(delt)
        t = self.sigmoid_time_2_inner(tt)
        T = self.sigmoid_time_2(x + t)
        return T

    def cell_curly_gate(self, i, f, x, h, c_prev, T1):
        x = self.linear_cell_curly_W_x(x)
        h = self.linear_cell_curly_W_h(h)
        k = self.tanh_cell_curly(x + h)
        g = i * T1 * k
        c = f * c_prev
        c_next = g + c
        return c_next

    def cell_memory_gate(self, i, f, x, h, c_prev, T2):
        x = self.linear_cell_W_x(x)
        h = self.linear_cell_W_h(h)
        k = self.tanh_cell(x + h)
        g = i * T2 * k
        c = f * c_prev
        c_next = g + c
        return c_next

    def out_gate(self, x, h, c_curl, delt):
        x = self.linear_output_W_x(x)
        t = self.linear_output_W_t(delt)
        h = self.linear_output_W_h(h)
        c = self.linear_output_w_c(c_curl)
        o = self.sigmoid_out(x + t + h + c)
        return o

    def forward(self, x, tuple_in, delt):
        (h, c_prev) = tuple_in
        i = self.input_gate(x, h, c_prev)
        f = self.forget_gate(x, h, c_prev)
        T1 = self.time_1_gate(x, delt)
        T2 = self.time_2_gate(x, delt)
        c_curl = self.cell_memory_gate(i, f, x, h, c_prev, T1)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev, T2)
        o = self.out_gate(x, h, c_curl, delt)
        h_next = o * self.activation_final(c_curl)
        return h_next, c_next


class Sequence_Time_LSTM_2(nn.Module):
    def __init__(self):
        super(Sequence_Time_LSTM_2, self).__init__()
        self.rnn1 = Time_LSTM_2(93, 93, device).to(device)
        self.rnn2 = Time_LSTM_2(93, 93, device).to(device)
        self.linear = nn.Linear(93, 93).to(device)

    def clamp_tg1_w(self):
        self.rnn1.clamp_time_1_gate_weights()

    def forward(self, x, t, ft):
        outputs = []
        h_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        h_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        self.clamp_tg1_w()
        for i in range(0, INPUT_LENGTH):
            h_1, c_1 = self.rnn1(x[i].float(), (h_1, c_1), t[i].float())
            h_2, c_2 = self.rnn2(h_1, (h_2, c_2), t[i].float())
        for i in range(0, OUTPUT_LENGTH):
            h_1, c_1 = self.rnn1(h_2.float(), (h_1, c_1), ft[i].float())
            h_2, c_2 = self.rnn2(h_1, (h_2, c_2), ft[i].float())
            output = self.linear(h_2).squeeze(0)
            outputs += [output]
        outputs = torch.stack(outputs, 0).to(device)
        return outputs

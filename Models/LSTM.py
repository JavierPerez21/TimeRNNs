from settings import *
import torch
import torch.nn as nn


class LSTM(torch.nn.Module):
    def __init__(self, input_length, hidden_length):
        super(LSTM, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        # input gate components
        self.linear_input_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_input_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_input_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()
        # forget gate components
        self.linear_forget_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()
        # cell memory components
        self.linear_cell_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_cell_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()
        # out gate components
        self.linear_output_W_x = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_output_W_h = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_output_w_c = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()
        self.activation_final = nn.Tanh()

    def input_gate(self, x, h, c_prev):
        x_temp = self.linear_input_W_x(x).to(device)
        h_temp = self.linear_input_W_h(h).to(device)
        c_temp = self.linear_input_w_c(c_prev).to(device)
        i = self.sigmoid_gate(x_temp + h_temp + c_temp).to(device)
        return i

    def forget_gate(self, x, h, c_prev):
        x = self.linear_forget_W_x(x).to(device)
        h = self.linear_forget_W_h(h).to(device)
        c = self.linear_forget_w_c(c_prev).to(device)
        f = self.sigmoid_forget(x + h + c).to(device)
        return f

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_cell_W_x(x).to(device)
        h = self.linear_cell_W_h(h).to(device)
        k = self.activation_gate(x + h).to(device)
        g = (k * i).to(device)
        c = f * c_prev.to(device)
        c_next = g + c
        return c_next.to(device)

    def out_gate(self, x, h, c_prev):
        x = self.linear_output_W_x(x).to(device)
        h = self.linear_output_W_h(h).to(device)
        c = self.linear_output_w_c(c_prev).to(device)
        o = self.sigmoid_hidden_out(x + h + c).to(device)
        return o

    def forward(self, x, tuple_in):
        (h, c_prev) = tuple_in
        i = self.input_gate(x, h, c_prev)
        f = self.forget_gate(x, h, c_prev)
        c_next = self.cell_memory_gate(i, f, x, h, c_prev)
        o = self.out_gate(x, h, c_prev)
        h_next = o * self.activation_final(c_next)
        return h_next.to(device), c_next.to(device)


class Sequence_LSTM(nn.Module):
    def __init__(self, device):
        super(Sequence_LSTM, self).__init__()
        self.rnn1 = LSTM(93, 93, device).to(device)
        self.rnn2 = LSTM(93, 93, device).to(device)
        self.linear = nn.Linear(93, 93).to(device)

    def forward(self, x):
        outputs = []
        h_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_1 = torch.zeros(1, 93, dtype=torch.float).to(device)
        h_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        c_2 = torch.zeros(1, 93, dtype=torch.float).to(device)
        for i in range(0, INPUT_LENGTH):
            h_1, c_1 = self.rnn1(x[i].float(), (h_1, c_1))
            h_2, c_2 = self.rnn2(h_1, (h_2, c_2))

        for i in range(0, OUTPUT_LENGTH):
            h_1, c_1 = self.rnn1(h_2.float(), (h_1, c_1))
            h_2, c_2 = self.rnn2(h_1, (h_2, c_2))
            output = self.linear(h_2).squeeze(0)
            outputs += [output]
        outputs = torch.stack(outputs, 0).to(device)
        return outputs


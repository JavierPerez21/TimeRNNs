from Models import GRU, LSTM, Time_GRU, Time_LSTM_1, Time_LSTM_2, Markov_Chain
from data_preprocessing import *
from settings import *
from utils import *
from torch.utils.data.dataloader import DataLoader
import time
from torch import optim

train, test, ohe, ohe_time = create_efficient_dataset(data)
train_loader=DataLoader(train, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
print("Creating loaders")
test_loader=DataLoader(test, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
print("Data loaders created")

print(device)
gru = GRU.Sequence_GRU(device)
lstm = LSTM.Sequence_LSTM(device)
time_lstm_1 = Time_LSTM_1.Sequence_Time_LSTM_1(device)
time_lstm_2 = Time_LSTM_2.Sequence_Time_LSTM_2(device)
time_gru = Time_GRU.Sequence_GRU(device)
uniques, unique_titles, transition_matrix = Markov_Chain.generate_table_of_frequencies(datapath)
markov_chain = Markov_Chain.Markov_Chain(transition_matrix, device)

rnns = [lstm, gru]
time_rnns = [time_lstm_1, time_lstm_2, time_gru]
hist = []

print("Begin testing")
print(markov_chain)
print("epoch, train_acc, train_loss, train_k, test_acc, test_loss, test_k, train_time_seconds")
test_err, test_loss, test_k = rnn_epoch(markov_chain, test_loader, ohe)
print("{{:.6f}   {:.6f}   {:.6f}    {:.0f}".format(1 - test_err,test_loss, test_k))

for rnn in rnns:
    print(rnn)
    print("epoch, train_acc, train_loss, train_k, test_acc, test_loss, test_k, train_time_seconds")
    opt = optim.Adadelta(rnn.parameters(), lr=1)
    for i in range(0, num_epochs):
        now = time.time()
        train_err, train_loss, train_k = rnn_epoch(rnn, train_loader, ohe, opt)
        time1 = time.time() - now
        test_err, test_loss, test_k = rnn_epoch(rnn, test_loader, ohe)
        print(
            "{}      {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f}    {:.0f}".format(i, 1 - train_err, train_loss,
                                                                                           train_k, 1 - test_err,
                                                                                           test_loss, test_k, time1))
        hist.append([1 - train_err, train_loss, 1 - test_err, test_loss])
for time_rnn in time_rnns:
    print(time_rnn)
    print("epoch, train_acc, train_loss, train_k, test_acc, test_loss, test_k, train_time_seconds")
    opt = optim.Adadelta(time_rnn.parameters(), lr=1)
    for i in range(0, num_epochs):
        now = time.time()
        train_err, train_loss, train_k = time_rnn_epoch(time_rnn, train_loader, ohe, ohe_time, opt)
        time1 = time.time() - now
        test_err, test_loss, test_k = time_rnn_epoch(time_rnn, test_loader, ohe, ohe_time)
        print(
            "{}      {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f}    {:.0f}".format(i, 1 - train_err, train_loss,
                                                                                           train_k, 1 - test_err,
                                                                                           test_loss, test_k, time1))
        hist.append([1 - train_err, train_loss, 1 - test_err, test_loss])

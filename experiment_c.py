from Models import Time_GRU
from data_preprocessing import *
from settings import *
from utils import *
from torch.utils.data.dataloader import DataLoader
import time
from torch import optim

train, test, ohe = create_efficient_continuous_dataset(datapath)
train_loader=DataLoader(train, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
print("Creating loaders")
test_loader=DataLoader(test, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
print("Data loaders created")

print(device)
print("Creating model")
time_gru = Time_GRU.Sequence_Time_GRU(device)
print("Model created")
opt = optim.Adadelta(time_gru.parameters(), lr=1)
hist = []
num_epochs = 80
print("Training ...")
print("epoch, train_err, train_loss, train_k, test_err, test_loss, test_k, train_time_seconds")
for i in range(0,num_epochs):
    now = time.time()
    train_err, train_loss, train_k = continuous_time_rnn_epoch(time_gru, train_loader, ohe, opt)
    time1 = time.time() - now
    test_err, test_loss, test_k = continuous_time_rnn_epoch(time_gru, test_loader, ohe)
    print("{}      {:.6f}   {:.6f}   {:.6f}   {:.6f}    {:.6f}   {:.6f}    {:.0f}".format(i, 1 - train_err, train_loss, train_k, 1 - test_err, test_loss, test_k, time1))
    hist.append([1 - train_err, train_loss, 1- test_err, test_loss])

in_lens = [10, 20, 50, 100, 200]
out_lens = [10, 20, 50, 100, 200]
for in_len in in_lens:
    for out_len in out_lens:
        INPUT_LENGTH = in_len
        OUTPUT_LENGTH = out_len
        LONGER_LENGTH = INPUT_LENGTH
        if OUTPUT_LENGTH > INPUT_LENGTH:
          LONGER_LENGTH = OUTPUT_LENGTH
        # Create loaders
        print(in_len, out_len)
        train, test, ohe = create_efficient_dataset(data)
        train_loader=DataLoader(train, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
        print("Creating loaders")
        test_loader=DataLoader(test, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
        print("Data loaders created")

        test_err, test_loss, k_test = continuous_time_rnn_epoch(time_gru, test_loader, ohe)
print("{:.6f}   {:.6f}    {:.6f}    {:.0f}".format(1-test_err, test_loss, k_test, time1))
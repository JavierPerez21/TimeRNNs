from settings import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from torch.utils.data import random_split, TensorDataset
import random
import psutil
import torch

def create_initial_data(filenames):
    single_events = []
    time = []
    for i in range(0, len(filenames)):
        data = pd.read_csv(filenames[i])
        time.append((data["time"]))
        single_events.append(data['class'][2:] + data['event'][2:])

    # Event encoding
    le = LabelEncoder()
    le.fit(single_events[0])
    single_events = pd.concat(single_events)
    single_events = le.transform(single_events).reshape(-1, 1)
    ohe = OneHotEncoder()
    ohe.fit(single_events)

    # Time encoding
    for i in range(0, len(time)):
        time[i] = np.array(time[i][2:]) - np.array(time[i][1:-1])
    time = np.concatenate(time).reshape(-1, 1)

    temp = np.concatenate((time, single_events), axis=1)
    i = 0
    while i < len(temp):
        if temp[i][0] < 0:
            temp = np.delete(temp, i, 0)
        else:
            i += 1
    single_events = temp[:, 1:]
    time = temp[:, :1]

    step2 = (time.max() - time.min()) / TIME_EMBED_SIZE
    step1 = step2 / (TIME_EMBED_SIZE * 0.8)
    bins1 = np.arange(time.min(), step2, step1)
    bins2 = np.arange(step2, time.max(), step2 * 5)
    bins = np.concatenate((bins1, bins2), axis=0)
    time = np.digitize(time, bins).reshape(-1, 1) - 1
    ohe_time = OneHotEncoder()
    ohe_time.fit(time)
    return time, single_events, ohe, ohe_time


def create_data_matrices(fraction, time, single_events):
    X = []
    T = []
    Y = []
    T_F = []
    X_test = []
    Y_test = []
    T_test = []
    T_F_test = []
    lim = int(len(single_events) * fraction) - (OUTPUT_LENGTH + INPUT_LENGTH)
    print(lim)
    for i in range(1000, lim):
        if random.random() < 0.8:
            X.append(single_events[i - INPUT_LENGTH:i, :])
            T.append(time[i - INPUT_LENGTH:i, :])
            Y.append(single_events[i:i + OUTPUT_LENGTH])
            T_F.append(time[i:i + OUTPUT_LENGTH])
        else:
            X_test.append(single_events[i - INPUT_LENGTH:i, :])
            T_test.append(time[i - INPUT_LENGTH:i, :])
            Y_test.append(single_events[i:i + OUTPUT_LENGTH])
            T_F_test.append(time[i:i + OUTPUT_LENGTH])
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)
    T_F = np.array(T_F)
    print(X.shape, T.shape, T_F.shape)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    T_test = np.array(T_test)
    T_F_test = np.array(T_F_test)

    return X, T, T_F, Y, X_test, T_test, T_F_test, Y_test


def create_efficient_dataset(filenames, fraction=1):
    time, single_events, ohe, ohe_time = create_initial_data(filenames)

    X_train, T_train, T_F_train, Y_train, X_test, T_test, T_F_test, Y_test = create_data_matrices(fraction, time,
                                                                                                  single_events)

    X_et_train = []
    X_et_train.append(X_train)
    X_et_train.append(T_train)
    Y_et_train = []
    Y_et_train.append(Y_train)
    Y_et_train.append(T_F_train)
    X_et_train = torch.from_numpy(np.array(X_et_train).squeeze()).permute(1, 0, 2)
    Y_et_train = torch.from_numpy(np.array(Y_et_train).squeeze()).permute(1, 0, 2)

    X_et_test = []
    X_et_test.append(X_test)
    X_et_test.append(T_test)
    Y_et_test = []
    Y_et_test.append(Y_test)
    Y_et_test.append(T_F_test)
    X_et_test = torch.from_numpy(np.array(X_et_test).squeeze()).permute(1, 0, 2)
    Y_et_test = torch.from_numpy(np.array(Y_et_test).squeeze()).permute(1, 0, 2)

    print(X_et_train.shape, Y_et_train.shape)
    print(X_et_test.shape, Y_et_test.shape)

    train = TensorDataset(X_et_train.float(), Y_et_train)
    test = TensorDataset(X_et_test.float(), Y_et_test)
    print("Available: ", psutil.virtual_memory().available / 1000000000)

    return train, test, ohe, ohe_time


def create_initial_continuous_data(filenames):
    single_events = []
    time = []
    for i in range(0, len(filenames)):
      data = pd.read_csv(filenames[i])
      time.append((data["time"]))
      single_events.append(data['class'][2:] + data['event'][2:])

    # Event encoding
    le = LabelEncoder()
    le.fit(single_events[0])
    single_events = pd.concat(single_events)
    single_events = le.transform(single_events).reshape(-1, 1)
    ohe = OneHotEncoder()
    ohe.fit(single_events)

    # Time encoding
    for i in range(0, len(time)):
      time[i] = np.array(time[i][2:]) - np.array(time[i][1:-1])
    time = np.concatenate(time).reshape(-1,1)

    temp = np.concatenate((time, single_events), axis = 1)
    i = 0
    while i < len(temp):
      if temp[i][0] < 0:
        temp = np.delete(temp, i, 0)
      else:
        i+=1
    single_events = temp[:, 1]
    time = temp[:, 0]
    return time, single_events, ohe


def create_efficient_continuous_dataset(filenames, fraction=1):
    time, single_events, ohe = create_initial_continuous_data(filenames)

    X_train, T_train, T_F_train, Y_train, X_test, T_test, T_F_test, Y_test = create_data_matrices(fraction, time,
                                                                                                  single_events)

    X_et_train = []
    X_et_train.append(X_train)
    X_et_train.append(T_train)
    Y_et_train = []
    Y_et_train.append(Y_train)
    Y_et_train.append(T_F_train)
    X_et_train = torch.from_numpy(np.array(X_et_train).squeeze()).permute(1, 0, 2)
    Y_et_train = torch.from_numpy(np.array(Y_et_train).squeeze()).permute(1, 0, 2)

    X_et_test = []
    X_et_test.append(X_test)
    X_et_test.append(T_test)
    Y_et_test = []
    Y_et_test.append(Y_test)
    Y_et_test.append(T_F_test)
    X_et_test = torch.from_numpy(np.array(X_et_test).squeeze()).permute(1, 0, 2)
    Y_et_test = torch.from_numpy(np.array(Y_et_test).squeeze()).permute(1, 0, 2)

    print(X_et_train.shape, Y_et_train.shape)
    print(X_et_test.shape, Y_et_test.shape)

    train = TensorDataset(X_et_train.float(), Y_et_train)
    test = TensorDataset(X_et_test.float(), Y_et_test)
    print("Available: ", psutil.virtual_memory().available / 1000000000)

    return train, test, ohe
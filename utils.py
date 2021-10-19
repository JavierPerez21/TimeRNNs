from settings import *
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
import numpy as np


def rnn_epoch(model, loader, ohe, opt=None):
    total_loss, total_err = 0., 0.
    kappa_scores = []
    for x, y in loader:
        batch = len(x)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        x = x[0, :, :].squeeze().reshape(batch * INPUT_LENGTH, 1)
        y = y[0, :, :].squeeze()
        x = ohe.transform(x).toarray()
        x = torch.from_numpy(x).reshape(batch, INPUT_LENGTH, 93).permute(1, 0, 2).to(device)
        yp = model(x).permute(1, 0, 2).reshape(-1, 93)
        y = y.reshape(-1).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
        kappa_scores.append(cohen_kappa_score(y.detach().cpu(), yp.max(1)[1].detach().cpu()))
    kappa_score = np.mean(kappa_scores)
    return total_err / (len(loader.dataset) * OUTPUT_LENGTH), total_loss / (
            len(loader.dataset) * OUTPUT_LENGTH), kappa_score


def time_rnn_epoch(model, loader, ohe, ohe_time, opt=None):
    total_loss, total_err = 0., 0.
    kappa_scores = []
    for x, y in loader:
        batch = len(x)
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        t = x[1:, :, :].squeeze().reshape(batch * INPUT_LENGTH, 1)
        x = x[:1, :, :].squeeze().reshape(batch * INPUT_LENGTH, 1)
        ft = y[1:, :, :].squeeze().reshape(batch * OUTPUT_LENGTH, 1)
        y = y[:1, :, :].squeeze().reshape(-1, 1).flatten().to(device)
        x = ohe.transform(x).toarray().reshape(batch, INPUT_LENGTH, 93)
        x = torch.from_numpy(x).permute(1, 0, 2).to(device)
        t = ohe_time.transform(t).toarray().reshape(batch, INPUT_LENGTH, TIME_EMBED_SIZE)
        t = torch.from_numpy(t).permute(1, 0, 2).to(device)
        ft = ohe_time.transform(ft).toarray().reshape(batch, OUTPUT_LENGTH, TIME_EMBED_SIZE)
        ft = torch.from_numpy(ft).permute(1, 0, 2).to(device)
        yp = model(x, t, ft).permute(1, 0, 2).reshape(-1, 93)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
        kappa_scores.append(cohen_kappa_score(y.detach().cpu(), yp.max(1)[1].detach().cpu()))
    kappa_score = np.mean(kappa_scores)
    return total_err / (len(loader.dataset) * OUTPUT_LENGTH), total_loss / (
                len(loader.dataset) * OUTPUT_LENGTH), kappa_score


def continuous_time_rnn_epoch(model, loader, ohe, opt=None):
    total_loss, total_err = 0., 0.
    kappa_scores = []
    for x, y in loader:
        batch = len(x)
        # Preprocess
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        t = x[1, :, :].squeeze().reshape(batch,INPUT_LENGTH,1).permute(1,0,2).to(device)
        x = x[0, :, :].squeeze().reshape(batch*INPUT_LENGTH,1)
        ft = y[1, :, :].squeeze().reshape(batch,OUTPUT_LENGTH,1).permute(1,0,2).to(device)
        y = y[0, :, :].squeeze()
        x = ohe.transform(x).toarray()
        x =  torch.from_numpy(x).reshape(batch, INPUT_LENGTH,93).permute(1,0,2).to(device)
        t_max = t.max()
        t = t/t_max
        ft_max = ft.max()
        ft = ft/ft_max
        yp = model(x, t, ft).permute(1, 0, 2).reshape(-1, 93)
        y = y.reshape(-1).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * x.shape[0]
        kappa_scores.append(cohen_kappa_score(y.detach().cpu(), yp.max(1)[1].detach().cpu()))
    kappa_score = np.mean(kappa_scores)
    return total_err / (len(loader.dataset) * OUTPUT_LENGTH), total_loss / (len(loader.dataset) * OUTPUT_LENGTH), kappa_score
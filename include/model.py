# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:08:55 2022

@author: minhee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
import math


def train(model, train_loader, scheduler, epoch, criterion, DEVICE):
    model.train()
    train_loss = 0

    tqdm_bar = tqdm(enumerate(train_loader))

    for batch_idx, (voltage, label) in tqdm_bar:
        voltage = voltage.to(DEVICE)
        label = label.to(DEVICE)

        scheduler.zero_grad()
        output = model(voltage)
        loss = criterion(yhat=output, y=label)
        train_loss += loss.item()

        loss.backward()
        scheduler.step()
        tqdm_bar.set_description(
            "Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))

    scheduler.update()
    # train_loss /= len(train_loader.dataset)

    return train_loss


def evaluate(model, test_loader, criterion, DEVICE):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for voltage, label in tqdm(test_loader):
            voltage = voltage.to(DEVICE)
            label = label.to(DEVICE)

            output = model(voltage)
            test_loss += criterion(yhat=output, y=label).item()

    # test_loss /= len(test_loader.dataset)

    return test_loss


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) /\
            max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) /\
            max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


class ScheduleOptim():
    def __init__(self, optimizer, n_warmup_steps=10, decay_rate=0.9):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.decay = decay_rate
        self.n_steps = 0
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_lr = optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def get_lr(self):
        return self.current_lr

    def update(self):
        if self.n_steps < self.n_warmup_steps:
            lr = self.n_steps / self.n_warmup_steps * self.initial_lr
        elif self.n_steps == self.n_warmup_steps:
            lr = self.initial_lr
        else:
            lr = self.current_lr * self.decay

        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self.n_steps += 1


class CustomDataset(Dataset):
    def __init__(self, input_length=5, data_dir="../data/RH02/" \
                 + "Calibration/260Left_1/force_conversion_test.csv"):
        data = pd.read_csv(data_dir,delimiter=",")[["vout", "force"]]
        data["idx"] = data.index
        self.data = np.array(data[["idx", "vout", "force"]])
        self.input_length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def transform(self, x, idx):
        if (int(idx) + 1) < self.input_length:
            x = np.append(self.data[0, 1] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 1])
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1]
        return x

    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for (idx, x, y) in data:
            x = self.transform(x, idx)
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.cat(batch_y, dim=0).unsqueeze(1).float()
        return batch_x, batch_y


# Dimension: Batch * input length * 1
class CustomLSTMDataset(Dataset):
    def __init__(self, input_length=5, data_dir="../CHAR_1010_280/" \
                 + "Calibration/280Left_1/force_conversion_test.csv"):
        data = pd.read_csv(data_dir,delimiter=",")[["vout", "force"]]
        data["idx"] = data.index
        self.data = np.array(data[["idx", "vout", "force"]])
        self.input_length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def transform(self, x, idx):
        if (int(idx) + 1) < self.input_length:
            x = np.append(self.data[0, 1] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 1])
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1]
        return x

    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for (idx, x, y) in data:
            x = self.transform(x, idx)
            x = torch.from_numpy(x)
            x = x.unsqueeze(1)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.cat(batch_y, dim=0).unsqueeze(1).float()
        return batch_x, batch_y


class LSTMDataset_data(Dataset):
    def __init__(self, data, target, input_length=5,
                 reweight='none', lds=False, lds_kernel='gaussian',
                 lds_ks=5, lds_sigma=2):
        print(data)
        print(target)
        data = pd.DataFrame(data)
        target = pd.DataFrame(target)
        data = pd.DataFrame(np.concatenate((data, target), axis=1),
                            columns=["vout", "force"])
        data["idx"] = data.index
        self.data = np.array(data[["idx", "vout", "force"]])
        self.input_length = input_length
        self.weights = self._prepare_weights(reweight=reweight, lds=lds,
                                             lds_kernel=lds_kernel,
                                             lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vout = self.data[idx, 1]
        label = self.data[idx, 2]
        weight = self.weights[idx] if\
            self.weights is not None else np.float32(1.)
        return vout, label, weight

    def transform(self, x, idx):
        if (int(idx) + 1) < self.input_length:
            x = np.append(self.data[0, 1] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 1])
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1]
        return x

    def _prepare_weights(self, reweight, max_target=72, lds=False,
                         lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.data[:, 2]
        for label in labels:
            value_dict[min(max_target - 1, math.ceil(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, math.ceil(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, math.ceil(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for (idx, x, y) in data:
            x = self.transform(x, idx)
            x = torch.from_numpy(x)
            x = x.unsqueeze(1)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.cat(batch_y, dim=0).unsqueeze(1).float()
        return batch_x, batch_y


class LSTMDataset(Dataset):
    def __init__(self, input_length=5, data_dir="../CHAR_1010_280/" \
                 + "Calibration/280Left_1/force_conversion_test.csv",
                 reweight='none', lds=False, lds_kernel='gaussian',
                 lds_ks=5, lds_sigma=2):
        data = pd.read_csv(data_dir,delimiter=",")[["vout", "force"]]
        data["idx"] = data.index
        self.data = np.array(data[["idx", "vout", "force"]])
        self.input_length = input_length
        self.weights = self._prepare_weights(reweight=reweight, lds=lds,
                                             lds_kernel=lds_kernel,
                                             lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vout = self.data[idx, 1]
        label = self.data[idx, 2]
        weight = self.weights[idx] if\
            self.weights is not None else np.float32(1.)
        return vout, label, weight

    def transform(self, x, idx):
        if (int(idx) + 1) < self.input_length:
            x = np.append(self.data[0, 1] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 1])
        else:
            x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1]
        return x

    def _prepare_weights(self, reweight, max_target=72, lds=False,
                         lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.data[:, 2]
        for label in labels:
            value_dict[min(max_target - 1, math.ceil(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, math.ceil(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, math.ceil(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights

    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for (idx, x, y) in data:
            x = self.transform(x, idx)
            x = torch.from_numpy(x)
            x = x.unsqueeze(1)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.cat(batch_y, dim=0).unsqueeze(1).float()
        return batch_x, batch_y


# 사용하려면 input data 크기도 2개 column으로 맞춰야 함
class Custom2DDataset(Dataset):
    def __init__(self, input_length=5, data_dir="../CHAR_1010_280/" \
                 + "Calibration/280Left_1/force_conversion_test.csv"):
        data = pd.read_csv(data_dir,delimiter=",")[["time", "vout", "force"]]
        data["idx"] = data.index
        self.data = np.array(data[["idx", "time", "vout", "force"]])
        self.input_length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # dimension: (input_length * 2)
    def transform(self, idx):
        ref_time = self.data[int(idx), 1]

        if (int(idx) + 1) < self.input_length:
            ref_time_data = np.append(self.data[0, 1] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 1]) - ref_time
            vout_data = np.append(self.data[0, 2] *
                          np.ones((1, self.input_length - int(idx) - 1)),
                          self.data[0:int(idx) + 1, 2])
        else:
            ref_time_data = self.data[
                int(idx) + 1 - self.input_length:int(idx) + 1, 1] - ref_time
            vout_data = self.data[
                int(idx) + 1 - self.input_length:int(idx) + 1, 2]

        ref_time_data = ref_time_data[:, np.newaxis]
        vout_data = vout_data[:, np.newaxis]
        x = np.hstack((ref_time_data, vout_data))
        return x

    # dimension: (batch, input_length, 2)
    def collate_fn(self, data):
        batch_x, batch_y = [], []
        for (idx, t, v, y) in data:
            x = self.transform(idx)
            x = torch.from_numpy(x)
            y = torch.Tensor([y])
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.stack(batch_x, dim=0).float()
        batch_y = torch.cat(batch_y, dim=0).unsqueeze(1).float()
        return batch_x, batch_y


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


############################################################################
# Imbalanced regression
############################################################################
def weighted_mse_loss(yhat, y, weights=None):
    loss = (yhat - y) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(yhat, y, weights=None):
    loss = F.l1_loss(yhat, y, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(yhat, y, weights=None,
                            activate='sigmoid', beta=.2, gamma=1):
    loss = (yhat - y) ** 2
    loss *= (torch.tanh(beta * torch.abs(yhat - y))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(yhat - y)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(yhat, y, weights=None,
                           activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(yhat, y, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(yhat - y))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(yhat - y)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(yhat, y, weights=None, beta=1.):
    l1_loss = torch.abs(yhat - y)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
###########################################################################


class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(num_features=4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=4, out_channels=16,
                      kernel_size=2, stride=1, padding=2),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

            nn.Flatten(),
            nn.Linear(16*11, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=2, drop_p=0.3):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = drop_p

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers =  num_layers,
            dropout = drop_p
            )
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        outn, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(outn[:,-1,:])

        return out






###########################################################
# torch shape check
###########################################################
###########################################################
# x = torch.randn(32,1,15)
# x = nn.Conv1d(in_channels=1, out_channels=4,
#           kernel_size=3, stride=1, padding=2)(x)
# x = nn.Conv1d(in_channels=4, out_channels=16,
#           kernel_size=2, stride=1, padding=2)(x)
# x = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)(x)
# x = nn.Conv1d(in_channels=16, out_channels=16,
#           kernel_size=3, stride=1, padding=2)(x)
# x = nn.Conv1d(in_channels=16, out_channels=32,
#           kernel_size=2, stride=1, padding=4)(x)
# x = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)(x)
# x = nn.Flatten()(x)
# x = nn.Linear(32*21, 128)(x)
# x = nn.Linear(128, 32)(x)
# x = nn.Linear(32, 1)(x)
# print(x.shape)
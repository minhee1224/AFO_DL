# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 16:55:23 2022

@author: minhee
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from include.model import *
from include.utils import *


def plot_history(history, name):
    plt.suptitle("%s" % str(name))
    plt.figure(figsize=(2 * 13, 4))
    plt.subplot(1, 5, 1)
    plt.title("Training and Validation Loss")
    plt.plot(history['train_loss'], label="train_loss")
    plt.plot(history['test_loss'], label="test_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 5, 2)
    plt.title("Learning Rate")
    plt.plot(history['lr'], label="learning rate")
    plt.xlabel("iterations")
    plt.ylabel("LR")
    plt.show()


def LSTMtraining_re(num, data_dir, DEVICE, model_dir=None,
                    BATCH_SIZE=128, EPOCHS=35, TRAIN_SIZE=0.85, LR=0.00001,
                    N_WARMUP_STEPS=5, DECAY_RATE=0.98, INPUT_LENGTH=15,
                    reweight='none', lds=False, lds_kernel='gaussian',
                    lds_ks=5, lds_sigma=2):
    dataset = LSTMDataset(input_length=INPUT_LENGTH, data_dir=data_dir)

    # force_value = np.array(dataset[0:][1])
    # force_range = np.arange(5.0,70.0,5.0)
    # RANDOM_STATE = 42
    
    # # class
    # force_class = pd.cut(force_value, bins = np.arange(0.0,70.0,5.0),
    #                      labels = [str(s)
    #                                for s in \
    #                                    np.arange(0, len(np.arange(5.0,70.0,5.0)))])
    # df_force_class = pd.get_dummies(force_class)
    # df_force_class = df_force_class.values.argmax(1)
    # print(df_force_class)
    # print(min(force_value))
    # print(max(force_value))
    # figa = plt.figure(int(num))
    # ax = figa.gca()
    # ax.plot(df_force_class)
    
    # SSS = StratifiedShuffleSplit(n_splits=1, test_size=(1.0-TRAIN_SIZE),
    #                                random_state=RANDOM_STATE)
    # for train_index, test_index in SSS.split(
    #         np.arange(len(dataset)), df_force_class):
    #     pass
    # print(df_force_class)
    # print("force class")
    # print(train_index)
    # print(force_value[train_index])
    # print(test_index)
    # print(force_value[test_index])
    # train_dataset = Subset(dataset, train_index)
    # test_dataset = Subset(dataset, test_index)
    # #######################################
    # # same number data for each label
    # counter = Counter(df_force_class)
    # # min_num = counter[sorted(counter, key=counter.get)[0]]
    # label_weights = [len(df_force_class) / list(counter.values())[i]
    #                  for i in range(len(counter))]
    # train_weights = [label_weights[int(df_force_class[i])] for i in train_index]
    # test_weights = [label_weights[int(df_force_class[i])] for i in test_index]
    # train_sampler = WeightedRandomSampler(torch.DoubleTensor(train_weights),
    #                                       len(train_index))
    # test_sampler = WeightedRandomSampler(torch.DoubleTensor(test_weights),
    #                                      len(test_index))
    # # dataloader implementation
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                               sampler = train_sampler, drop_last=True,
    #                               collate_fn=dataset.collate_fn)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                              sampler = test_sampler, drop_last=True,
    #                              collate_fn=dataset.collate_fn)

    # stratified sampling
    RANDOM_STATE = 42
    data = np.array(dataset[0:][0])
    target = np.array(dataset[0:][1])
    dataset_size = len(dataset)
    train_size = int(dataset_size * TRAIN_SIZE)
    test_size = dataset_size - train_size

    x_train, x_valid, y_train, y_valid = train_test_split(data, target,
                                                          test_size=test_size,
                                                          shuffle=False,
                                                          random_state=RANDOM_STATE)
    train_dataset = LSTMDataset_data(data=x_train, target=y_train,
                                     input_length=INPUT_LENGTH)
    test_dataset = LSTMDataset_data(data=x_valid, target=y_valid,
                                    input_length=INPUT_LENGTH)

    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                         collate_fn=dataset.collate_fn, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=True,
                                  collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, drop_last=True,
                                  collate_fn=test_dataset.collate_fn)
    #########################################################################
    #########################################################################
    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    model = LSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    scheduler = ScheduleOptim(optimizer, N_WARMUP_STEPS, DECAY_RATE)
    # criterion = nn.HuberLoss()
    # criterion = RMSELoss()
    criterion = weighted_huber_loss

    patience = 0
    best_loss = 1000
    history = {'train_loss':[], 'test_loss':[], 'lr':[]}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_dataloader, scheduler,
                           epoch, criterion, DEVICE)
        test_loss = evaluate(model, test_dataloader, criterion, DEVICE)
        lr = scheduler.get_lr()
        print("\n[EPOCH: {:2d}], \tModel: LSTM, \tLR: {:8.5f}, ".format(
            epoch, lr) \
            + "\tTrain Loss: {:8.3f}, \tTest Loss: {:8.3f} \n".format(
                train_loss*100, test_loss*100))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    # LSTM model save
    if model_dir is not None:
        torch.save(model.state_dict(), model_dir)

    return test_loss, history


def LSTMtraining(data_dir, DEVICE, model_dir=None,
                BATCH_SIZE=128, EPOCHS=35, TRAIN_SIZE=0.85, LR=0.00001,
                N_WARMUP_STEPS=5, DECAY_RATE=0.98, INPUT_LENGTH=15):
    dataset = CustomLSTMDataset(input_length=INPUT_LENGTH, data_dir=data_dir)

    dataset_size = len(dataset)
    train_size = int(dataset_size * TRAIN_SIZE)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                         collate_fn=dataset.collate_fn, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  collate_fn=dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  collate_fn=dataset.collate_fn)

    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    model = LSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    scheduler = ScheduleOptim(optimizer, N_WARMUP_STEPS, DECAY_RATE)
    # criterion = nn.HuberLoss()
    criterion = RMSELoss()

    patience = 0
    best_loss = 1000
    history = {'train_loss':[], 'test_loss':[], 'lr':[]}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_dataloader, scheduler,
                           epoch, criterion, DEVICE)
        test_loss = evaluate(model, test_dataloader, criterion, DEVICE)
        lr = scheduler.get_lr()
        print("\n[EPOCH: {:2d}], \tModel: LSTM, \tLR: {:8.5f}, ".format(
            epoch, lr) \
            + "\tTrain Loss: {:8.3f}, \tTest Loss: {:8.3f} \n".format(
                train_loss, test_loss))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    # LSTM model save
    if model_dir is not None:
        torch.save(model.state_dict(), model_dir)

    return test_loss, history


def CNNtraining(data_dir, DEVICE, model_dir=None,
                BATCH_SIZE=128, EPOCHS=35, TRAIN_SIZE=0.85, LR=0.00001,
                N_WARMUP_STEPS=5, DECAY_RATE=0.98, INPUT_LENGTH=15):
    dataset = CustomDataset(input_length=INPUT_LENGTH, data_dir=data_dir)

    dataset_size = len(dataset)
    train_size = int(dataset_size * TRAIN_SIZE)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                         collate_fn=dataset.collate_fn, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  collate_fn=dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=True,
                                  collate_fn=dataset.collate_fn)

    DEVICE = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(
        torch.__version__, DEVICE))

    model = Conv1DNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    scheduler = ScheduleOptim(optimizer, N_WARMUP_STEPS, DECAY_RATE)
    # criterion = nn.HuberLoss()
    criterion = RMSELoss()

    patience = 0
    best_loss = 1000
    history = {'train_loss':[], 'test_loss':[], 'lr':[]}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_dataloader, scheduler,
                           epoch, criterion, DEVICE)
        test_loss = evaluate(model, test_dataloader, criterion, DEVICE)
        lr = scheduler.get_lr()
        print("\n[EPOCH: {:2d}], \tModel: Conv1DNet, \tLR: {:8.5f}, ".format(
            epoch, lr) \
            + "\tTrain Loss: {:8.3f}, \tTest Loss: {:8.3f} \n".format(
                train_loss, test_loss))

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(lr)

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    # CNN model save
    if model_dir is not None:
        torch.save(model.state_dict(), model_dir)

    return test_loss, history

# hyperparameter settings
BATCH_SIZE = 32
EPOCHS = 300
TRAIN_SIZE = 0.85
LR = 0.0001
N_WARMUP_STEPS = 10
DECAY_RATE = 0.98
INPUT_LENGTH = 15 #1, 3, 5, 10, 13, 15

# device setting
DEVICE = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')
print("Using PyTorch version: {}, Device: {}".format(
    torch.__version__, DEVICE))

# define calib data path and model path
calib_data_path = "./CHAR_1114_260/Calibration/"
model_path = "./CHAR_1114_260/LSTM_model_balanced/"
try:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
except:
    pass

# for loop for 12 sensor training
_, sensor_name_list = folder_path_name(calib_data_path)

df_loss = pd.DataFrame(columns=["Date", "sensor_name", "LR", "loss"])
for LR in [0.0001]:
    # [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    num = 0
    for n, name in enumerate(sensor_name_list):
        
        # LR_list = [0.001,0.001,0.001,0.001,0.001,0.001,
        #             0.001,0.001,0.001,0.001,0.001,0.001]
        # LR = LR_list[n]
        
        # for lr in [LR-LR/5, LR-LR/10, LR, LR+LR/10, LR+LR/5]:
        for lr in [LR]:
            # if (name.endswith("Right_4") == 1) | (name.endswith("Left_3") == 1):
            print("START LSTM MODEL TRAINING!!! %s" % name)
            print("LR: %f" % lr)
            num += 1
            data_dir = calib_data_path + name + "/force_conversion_test.csv"
            model_dir = model_path + name + "LR%s.pt" % lr
            final_test_loss, history = LSTMtraining(data_dir, DEVICE,
                                          model_dir=model_dir,
                                          BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS,
                                          LR=lr, N_WARMUP_STEPS=N_WARMUP_STEPS,
                                          DECAY_RATE=DECAY_RATE,
                                          INPUT_LENGTH=INPUT_LENGTH)
            df_loss = pd.concat([df_loss, pd.DataFrame([
                {'Date': "221114_260_LSTM", 'sensor_name':name,
                 'LR':lr, 'loss':final_test_loss}])], ignore_index=True)
            plot_history(history, "221114_LSTM" + name)
            # else:
            #     pass

# define calib data path and model path
# calib_data_path = "./CHAR_1010_280/Calibration/"
# model_path = "./CHAR_1010_280/CNN_model/"
# try:
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
# except:
#     pass

# # for loop for 12 sensor training
# _, sensor_name_list = folder_path_name(calib_data_path)

# df_loss = pd.DataFrame(columns=["Date", "sensor_name", "LR", "loss"])
# for LR in [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:

#     for n, name in enumerate(sensor_name_list):
        
#         # LR_list = [0.001,0.001,0.001,0.001,0.001,0.001,
#         #             0.001,0.001,0.001,0.001,0.001,0.001]
#         # LR = LR_list[n]
        
#         # for lr in [LR-LR/5, LR-LR/10, LR, LR+LR/10, LR+LR/5]:
#         for lr in [LR-LR/5, LR-LR/10, LR, LR+LR/10, LR+LR/5]:
#         # if (name.endswith("Right_4") == 1) | (name.endswith("Left_3") == 1):
#             print("START 1D CNN MODEL TRAINING!!! %s" % name)
#             print("LR: %f" % lr)
#             data_dir = calib_data_path + name + "/force_conversion_test.csv"
#             model_dir = model_path + name + "LR%s.pt" % lr
#             final_test_loss, history = CNNtraining(data_dir, DEVICE,
#                                           model_dir=model_dir,
#                                           BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS,
#                                           LR=lr, N_WARMUP_STEPS=N_WARMUP_STEPS,
#                                           DECAY_RATE=DECAY_RATE,
#                                           INPUT_LENGTH=INPUT_LENGTH)
#             df_loss = df_loss.append({'Date': "221010_280_CNN", 'sensor_name':name,
#                                       'LR':lr, 'loss':final_test_loss},
#                             ignore_index=True)
#             plot_history(history, "221010_CNN" + name)
        # else:
        #     pass
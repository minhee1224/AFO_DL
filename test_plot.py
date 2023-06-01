# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:21:51 2022

@author: mleem
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statistics
import math
import operator

def basic_plotting(time_data, target_data):
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(time_data, target_data.loc[:, "Heel"])
    axs[0, 0].set_title("Heel")
    axs[0, 1].plot(time_data, target_data.loc[:, "Fifth metatarsal head"],
                   'tab:orange')
    axs[0, 1].set_title("Fifth metatarsal head")
    axs[1, 0].plot(time_data, target_data.loc[:, "Third metatarsal head"],
                   'tab:green')
    axs[1, 0].set_title("Third metatarsal head")
    axs[1, 1].plot(time_data, target_data.loc[:, "First metatarsal head"],
                   'tab:red')
    axs[1, 1].set_title("First metatarsal head")
    axs[2, 0].plot(time_data, target_data.loc[:, "First toe"], 'tab:purple')
    axs[2, 0].set_title("First toe")
    axs[2, 1].plot(time_data, target_data.loc[:, "Second toe"], 'tab:brown')
    axs[2, 1].set_title("Second toe")

    for ax in axs.flat:
        ax.set(xlabel='time', ylabel='Force [N]')
    fig.tight_layout()


path_CNN = "D:/OneDrive - SNU/Projects/pytorch112_cpu/DATA/RH-12/ref/CNN.csv"
path_LSTM = "D:/OneDrive - SNU/Projects/pytorch112_cpu/DATA/RH-12/ref/LSTM.csv"

col = ["time",
       "Fifth metatarsal head", "Second toe",
       "Third metatarsal head", "First toe",
       "First metatarsal head", "Heel"]

CNN_data = pd.read_csv(path_CNN)
CNN_data = CNN_data.iloc[:1000, 2:]
LSTM_data = pd.read_csv(path_LSTM)
LSTM_data = LSTM_data.iloc[:1000, 2:]

CNN_data.columns = col
LSTM_data.columns = col

# basic_plotting(CNN_data.loc[:, "time"], CNN_data.iloc[:, 1:])
basic_plotting(LSTM_data.loc[:, "time"], LSTM_data.iloc[:, 1:])
# 
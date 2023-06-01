# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:00:11 2022

@author: minhee
"""

import os
import math
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# from sklearn.metrics import mean_squared_error

from include.model import *
from include.utils import *


# input size = required length*6
# required length만큼 읽어온다고 가정
# output size = 1*6 (2D array)
class dataPredictor:
    def __init__(self, data, model_name="LSTM",
                 model_dir="./CHAR_1114_260/",
                 sensor_dir="Left", sensor_size="260",
                 input_length=15, sensor_num=6):
        self.sensor_num = sensor_num
        self.sensor_dir = sensor_dir
        self.sensor_size = sensor_size
        self.model_name = model_name
        self.model_path = model_dir
        self.total_data = data
        self.input_length = input_length
        # self.data_load()
        self.device = torch.device('cpu')
        self.model_load()

    # def data_load(self):
    #     data_path = self.data_path + "450Right_4/force_conversion_test.csv"
    #     self.data = np.loadtxt(data_path, delimiter=",")
    def data_indexing(self, idx):
        if idx <= len(self.total_data)-2:
            self.data = self.total_data[0:idx+1][:]
        else:
            self.data = np.zeros((1,3))

    def model_load(self):
        if self.model_name == "CNN":
            model = Conv1DNet()
        elif self.model_name == "LSTM":
            model = LSTM()
        else:
            pass
        model.load_state_dict(torch.load(
            self.model_path + self.model_name + "_model_balanced/" +
            self.sensor_size + self.sensor_dir + "_1LR0.0001.pt",
            map_location=self.device))
        self.model = model

    def LSTMtransform(self):
        if len(self.data) < self.input_length:
            # Buffer에 데이터가 없어서 required length만큼 읽어오지 못했을 경우
            # 0열을 부족한만큼 복사
            extra_row = np.repeat([self.data[0]], repeats=
                                  self.input_length - len(self.data), axis=0)
            input_data = np.concatenate((extra_row, self.data), axis=0)
            x = input_data[
                (len(input_data) - self.input_length) :
                    len(input_data),
                    1]
            # print(x)
        else:
            input_data = self.data
            x = input_data[
                (len(self.data) - self.input_length) :
                    len(self.data),
                    1]

        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = torch.stack([x], dim=0).float()
        # print(x)
        # print(x.shape)
        
        # if (int(idx) + 1) < self.input_length:
        #     x = np.append(self.data[0, 1] *
        #                   np.ones((1, self.input_length - int(idx) - 1)),
        #                   self.data[0:int(idx) + 1, 1])
        # else:
        #     x = self.data[int(idx) + 1 - self.input_length:int(idx) + 1, 1]
        # return x

        return x

    def prediction(self):
        output = np.array([])

        model = self.model
        model.eval()
        with torch.no_grad():
            start = time.time()
            for num in np.arange(0, len(self.total_data)):
                self.data_indexing(num)
                if self.model_name == "CNN":
                    x = self.CNNtransform()
                elif self.model_name == "LSTM":
                    x = self.LSTMtransform()
                else:
                    pass
                output = np.append(output, model(x))
            end = time.time()
            data_num = len(self.total_data)

        freq_output = 1/((end - start)/data_num)
        df_output = pd.DataFrame(columns=["Prediction", "True"])
        df_output["Prediction"] = pd.Series(output)
        df_output["True"] = pd.Series(self.total_data[:, 2])
        df_output.to_csv(self.model_path +
                         "prediction_test_left1_b.csv",
                          sep=",",
                          columns=["Prediction", "True"])

        return df_output, freq_output


if __name__ == "__main__":

    trial_num_list = [15]

    model_list = ["LSTM"]

    calib_dict = {"11": ["CHAR_0919_280", "280"],
                  "12": ["CHAR_0927_260", "260"],
                  "13": ["CHAR_1004_280", "280"],
                  "14": ["CHAR_1010_280", "280"],
                  "15": ["CHAR_1114_260", "260"],
                  "16": ["CHAR_1121_260", "260"]}

    for model_name in model_list:
        print("START %s model prediction!" % model_name)
        # for loop for trial num
        for trial_num in trial_num_list:
            print("Current trial num: %s" % str(trial_num).zfill(2))

            # calibration path, sensor size
            calibration_path = calib_dict[str(trial_num).zfill(2)][0]
            calibration_size = calib_dict[str(trial_num).zfill(2)][1]

            test_data_path = "./%s/Calibration/260Left_1/force_conversion_test.csv" \
                % str(calibration_path)
            test_data = np.loadtxt(test_data_path, delimiter=",", skiprows=1)

            left_predictor = dataPredictor(
                test_data,
                model_name = model_name,
                model_dir="./"+calibration_path+"/",
                sensor_size=calibration_size)
            df_output, freq_output = left_predictor.prediction()

            loss_output = sqrt(torch.mean(
                (torch.Tensor(df_output["True"]) - torch.Tensor(df_output["Prediction"])**2)))

    # right_predictor = dataPredictor(sensor_dir="Right")
    # df4_450, freq4_450 = right_predictor.prediction()
    
   
    # loss4_450 = sqrt(mean_squared_error(df4_450["True"], df4_450["Prediction"]))

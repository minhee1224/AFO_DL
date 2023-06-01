# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 01:38:40 2022

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

from sklearn.metrics import mean_squared_error

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
            self.data = np.zeros((1,6))

    def model_load(self):
        self.model = np.array([])
        for num in np.arange(self.sensor_num):
            if self.model_name == "CNN":
                model = Conv1DNet()
            elif self.model_name == "LSTM":
                model = LSTM()
            else:
                pass
            model.load_state_dict(torch.load(
                self.model_path + self.model_name + "_model/" +
                self.sensor_size + self.sensor_dir + "_" +
                str(num + 1) + ".pt",
                map_location=self.device))
            self.model = np.append(self.model, model)

    def LSTMtransform(self, idx):
        if len(self.data) < self.input_length:
            # Buffer에 데이터가 없어서 required length만큼 읽어오지 못했을 경우
            # 0열을 부족한만큼 복사
            extra_row = np.repeat([self.data[0]], repeats=
                                  self.input_length - len(self.data), axis=0)
            input_data = np.concatenate((extra_row, self.data), axis=0)
            x = input_data[(len(self.data) - self.input_length) : len(self.data),
                           (idx - 1)]
        else:
            input_data = self.data
            x = input_data[(len(self.data) - self.input_length) : len(self.data),
                           (idx - 1)]

        x = torch.from_numpy(x)
        x = x.unsqueeze(1)
        x = torch.stack([x], dim=0).float()

        return x

    def prediction(self, num):
        self.data_indexing(num)
        _, sensor_name_list = folder_path_name(
            self.model_path + self.model_name +
            "_model/", "include", self.sensor_dir)
        sensor_name_list = [name for name in sensor_name_list if \
                            int(name[-4]) <= self.sensor_num]
        sorted_name_list = sorted(sensor_name_list, key=lambda x: int(x[-4]),
                                  reverse=False)
        output = np.array([])

        for name in sorted_name_list:

            model = self.model[int(name[-4]) - 1]
            model.eval()
            with torch.no_grad():
                # start = time.time()
                if self.model_name == "CNN":
                    x = self.CNNtransform(int(name[-4]))
                elif self.model_name == "LSTM":
                    x = self.LSTMtransform(int(name[-4]))
                else:
                    pass
                output = np.append(output, model(x))
                # end = time.time()
                # data_num = len(self.data)


        # freq_output = 1/((end - start)/data_num)
        # df_output = pd.DataFrame(columns=["Prediction", "True"])
        # df_output["Prediction"] = pd.Series(output)
        # df_output["True"] = pd.Series(self.data[:, 3])
        # df_output.to_csv(self.model_path + "prediction_450um.csv",
        #                  sep=",", columns=["Prediction", "True"])

        return np.expand_dims(output, axis=0)


if __name__ == "__main__":

    trial_num_list = [15]

    model_list = ["LSTM"]

    test_index_list = ["main_"]

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
            # for loop for test index
            for test_index in test_index_list:
                print("Current test index: %s" % test_index)

                clinical_data_path = "./DATA/RH-%s/sensor/" \
                    % str(trial_num).zfill(2)
                prediction_path = "./DATA/RH-%s/force/%s/" \
                    % (str(trial_num).zfill(2), model_name)
                try:
                    if not os.path.exists(prediction_path):
                        os.makedirs(prediction_path)
                except:
                    pass

                # calibration path, sensor size
                calibration_path = calib_dict[str(trial_num).zfill(2)][0]
                calibration_size = calib_dict[str(trial_num).zfill(2)][1]

                # sensor path assignment
                (sensor_path, _) = folder_path_name(
                    clinical_data_path, "start", test_index, 1)
                left_sensor_path = [path for path in sensor_path 
                                    if path.endswith("Leftsole.csv")==1]
                right_sensor_path = [path for path in sensor_path 
                                    if path.endswith("Rightsole.csv")==1]

                # Exception in test index
                if len(left_sensor_path) != 1:
                    pass
                else:
                    # left sensor replay
                    left_data, left_data_front, left_header = vout_preprocessing(
                        left_sensor_path[0])
                    left_predictor = dataPredictor(
                        left_data,
                        model_name = model_name,
                        model_dir="./"+calibration_path+"/",
                        sensor_size=calibration_size)
                    left_force_data = left_predictor.prediction(num=0)
                    left_idx = 1
                    for _ in np.arange(1, len(left_predictor.total_data)):
                        left_force_data = np.append(
                            left_force_data,
                            left_predictor.prediction(num=left_idx),
                            axis=0)
                        left_idx += 1

                    # right sensor replay
                    right_data, right_data_front, right_header = vout_preprocessing(
                        right_sensor_path[0])
                    right_predictor = dataPredictor(
                        right_data,
                        model_name = model_name,
                        model_dir="./"+calibration_path+"/",
                        sensor_size=calibration_size,
                        sensor_dir="Right")
                    right_force_data = right_predictor.prediction(num=0)
                    right_idx = 1
                    for _ in np.arange(1, len(right_predictor.total_data)):
                        right_force_data = np.append(
                            right_force_data,
                            right_predictor.prediction(num=right_idx),
                            axis=0)
                        right_idx += 1

                    # save sensor data csv files
                    left_data_front = pd.DataFrame(left_data_front)
                    left_force_data = pd.DataFrame(left_force_data)
                    left_force_data = pd.concat(
                        [left_data_front, left_force_data], axis=1)
                    # left_vout_data = pd.concat(
                    #     [left_data_front, pd.DataFrame(left_data)], axis=1)
                    left_force_data.columns = left_header
                    # left_vout_data.columns = left_header
                    # for v_col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
                    #     v_col = str(v_col)
                    #     v_min = 0.0
                    #     v_min = min(left_vout_data[v_col])
                    #     print("L")
                    #     print(v_min)
                    #     left_vout_data[v_col] -= v_min
                    #     left_vout_data[v_col] /= v_min
                    left_force_data.to_csv(
                        prediction_path+"%sLeftsole.csv" % (test_index),
                        sep=",", header=True, index=False)
                    # left_vout_data.to_csv(
                    #     prediction_path+"%s_vout_Leftsole.csv" % (test_index),
                    #     sep=",", header=True, index=False)
                
                    right_data_front = pd.DataFrame(right_data_front)
                    right_force_data = pd.DataFrame(right_force_data)
                    right_force_data = pd.concat(
                        [right_data_front, right_force_data], axis=1)
                    # right_vout_data = pd.concat(
                    #     [right_data_front, pd.DataFrame(right_data)], axis=1)
                    right_force_data.columns = right_header
                    # right_vout_data.columns = right_header
                    # for v_col in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
                    #     v_col = str(v_col)
                    #     v_min = 0.0
                    #     v_min = min(right_vout_data[v_col])
                    #     print("R")
                    #     print(v_min)
                    #     right_vout_data[v_col] -= v_min
                    #     right_vout_data[v_col] /= v_min
                    right_force_data.to_csv(
                        prediction_path+"%sRightsole.csv" % (test_index),
                        sep=",", header=True, index=False)
                    # right_vout_data.to_csv(
                    #     prediction_path+"%s_vout_Rightsole.csv" % (test_index),
                    #     sep=",", header=True, index=False)

    # right_predictor = dataPredictor(sensor_dir="Right")
    # df4_450, freq4_450 = right_predictor.prediction()
    
   
    # loss4_450 = sqrt(mean_squared_error(df4_450["True"], df4_450["Prediction"]))
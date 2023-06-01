# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:35:03 2022

@author: minhee
"""

import glob
import os
import numpy as np
import pandas as pd

def folder_path_name(path, option=None, char=None, T_F=None):

    folder_name_path = str(path)
    folder_path = folder_name_path + "*"

    file_list = glob.glob(folder_path)
    file_name_list = os.listdir(folder_name_path)

    if option == "start":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.startswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name) in
                         zip(file_list, file_name_list)
                         if name.startswith(str(char)) == int(T_F)]

    elif option == "end":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.endswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if name.endswith(str(char)) == int(T_F)]
        
    elif option == "include":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if str(char) in name]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if str(char) in name]

    else:
        exp_list = [file for (file, name) in zip(file_list, file_name_list)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)]

    exp_list = [file.replace('\\', '/') for file in exp_list]

    return exp_list, exp_name_list


def replace_multiple_text(text, dic={",": " ", "  ": " "}):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def vout_preprocessing(path):

    sensor_csv = path

    # RH-11 exception
    if "RH-11" in sensor_csv:
        dict_ = {",": " ", "  ": " ", "(": "", ")": "",
                 "False": "0", "True": "1"}
    else:
        dict_ = {",": " ", "  ": " "}

    # Multiple delimiter
    with open(sensor_csv) as data:
        conv_data = [replace_multiple_text(line, dict_) for line in data]

    sensor_vout_header = ["swing_phase", "sync", "time",
                      "v1" ,"v2", "v3", "v4", "v5", "v6"]
    sensor_force_header = ["swing_phase", "sync", "time",
                      "f1" ,"f2", "f3", "f4", "f5", "f6"]

    # RH-11 exception data loading
    if "RH-11" in sensor_csv:
        data_pre = np.loadtxt(conv_data, delimiter=" ")
        sensor_vout_header_pre = ["time", "sync", "swing_phase",
                                  "v1" ,"v2", "v3", "v4", "v5", "v6"]
        data_1 = pd.DataFrame(data_pre, columns=sensor_vout_header_pre)
        data_ = data_1[sensor_vout_header]
        sensor_data = np.array(data_)
    else:
        sensor_data = np.loadtxt(conv_data, delimiter=" ", skiprows=1)

    data_buffer = sensor_data[:,3:]
    data_front = sensor_data[:,:3]
    return data_buffer, data_front, sensor_force_header

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 22:25:26 2022

@author: mleem
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import joblib
import re


prop_title = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=30
)

prop_label = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=25
)

prop_tick = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=20
)

prop_legend = dict(
    family='Times New Roman',
    style='normal',
    weight='bold',
    size=15
)


def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)

    return np.where((data > upper_bound) | (data < lower_bound))


def folder_path_name(path, start_or_end=None, char=None, T_F=None):

    folder_name_path = str(path)
    folder_path = folder_name_path + "*"

    file_list = glob.glob(folder_path)
    file_name_list = os.listdir(folder_name_path)

    if start_or_end == "start":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.startswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name) in
                         zip(file_list, file_name_list)
                         if name.startswith(str(char)) == int(T_F)]

    elif start_or_end == "end":
        exp_list = [file for (file, name) in zip(file_list, file_name_list)
                    if name.endswith(str(char)) == int(T_F)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)
                         if name.endswith(str(char)) == int(T_F)]

    else:
        exp_list = [file for (file, name) in zip(file_list, file_name_list)]
        exp_name_list = [name for (file, name)
                         in zip(file_list, file_name_list)]

    exp_list = [file.replace('\\', '/') for file in exp_list]

    return exp_list, exp_name_list


def instron_interp_preprocessing(instron_path, save_path):

    instron_csv = instron_path
    instron_data = pd.read_csv(instron_csv, header=0)
    instron_data = pd.DataFrame.drop(instron_data, index=0, axis=0)
    instron_data.reset_index(drop=True, inplace=True)
    instron_data = instron_data.astype(float)
    ini_vol = instron_data.at[0, 'voltage_1']
    sync = np.where(abs(instron_data["voltage_1"] - ini_vol) >= 0.03)
    sync_index = sync[0][0]
    sync_time = instron_data.at[sync_index, 'Time']
    instron_data.to_csv(str(save_path), sep=",", index=False)

    return sync_time


def raspi_interp_preprocessing(raspi_path, sensor_num, save_path):

    sensor_csv = raspi_path
    sensor_data = pd.read_csv(sensor_csv, sep=" |,", header=1)
    # sensor_data = sensor_data.astype(float)
    try:
        # sensor_data.iloc[:, 10]
        sensor_data_ = sensor_data.iloc[:, [2, 4, int(sensor_num)+5]]
        pass
    except KeyError:
        pass
        # if sensor_data.iloc[:, 2].isnull().sum().iloc[:, 2] > 100:
        #     sensor_data = sensor_data.iloc[:, [0, 1, int(sensor_num)+2]]
        #     pass
        # else:
        #     sensor_data = sensor_data.iloc[:, [1, 2, int(sensor_num)+3]]
        #     pass
    else:
        sensor_data_ = sensor_data.iloc[:, [2, 4, int(sensor_num)+5]]
        pass
    sensor_data = sensor_data_
    sensor_data.columns = ['sync', 'time', 'vout']
    ini_sync = sensor_data.at[0, 'sync']
    sync = np.where(abs(sensor_data["sync"] - ini_sync) >= 1)
    sync_index = sync[0]
    sensor_data = sensor_data.iloc[sync_index, :]
    sensor_data.reset_index(inplace=True, drop=True)
    print(sensor_data)
    ini_time = sensor_data.at[0, 'time']
    sensor_index = list(sensor_data.index)
    del_time = []
    for j in sensor_index:
        if j == 0:
            del_time = np.append(del_time, sensor_data.at[j, 'time']-ini_time)
        else:
            del_time = np.append(del_time,
                                 sensor_data.at[j, 'time']
                                 - sensor_data.at[j-1, 'time'])
    d_time = pd.DataFrame(del_time, columns=["flag"])
    d_time = d_time[d_time["flag"] > 0.003]
    sensor_time_index = list(d_time.index)[0]
    sensor_data = sensor_data[sensor_data.index >= sensor_time_index]
    sensor_data["time"] = sensor_data["time"] - ini_time

    # only for CHAR_0927_260
    # sensor_data["vout"] = sensor_data["vout"] * (0.625/0.5)
    sensor_data.to_csv(str(save_path), sep=",", index=False)


def calib_interp(instron_path, instron_sync_time, sensor_path, save_path):

    instron_data = pd.read_csv(str(instron_path), header=0)
    instron_data = instron_data.astype(float)
    instron_synced = instron_data[(abs(instron_data["voltage_1"] -
                                       instron_data.at[0, "voltage_1"]) > 3.0)
                                  & (instron_data["Time"]
                                     >= instron_sync_time)]
    instron_synced["Time"] = instron_synced["Time"] - instron_sync_time
    instron_synced.reset_index(drop=True, inplace=True)

    sensor_data = pd.read_csv(str(sensor_path), header=0)
    sensor_data = sensor_data.astype(float)
    sensor_synced = sensor_data[abs(sensor_data["sync"] -
                                    sensor_data.at[0, "sync"]) == 0]
    sensor_synced.reset_index(drop=True, inplace=True)

    vout_synced = pd.DataFrame(np.interp(instron_synced["Time"],
                                         sensor_synced["time"],
                                         sensor_synced["vout"]),
                                columns=["vout"])
    calib_synced = pd.concat([instron_synced["Time"],
                              vout_synced, instron_synced["Force"]], axis=1)
    calib_synced.columns = ["time", "vout", "force"]
    calib_synced.to_csv(str(save_path), sep=",", index=False)


def calib_result_plot(path, RH_num, folder_name):

    data = pd.read_csv(str(path), delimiter=",", header=0)

    plt.figure(figsize=(6, 8))
    plt.plot(data.loc[:, "time"], data.loc[:, "vout"],
             label='%s_%s' % (str(RH_num), str(folder_name)))
    plt.title("time vs vout")
    plt.xlabel("Time [s]")
    plt.ylabel("vout [V]")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 8))
    plt.plot(data.loc[:, "time"], data.loc[:, "force"],
             label='%s_%s' % (str(RH_num), str(folder_name)))
    plt.title("time vs force")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 8))
    plt.plot(data.loc[:, "vout"], data.loc[:, "force"],
             label='%s_%s' % (str(RH_num), str(folder_name)))
    plt.title("vout vs force")
    plt.ylabel("Force [N]")
    plt.xlabel("vout [V]")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def N_data_preprocessing(data, NUM_PRE, WINDOWS, tol):

    data.sort_values(by=['time'], axis=0, ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)

    time_tmp = data.time.values.tolist()
    time_tmp = np.insert(time_tmp, 0, np.nan)
    time_tmp = np.delete(time_tmp, -1)

    data["pre_time"] = time_tmp

    for h in np.arange(1, NUM_PRE+1, 1):
        tmp = data.vout.values.tolist()
        for p in np.arange(0, h, 1):
            tmp = np.insert(tmp, int(p), np.nan)
            tmp = np.delete(tmp, -1)

        data["pre_%s" % (str(h))] = tmp

    data = data.astype(float)

    data["del_V"] = (data["vout"] - data["pre_1"])
    data["del_V"] = data["del_V"].rolling(window=WINDOWS,
                                          min_periods=1, center=True).mean()

    data["del_time"] = (data["time"] - data["pre_time"])

    data["loading_type"] = 0.0

    for q in np.arange(1, NUM_PRE+1, 1):
        data["loading_type"] += data["pre_%s" % (str(q))]

    data["loading_type"] = (data["loading_type"])/float(len(
        np.arange(1, NUM_PRE+1, 1)))
    data["loading_type"] -= data["vout"]
    data["loading_type"] = data["loading_type"].rolling(
        window=WINDOWS, min_periods=1, center=True).mean()

    loading_index1 = []
    loading_index1 = data[abs(data["loading_type"]) > tol].index
    # data["loading_type"] = np.sign(-data["loading_type"])

    data["loading_type1"] = 0.0
    data.loc[loading_index1, "loading_type1"] = np.sign(-data.loc[
        loading_index1, "loading_type"])

    delV_index1 = []
    delV_index1 = data[abs(data["del_V"]) > tol].index

    data["loading_type2"] = 0.0
    data.loc[delV_index1, "loading_type2"] = np.sign(-data.loc[
        delV_index1, "del_V"])

    delT_index1_1 = []
    delT_index1_1 = data[data["loading_type1"] != 0.0].index

    data["del_time1"] = 0.0
    data.loc[delT_index1_1, "del_time1"] = data.loc[delT_index1_1, "del_time"]

    data["elapsed_time1"] = np.cumsum(data["del_time1"])

    delT_index1_2 = []
    delT_index1_2 = data[data["loading_type2"] != 0.0].index

    data["del_time2"] = 0.0
    data.loc[delT_index1_2, "del_time2"] = data.loc[delT_index1_2, "del_time"]

    data["elapsed_time2"] = np.cumsum(data["del_time2"])

    data = data.dropna(axis=0)
    data.reset_index(drop=True, inplace=True)

    return data


def clinical_force_data_reading(path, name, save_path, RH_num):

    # raw data reading
    force_data = pd.read_csv(path, delimiter="\t", header=4)
    force_data = pd.DataFrame.drop(force_data, index=[0, 1], axis=0)
    force_data.reset_index(drop=True, inplace=True)
    force_data = force_data.astype(float)

    force1_data = force_data[["F1X1", "F1X3", "F1Y1", "F1Y2", "F1Z1",
                              "F1Z2", "F1Z3", "F1Z4"]]
    force2_data = force_data[["F2X1", "F2X3", "F2Y1", "F2Y2", "F2Z1",
                              "F2Z2", "F2Z3", "F2Z4"]]
    time_data = force_data[["Name"]]
    sync_data = force_data[["Sync"]]

    # raw data plotting
    fig1 = plt.figure(figsize=(6, 8))
    fig1.suptitle("%s" % (name), fontsize=15)

    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time_data, force1_data)
    plt.title("Force1 raw data")
    ax1.legend(list(force1_data.columns))

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(time_data, force2_data)
    plt.title("Force2 raw data")
    ax2.legend(list(force2_data.columns))

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(time_data, sync_data)
    plt.title("Sync data")
    ax3.legend("Sync")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    try:
        if not os.path.exists(str(save_path)):
            os.makedirs(str(save_path))
    except OSError:
        pass
    plt.savefig(str(save_path)+"/RH-%s_%s_raw.png"
                % (str(RH_num), name))

    return force1_data, force2_data, time_data, sync_data


def sync_for_clinical_data(sync_data, sync_condition="large"):

    if sync_condition == "small":
        synced = sync_data[sync_data[["Sync"]] < 500]
    else:
        synced = sync_data[sync_data[["Sync"]] > 500]
    synced.dropna(inplace=True)
    tmp_index = list(synced.index)

    start_index = tmp_index[0]

    tmp_index_synced = [float(a) for a in tmp_index]
    tmp = np.insert(tmp_index_synced, 0, np.nan, axis=0)
    tmp = np.delete(tmp, -1)

    tmp_index_synced = pd.DataFrame(tmp_index_synced,
                                    columns=["sync_index"])
    tmp_index_synced["sync_pre_index"] = tmp
    tmp_index_synced = tmp_index_synced.astype(float)

    tmp_index_synced["del_sync"] = (tmp_index_synced[
        "sync_index"] - tmp_index_synced["sync_pre_index"])
    tmp_index_synced = tmp_index_synced.dropna(axis=0)

    d_index = tmp_index_synced[tmp_index_synced[
        "del_sync"] > 1.0]

    if len(d_index) == 0:
        end_index = tmp_index[-1]
    else:
        end_index = d_index["sync_pre_index"].values.tolist()
        end_index = [int(a) for a in end_index]
        end_index = end_index[0]

    synced_index = np.arange(start_index, end_index+1, 1)

    return synced_index


def synced_data_plot(data1, data2, time_data, synced_data, synced_index,
                     RH_num, name, save_path, csv_path):

    data1_synced = data1.loc[synced_index]
    data2_synced = data2.loc[synced_index]
    time_data_synced = time_data.loc[synced_index]

    fig = plt.figure(figsize=(6, 5))
    fig.suptitle("%s_synced" % (name), fontsize=15)

    bx1 = plt.subplot(2, 1, 1)
    plt.plot(time_data_synced, data1_synced)
    plt.title("Synced raw data1")
    bx1.legend(list(data1_synced.columns))

    bx2 = plt.subplot(2, 1, 2, sharex=bx1)
    plt.plot(time_data_synced, data2_synced)
    plt.title("Synced raw data2")
    bx2.legend(list(data2_synced.columns))

    plt.tight_layout()
    directory = str(save_path)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        pass

    plt.savefig(str(directory)+"/RH-%s_%s_synced_FORCE(raw).png"
                % (str(RH_num), name))

    csv_directory = str(csv_path)

    try:
        if not os.path.exists(csv_directory):
            os.makedirs(csv_directory)
    except OSError:
        pass

    data_synced = pd.concat([time_data_synced, synced_data], axis=1)
    data_synced = pd.concat([data_synced, data1_synced], axis=1)
    data_synced = pd.concat([data_synced, data2_synced], axis=1)
    data_synced.to_csv(str(csv_directory) +
                       "/RH-%s_%s_synced_FORCE(raw).csv" % (
                           str(RH_num), name), header=True, index=True,
                       sep=',')


def COP_data_read_plot(path, synced, synced_index, time_data, name, RH_num,
                       save_path, csv_path):

    COP_data = pd.read_csv(path, delimiter="\t", header=4)
    COP_data.reset_index(drop=True, inplace=True)
    COP_data = COP_data.astype(float)

    COP1_data = COP_data[["FX1", "FY1", "FZ1", "X1", "Y1", "Z1", "MZ1"]]
    COP2_data = COP_data[["FX2", "FY2", "FZ2", "X2", "Y2", "Z2", "MZ2"]]

    #########################################################
    # COP synced data plotting
    time_data_synced = time_data.loc[synced_index]
    COP1_data_synced = COP1_data.loc[synced_index]
    COP2_data_synced = COP2_data.loc[synced_index]

    fig = plt.figure(figsize=(6, 5))
    fig.suptitle("%s_synced" % (name), fontsize=15)

    bx1 = plt.subplot(2, 1, 1)
    plt.plot(time_data_synced, COP1_data_synced)
    plt.title("Force1 processed data")
    bx1.legend(list(COP1_data_synced.columns))

    bx2 = plt.subplot(2, 1, 2, sharex=bx1)
    plt.plot(time_data_synced, COP2_data_synced)
    plt.title("Force2 processed data")
    bx2.legend(list(COP2_data_synced.columns))

    plt.tight_layout()
    COP_directory = str(save_path)
    try:
        if not os.path.exists(COP_directory):
            os.makedirs(COP_directory)
    except OSError:
        pass

    plt.savefig(str(COP_directory) +
                "/RH-%s_%s_synced_FORCE(processed).png"
                % (str(RH_num), name))

    csv_COP_directory = str(csv_path)
    try:
        if not os.path.exists(csv_COP_directory):
            os.makedirs(csv_COP_directory)
    except OSError:
        pass

    COP_synced = pd.concat([time_data_synced, synced], axis=1)
    COP_synced = pd.concat([COP_synced, COP1_data_synced], axis=1)
    COP_synced = pd.concat([COP_synced, COP2_data_synced], axis=1)
    # add del_time column
    ini_COPtime = COP_synced.iloc[0, 0]
    COP_synced["del_time"] = COP_synced["Name"] - float(ini_COPtime)
    COP_synced.to_csv(str(csv_COP_directory) +
                      "/RH-%s_%s_synced_FORCE(processed).csv"
                      % (str(RH_num), name), header=True,
                      index=False, sep=',')


def GRF_data_read_plot(path, name, RH_num, save_path, csv_path):

    GRF_data = pd.read_csv(path, delimiter="\t", header=23)
    GRF_data = GRF_data.astype(float)

    index_data = list(GRF_data.index)
    df_index = pd.DataFrame(index_data, columns=["index data"])
    df_index = (df_index * 5) / 600
    R_GRF_data = GRF_data[["R_GRF_FWD", "R_GRF_LAT", "R_GRF_VRT"]]
    L_GRF_data = GRF_data[["L_GRF_FWD", "L_GRF_LAT", "L_GRF_VRT"]]
    ##############################################################
    # GRF data plotting
    fig = plt.figure(figsize=(6, 5))
    fig.suptitle("%s" % (name), fontsize=15)

    cx1 = plt.subplot(2, 1, 1)
    plt.plot(df_index["index data"], R_GRF_data)
    plt.title("Right GRF")
    cx1.legend(list(R_GRF_data.columns))

    cx2 = plt.subplot(2, 1, 2, sharex=cx1)
    plt.plot(df_index["index data"], L_GRF_data)
    plt.title("Left GRF")
    plt.xlabel("Index")
    cx2.legend(list(L_GRF_data.columns))

    plt.tight_layout()
    GRF_walk_directory = str(save_path)

    try:
        if not os.path.exists(GRF_walk_directory):
            os.makedirs(GRF_walk_directory)
    except OSError:
        pass

    plt.savefig(str(GRF_walk_directory)+"/RH-%s_%s_GRF.png"
                % (str(RH_num), name))

    csv_GRF_directory = str(csv_path)

    try:
        if not os.path.exists(csv_GRF_directory):
            os.makedirs(csv_GRF_directory)
    except OSError:
        pass

    walk_GRF = pd.concat([df_index, L_GRF_data], axis=1)
    walk_GRF = pd.concat([walk_GRF, R_GRF_data], axis=1)
    walk_GRF.to_csv(str(csv_GRF_directory) +
                    "/RH-%s_%s_GRF.csv" % (str(RH_num), name),
                    header=True, index=True, sep=',')


def sensor_sync(path, RH_num, state):

    # data reading ##################################################
    data = pd.read_csv(path, sep=" |,", header=None)

    data = data.astype(float)
    data = data.dropna(axis=1)
    data = data.dropna(axis=0)
    data.columns = range(data.columns.size)
    data.reset_index(drop=True, inplace=True)

    data = data[np.arange(data.columns[-10], data.columns[-1]+1, 1)]
    data.columns = range(data.columns.size)

    if str(RH_num) == "00":
        data.columns = ['time', 'v0', 'v1', 'v2', 'v3',
                        'v4', 'v5', 'v6', 'v7']
        data.reset_index(drop=True, inplace=True)

        return data

    else:
        data.columns = ['sync', 'time', 'v0', 'v1', 'v2',
                        'v3', 'v4', 'v5', 'v6', 'v7']
        data.reset_index(drop=True, inplace=True)

        # sync start ###################################################
        data_synced = data[data["sync"] > 0.0]
        index_synced = list(data_synced.index)
        index_synced = [float(a) for a in index_synced]
        tmp = np.insert(index_synced, 0, np.nan, axis=0)
        tmp = np.delete(tmp, -1)

        index_synced = pd.DataFrame(index_synced, columns=["sync_index"])
        index_synced["sync_pre_index"] = tmp
        index_synced = index_synced.astype(float)

        index_synced["del_sync"] = index_synced[
            "sync_index"] - index_synced["sync_pre_index"]
        index_synced = index_synced.dropna(axis=0)

        d_index = index_synced[index_synced["del_sync"] > 1.0]

        # exceptional sync processing ###################################
        #################################################################
        if (str(state) == "walk") & (str(RH_num) == "04"):
            sync_start = d_index["sync_index"].values.tolist()
            sync_start = [int(a) for a in sync_start]

            sync_end = d_index["sync_pre_index"].values.tolist()
            sync_end = [int(a) for a in sync_end]
            sync_end.append(int(index_synced.loc[
                index_synced.index[-1], "sync_index"]))

            sync_end.insert(1, sync_start[0])
            del sync_end[2]
            sync_end.insert(-1, sync_start[-1])

            del sync_start[0]
            sync_start.insert(-1, sync_end[-3])
            sync_start.insert(0, sync_end[0])
            del sync_end[0]

        elif (str(state) == "walk") & (str(RH_num) == "06"):
            sync_start = d_index["sync_index"].values.tolist()
            sync_start = [int(a) for a in sync_start]
            sync_start.insert(0, 0)

            sync_end = d_index["sync_pre_index"].values.tolist()
            sync_end = [int(a) for a in sync_end]
            sync_end.append(int(index_synced.loc[
                index_synced.index[-1], "sync_index"]))

        elif (str(state) == "walk") & (str(RH_num) == "03"):
            sync_start = d_index["sync_index"].values.tolist()
            sync_start = [int(a) for a in sync_start]
            sync_start.insert(0, 0)
            sync_start = sync_start[:-1]

            sync_end = d_index["sync_pre_index"].values.tolist()
            sync_end = [int(a) for a in sync_end]

        else:
            sync_start = d_index["sync_index"].values.tolist()
            sync_start = [int(a) for a in sync_start]

            sync_end = d_index["sync_pre_index"].values.tolist()
            sync_end = [int(a) for a in sync_end]
            sync_end.append(int(index_synced.loc[
                index_synced.index[-1], "sync_index"]))
            sync_end = sync_end[1:]

        return data, data_synced, sync_start, sync_end


def sensor_synced_plot_csv(R_data, R_name, L_data, L_name,
                           save_path, csv_path, df_sync, RH_num, state):

    vol = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']

    df_sync_R = df_sync[(df_sync["RH_num"] == "RH-" + str(RH_num)) &
                        (df_sync["R or L"] == "R") &
                        (df_sync["stance or walk"] == str(state))]
    df_sync_R.reset_index(drop=True, inplace=True)
    df_sync_L = df_sync[(df_sync["RH_num"] == "RH-" + str(RH_num)) &
                        (df_sync["R or L"] == "L") &
                        (df_sync["stance or walk"] == str(state))]
    df_sync_L.reset_index(drop=True, inplace=True)

    if str(RH_num) == "00":
        fig = plt.figure(figsize=(6, 5))
        fig.suptitle("RH-%s_SENSOR_raw" % (str(RH_num)), fontsize=15)

        for y in vol:
            dx1 = plt.subplot(2, 1, 1)
            plt.plot(R_data["time"], R_data[y])
            plt.title("%s" % (R_name))
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            dx1.legend(vol)

            dx2 = plt.subplot(2, 1, 2, sharex=dx1)
            plt.plot(L_data["time"], L_data[y])
            plt.title("%s" % (L_name))
            plt.xlabel("Time [s]")
            plt.ylabel("Voltage [V]")
            dx2.legend(vol)

        plt.tight_layout()

    else:
        for x in np.arange(len(df_sync_R)):
            fig = plt.figure(figsize=(6, 5))
            fig.suptitle("RH-%s Trimmed_%s_%s_synced" % (
                str(RH_num), str(state), str(df_sync_L.at[x, "number"])),
                fontsize=15)

            for y in vol:
                dx1 = plt.subplot(2, 1, 1)
                plt.plot(R_data[df_sync_R.at[x, "index start"]:
                                df_sync_R.at[x, "index end"]]["time"],
                         R_data[df_sync_R.at[x, "index start"]:
                                df_sync_R.at[x, "index end"]][y])
                plt.title("%s" % (R_name))
                plt.xlabel("Time [s]")
                plt.ylabel("Voltage [V]")
                dx1.legend(vol)

                dx2 = plt.subplot(2, 1, 2, sharex=dx1)
                plt.plot(L_data[df_sync_L.at[x, "index start"]:
                                df_sync_L.at[x, "index end"]]["time"],
                         L_data[df_sync_L.at[x, "index start"]:
                                df_sync_L.at[x, "index end"]][y])
                plt.title("%s" % (L_name))
                plt.xlabel("Time [s]")
                plt.ylabel("Voltage [V]")
                dx2.legend(vol)

            plt.tight_layout()

            if len(str(df_sync_L.at[x, "number"])) == 1:
                vol_stance_directory = str(save_path) + "/%s_%s" % (
                    str(state), "0" + str(df_sync_L.at[x, "number"]))
                csv_stance_directory = str(csv_path) + "/%s_%s" % (
                    str(state), "0" + str(df_sync_L.at[x, "number"]))
            else:
                vol_stance_directory = str(save_path) + "/%s_%s" % (
                    str(state), str(df_sync_L.at[x, "number"]))
                csv_stance_directory = str(csv_path) + "/%s_%s" % (
                    str(state), str(df_sync_L.at[x, "number"]))

            try:
                if not os.path.exists(vol_stance_directory):
                    os.makedirs(vol_stance_directory)
            except OSError:
                pass

            try:
                if not os.path.exists(csv_stance_directory + "/L"):
                    os.makedirs(csv_stance_directory + "/L")
                    os.makedirs(csv_stance_directory + "/R")
            except OSError:
                pass

            plt.savefig(str(vol_stance_directory) +
                        "/RH-%s_Trimmed_%s_%s_synced_SENSOR.png" % (
                            str(RH_num), str(state),
                            str(df_sync_L.at[x, "number"])))

            for y in vol:
                col = ["time", "vout"]

                Lsensor_synced = pd.concat([
                    L_data[df_sync_L.at[x, "index start"]:
                           df_sync_L.at[x, "index end"]]["time"],
                    L_data[df_sync_L.at[x, "index start"]:
                           df_sync_L.at[x, "index end"]][y]],
                    axis=1)
                Lsensor_synced.columns = col
                Lsensor_synced.to_csv(
                    str(csv_stance_directory) +
                    "/L/RH-%s_Trimmed_%s_%s_synced_SENSOR_%s.csv" % (
                        str(RH_num), str(state),
                        str(df_sync_L.at[x, "number"]), str(int(y[1]) + 1)),
                    header=True, index=True, sep=',')

                Rsensor_synced = pd.concat([
                    R_data[df_sync_R.at[x, "index start"]:
                           df_sync_R.at[x, "index end"]]["time"],
                    R_data[df_sync_R.at[x, "index start"]:
                           df_sync_R.at[x, "index end"]][y]], axis=1)
                Rsensor_synced.columns = col
                Rsensor_synced.to_csv(
                    str(csv_stance_directory) +
                    "/R/RH-%s_Trimmed_%s_%s_synced_SENSOR_%s.csv" % (
                        str(RH_num), str(state),
                        str(df_sync_L.at[x, "number"]), str(int(y[1]) + 1)),
                    header=True, index=False, sep=',')


def sensor_stance_data_sync_plot_save(L_stance_data_list, L_stance_name_list,
                                      R_stance_data_list, R_stance_name_list,
                                      df_sync, RH_num, stance_num,
                                      save_path, csv_path):

    for (L_stance_file, L_stance_name) in zip(
            L_stance_data_list, L_stance_name_list):

        # multiple csv file (ex. L1, L2)
        if L_stance_name[1:2] != "_":
            num = L_stance_name[1:2]
            L_stance_path = L_stance_file
            L_stance_path_name = L_stance_name

            for (R_stance_file, R_stance_name) in zip(
                    R_stance_data_list, R_stance_name_list):
                if R_stance_name[1:2] == str(num):
                    R_stance_path = R_stance_file
                    R_stance_path_name = R_stance_name
        # one csv file exist
        else:
            L_stance_path_name = str(L_stance_name_list[0])
            L_stance_path = str(L_stance_data_list[0])
            R_stance_path_name = str(R_stance_name_list[0])
            R_stance_path = str(R_stance_data_list[0])
        #########################################################
        # R stance data reading
        R_stance_data, R_stance_data_synced, \
            sync_start_R, sync_end_R = sensor_sync(
                R_stance_path, str(RH_num), "stance")

        for x in np.arange(len(sync_start_R)):

            df_sync = df_sync.append({
                "RH_num": "RH-" + str(RH_num), "R or L": "R",
                "stance or walk": "stance",
                "number": int(stance_num[RH_num]) + x,
                "index start": sync_start_R[x],
                "index end": sync_end_R[x],
                "time start": R_stance_data_synced.at[sync_start_R[x], "time"],
                "time end": R_stance_data_synced.at[sync_end_R[x], "time"]},
                ignore_index=True)

        ##############################################################
        # L stance data reading
        L_stance_data, L_stance_data_synced, \
            sync_start_L, sync_end_L = sensor_sync(
                L_stance_path, str(RH_num), "stance")

        for x in np.arange(len(sync_start_L)):

            df_sync = df_sync.append({
                "RH_num": "RH-" + str(RH_num),
                "R or L": "L", "stance or walk": "stance",
                "number": int(stance_num[RH_num]) + x,
                "index start": sync_start_L[x], "index end": sync_end_L[x],
                "time start": L_stance_data_synced.at[sync_start_L[x], "time"],
                "time end": L_stance_data_synced.at[sync_end_L[x], "time"]},
                ignore_index=True)

        #################################################################
        # stance data plot, csv save
        sensor_synced_plot_csv(R_stance_data, R_stance_path_name[:-4],
                               L_stance_data, L_stance_path_name[:-4],
                               save_path, csv_path, df_sync, RH_num, "stance")

    return df_sync


def sensor_walk_data_sync_plot_save(L_walk_data_list, L_walk_name_list,
                                    R_walk_data_list, R_walk_name_list,
                                    df_sync, RH_num, walk_num,
                                    save_path, csv_path):

    for (L_walk_file, L_walk_name) in zip(
            L_walk_data_list, L_walk_name_list):
        # multiple csv file (ex. L1, L2)
        if L_walk_name[1:2] != "_":
            num = L_walk_name[1:2]
            L_walk_path = L_walk_file
            L_walk_path_name = L_walk_name

            for (R_walk_file, R_walk_name) in zip(
                    R_walk_data_list, R_walk_name_list):
                if R_walk_name[1:2] == str(num):
                    R_walk_path = R_walk_file
                    R_walk_path_name = R_walk_name
        # one csv file exist
        else:
            L_walk_path = L_walk_data_list[0]
            L_walk_path_name = L_walk_name_list[0]
            R_walk_path = R_walk_data_list[0]
            R_walk_path_name = R_walk_name_list[0]
        ##############################################################
        # R walk data reading
        if str(RH_num) == "00":
            R_walk_data = sensor_sync(R_walk_path, str(RH_num), "walk")

            df_sync = df_sync.append({
                "RH_num": "RH-" + str(RH_num),
                "R or L": "R",
                "stance or walk": "walk"}, ignore_index=True)
        else:
            R_walk_data, R_walk_data_synced, \
                sync_start_R, sync_end_R = sensor_sync(
                    R_walk_path, str(RH_num), "walk")

            for x in np.arange(len(sync_start_R)):

                df_sync = df_sync.append({
                    "RH_num": "RH-" + str(RH_num),
                    "R or L": "R",
                    "stance or walk": "walk",
                    "number": int(walk_num[RH_num]) + x,
                    "index start": sync_start_R[x],
                    "index end": sync_end_R[x],
                    "time start": R_walk_data_synced.at[
                        sync_start_R[x], "time"],
                    "time end": R_walk_data_synced.at[
                        sync_end_R[x], "time"]}, ignore_index=True)

        ##############################################################
        # L walk data reading
        if str(RH_num) == "00":
            L_walk_data = sensor_sync(L_walk_path, str(RH_num), "walk")

            df_sync = df_sync.append({
                "RH_num": "RH-" + str(RH_num),
                "R or L": "L",
                "stance or walk": "walk"}, ignore_index=True)
        else:
            L_walk_data, L_walk_data_synced, \
                sync_start_L, sync_end_L = sensor_sync(
                    L_walk_path, str(RH_num), "walk")

            for x in np.arange(len(sync_start_L)):

                df_sync = df_sync.append({
                    "RH_num": "RH-" + str(RH_num),
                    "R or L": "L",
                    "stance or walk": "walk",
                    "number": int(walk_num[RH_num]) + x,
                    "index start": sync_start_L[x],
                    "index end": sync_end_L[x],
                    "time start": L_walk_data_synced.at[
                        sync_start_L[x], "time"],
                    "time end": L_walk_data_synced.at[sync_end_L[x], "time"]},
                    ignore_index=True)

        #################################################################
        # walk data plot, csv save
        sensor_synced_plot_csv(R_walk_data, R_walk_path_name[:-4],
                               L_walk_data, L_walk_path_name[:-4],
                               save_path, csv_path, df_sync, RH_num, "walk")

    return df_sync


def force_sync_indexing(df_sync_force, RH_num, state, num):

    if len(num) <= 1:
        num = "0" + str(num)
    ref_start_index = df_sync_force.loc[
        (df_sync_force["RH_num"] == "RH-" + str(RH_num)) &
        (df_sync_force["stance or walk"] == str(state)) &
        (df_sync_force["number"] == num)][
            "index start"].values[0]
    ref_end_index = df_sync_force.loc[
        (df_sync_force["RH_num"] == "RH-" + str(RH_num)) &
        (df_sync_force["stance or walk"] == str(state)) &
        (df_sync_force["number"] == num)][
            "index end"].values[0]
    true_start_index = df_sync_force.loc[
        (df_sync_force["RH_num"] == "RH-" + str(RH_num)) &
        (df_sync_force["stance or walk"] == str(state)) &
        (df_sync_force["number"] == num)][
            "total end of index"].values[0]
    ref_start_time = df_sync_force.loc[
        (df_sync_force["RH_num"] == "RH-" + str(RH_num)) &
        (df_sync_force["stance or walk"] == str(state)) &
        (df_sync_force["number"] == num)][
            "time start"].values[0]
    ref_end_time = df_sync_force.loc[
        (df_sync_force["RH_num"] == "RH-" + str(RH_num)) &
        (df_sync_force["stance or walk"] == str(state)) &
        (df_sync_force["number"] == num)][
            "time end"].values[0]

    # ref_start_index = ref_start_index.astype(int)
    # ref_end_index = ref_end_index.astype(int)
    # true_start_index = true_start_index.astype(int)
    ref_start_time = ref_start_time.astype(float)
    ref_end_time = ref_end_time.astype(float)
    ref_del_time = (ref_end_time - ref_start_time).astype(float)

    return ref_start_index, ref_end_index, true_start_index, \
        ref_start_time, ref_end_time, ref_del_time


def sensor_re_sync(path, sensor_num, csv_path, ref_start_index,
                   ref_end_index, true_start_index,
                   ref_start_time, ref_end_time, ref_del_time):

    data = pd.read_csv(path, delimiter=",", header=0)
    data = data.astype(float)

    ini_time = data.loc[0, "time"]
    data["del_time"] = data["time"] - ini_time
    # 뒤가 기준
    if ref_start_index == int(0):

        extra_subtract = float(data.loc[
            data.index[-1], "del_time"] - ref_del_time)
        data["del_time"] = data["del_time"] - extra_subtract

        re_sync = np.where(data["del_time"] >= 0.0)
        re_sync_index = re_sync[0]
        data = data.loc[re_sync_index, :]
    # 앞이 기준임
    elif ref_end_index == true_start_index:

        re_sync = np.where(data["del_time"] <= ref_del_time)
        re_sync_index = re_sync[0]
        data = data.loc[re_sync_index, :]
    else:
        pass

    data.to_csv(str(csv_path)+"SENSOR%s_mod.csv" % (
            sensor_num), header=True, index=False, sep=',')


def force_interp(sensor_data, force_data, sensor_num, csv_path):

    COP_list = ["Name", "FX1", "FY1", "FZ1", "X1", "Y1", "Z1", "MZ1", "FX2",
                "FY2", "FZ2", "X2", "Y2", "Z2", "MZ2"]
    sensor_time = sensor_data.loc[:, "del_time"]
    sensor_ref_time = sensor_data.loc[:, "time"]
    vout = sensor_data.loc[:, "vout"]
    force_time = force_data.loc[:, "del_time"]
    force_list = pd.DataFrame(columns=COP_list)
    for ref_f in COP_list:
        force_list.loc[:, ref_f] = np.interp(sensor_time, force_time,
                                             force_data.loc[:, ref_f])

    total_data = pd.concat([sensor_time, sensor_ref_time, vout, force_list],
                           axis=1)
    total_data.to_csv(csv_path + "interp_SENSOR_%s.csv" % (sensor_num),
                      header=True, index=False, sep=',')


def real_time_initialization_GPR(NUM_PRE):

    # index_tuple = []
    # for i in ["time", "vout"]:
    #     for j in np.arange(0,NUM_PRE+1,1):
    #         index_tuple = np.append(index_tuple, tuple((str(i), int(j))))

    multi_col_pre = pd.MultiIndex.from_tuples([("time", 0), ("time", 1),
                                               ("time", 2), ("time", 3),
                                               ("time", 4), ("time", 5),
                                               ("vout", 0), ("vout", 1),
                                               ("vout", 2), ("vout", 3),
                                               ("vout", 4), ("vout", 5)])
    pre_first = np.concatenate([np.zeros(NUM_PRE+1), np.ones(NUM_PRE+1)])
    df_pre = pd.DataFrame(np.stack((pre_first, pre_first, pre_first,
                                    pre_first, pre_first, pre_first,
                                    pre_first, pre_first), axis=0),
                          index=np.arange(0, 8, 1), columns=multi_col_pre)
    df_pre = df_pre.stack()
    df_pre = df_pre.astype('float')

    # df_pre = pd.DataFrame(np.concatenate([np.zeros((NUM_PRE+1,1)),
    # np.ones((NUM_PRE+1,1))],axis=1), columns = ["time", "vout"])
    # df_pre = df_pre.astype('float')
    # df_N_pre = pd.DataFrame(index = np.arange(0,8,1),
    # columns = ["vout","loading_type2", "del_V", "pre_1", "pre_2", "pre_3",
    # "pre_4", "pre_5", "elapsed_time2"])
    df_N_pre_first = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    df_N_pre = pd.DataFrame(np.stack((df_N_pre_first, df_N_pre_first,
                                      df_N_pre_first, df_N_pre_first,
                                      df_N_pre_first, df_N_pre_first,
                                      df_N_pre_first, df_N_pre_first), axis=0),
                            index=np.arange(0, 8, 1),
                            columns=["vout", "loading_type2", "del_V", "pre_1",
                                     "pre_2", "pre_3", "pre_4", "pre_5",
                                     "elapsed_time2"])
    df_N_pre = df_N_pre.stack()
    df_N_pre = df_N_pre.astype('float')

    return df_pre, df_N_pre


def real_time_prediction_GPR(current_string, current_size, current_dir,
                             current_sequence, NUM_PRE, tol, df_pre_stack,
                             df_N_pre):

    # GPR_280Left_1, GPR_280Right_1
    WINDOWS = NUM_PRE
    multi_col_pre = pd.MultiIndex.from_tuples([("time", 0), ("time", 1),
                                               ("time", 2), ("time", 3),
                                               ("time", 4), ("time", 5),
                                               ("vout", 0), ("vout", 1),
                                               ("vout", 2), ("vout", 3),
                                               ("vout", 4), ("vout", 5)])
    df_pre_final = pd.DataFrame(index=np.arange(0, 8, 1),
                                columns=multi_col_pre)

    df_cur = pd.DataFrame(columns=["time", "vout"])
    df_N_cur = pd.DataFrame(columns=["vout", "loading_type2", "del_V", "pre_1",
                                     "pre_2", "pre_3", "pre_4", "pre_5",
                                     "elapsed_time2"])

    prediction = []
    sum_prediction = 0.0
    # for i, seq in enumerate(current_sequence):
    for i, seq in enumerate(current_sequence):

        df_pre = df_pre_stack.loc[i]

        loaded_model = joblib.load("GPR_220515_CASE8_%s%s_%s"
                                   % (str(current_size), str(current_dir),
                                      str(seq)))

        split_1 = current_string.split(",")
        cur_time = float(split_1[2])
        v_string = re.split("[ |\n]", split_1[3])
        v_string = ' '.join(v_string).split()
        cur_vout = float(v_string[int(i)])

        df_cur["time"] = df_pre.loc[1:, "time"]
        df_cur["vout"] = df_pre.loc[1:, "vout"]
        df_cur.reset_index(inplace=True, drop=True)
        df_cur.loc[NUM_PRE, "time"] = cur_time
        df_cur.loc[NUM_PRE, "vout"] = cur_vout

        df_N_cur.loc[0, "vout"] = cur_vout
        df_N_cur.loc[0, "del_V"] = (df_cur[["vout"]] -
                                    df_pre[["vout"]]).rolling(
                                        window=WINDOWS, min_periods=1,
                                        center=True).mean().loc[NUM_PRE,
                                                                "vout"]

        df_N_cur.loc[0, "loading_type2"] = 0.0
        if abs(df_N_cur.loc[0, "del_V"]) > tol:
            df_N_cur.loc[0, "loading_type2"] = -np.sign(df_N_cur.loc[0,
                                                                     "del_V"])

        df_N_cur.loc[0, ["pre_1", "pre_2", "pre_3",
                         "pre_4", "pre_5"]] = df_pre.loc[1:, "vout"].values

        df_N_cur.loc[0, "elapsed_time2"] = df_N_pre.loc[i,
                                                        "elapsed_time2"] + 0.0
        if df_N_cur.loc[0, "loading_type2"] != 0.0:
            df_N_cur.loc[0, "elapsed_time2"] = df_N_cur.loc
            [0, "elapsed_time2"] + (df_cur.loc[NUM_PRE, "time"] -
                                    df_pre.loc[NUM_PRE, "time"])

        result = loaded_model.predict(
            df_N_cur[["vout", "loading_type2", "del_V",
                      "pre_1", "pre_2", "pre_3", "pre_4", "pre_5",
                      "elapsed_time2"]].values.reshape(1, -1))
        prediction = np.append(prediction, result[0][0])
        sum_prediction = sum_prediction + prediction[i]

        df_pre = df_cur
        df_N_pre.loc[i, :] = df_N_cur.loc[0, :]

        T = df_pre["time"].values.tolist()
        V = df_pre["vout"].values.tolist()
        T.extend(V)
        df_pre_final.loc[i, :] = T

    df_pre_final = df_pre_final.stack()
    df_pre_final = df_pre_final.astype('float')

    return df_pre_final, df_N_pre, prediction, sum_prediction


def GRF_indexing(data, walk_num, col_name, gait_cycle_tol, walk_ref_num):

    # L or R
    LorR = str(col_name)[0]

    # GRF indexing
    GRF_index = np.where(data[[col_name]] > 0)[0]

    # setting for previous index
    index_tmp = np.array(GRF_index, dtype=float)
    index_tmp = np.insert(index_tmp, 0, np.nan)
    index_tmp = np.delete(index_tmp, -1)

    # gait index setting (current, previous, del)
    gait_index = pd.DataFrame()
    gait_index["current"] = GRF_index
    gait_index["previous"] = index_tmp
    gait_index["del"] = gait_index["current"] - gait_index["previous"]
    gait_index.reset_index(drop=True, inplace=True)

    start_end_index = pd.DataFrame(columns=["walk_num", "start_index",
                                            "end_index"])
    final_index = pd.DataFrame(columns=["walk_num", "start_index",
                                        "end_index"])

    # gait number per each walk
    within_index = np.where(abs(gait_index["del"]) > gait_cycle_tol*0.4)[0]
    num_within_walk = len(within_index)

    # start, end index per each walk
    for n in range(num_within_walk + 1):
        start_end_index.loc[int(n), "walk_num"] = walk_num
        # initial walk
        if n == 0:
            start_end_index.loc[int(n), "start_index"] = float(GRF_index[0])
        else:
            start_end_index.loc[int(n), "start_index"] = np.array(
                    gait_index.loc[within_index, "current"], dtype="int")[n-1]
        # final walk
        if n == num_within_walk:
            start_end_index.loc[int(n), "end_index"] = float(GRF_index[-1])
        else:
            start_end_index.loc[int(n), "end_index"] = np.array(
                    gait_index.loc[within_index, "previous"], dtype="int")[n]

    # final indexing for gait analysis
    GRF_num = walk_ref_num[int(walk_num)][str(LorR)]
    if GRF_num == len(start_end_index):
        swing_index = start_end_index.loc[
            int(GRF_num - 1), "start_index"] - (start_end_index.loc[
                int(GRF_num - 2), "end_index"] + 1)
        gait_start_index = start_end_index.loc[int(GRF_num - 1), "start_index"]
        gait_end_index = start_end_index.loc[int(GRF_num - 1),
                                             "end_index"] + swing_index
    else:
        gait_start_index = start_end_index.loc[int(GRF_num - 1), "start_index"]
        gait_end_index = start_end_index.loc[int(GRF_num), "start_index"] - 1

    final_index.loc[0, "walk_num"] = walk_num
    final_index.loc[0, "start_index"] = gait_start_index
    final_index.loc[0, "end_index"] = gait_end_index

    return start_end_index, final_index


def possible_sensor_visualization(path, WINDOWS, RH_num, walk_num, R_or_L):

    data = pd.read_csv(path, header=0)

    time_data = data[["time"]]
    sumforce_data = data[["pred_sum"]]
    raw_volt_data = data[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]]
    raw_force_data = data[["pred_1", "pred_2", "pred_3",
                           "pred_4", "pred_5", "pred_6", "pred_7", "pred_8"]]
    KPa_data = (raw_force_data/(pow(0.004, 2)*math.pi))/pow(10, 3)
    KPa_data.reset_index(drop=True, inplace=True)
    sumforce_data = (sumforce_data/(pow(0.004, 2)*math.pi))/pow(10, 3)
    sumforce_data.reset_index(drop=True, inplace=True)
    sumforce_data = sumforce_data.rolling(
        window=WINDOWS, min_periods=1, center=True, axis=0).mean()
    filter_force_data = KPa_data.rolling(
        window=WINDOWS, min_periods=1, center=True, axis=0).mean()
    filter_volt_data = raw_volt_data.rolling(
        window=WINDOWS, min_periods=1, center=True, axis=0).mean()
    comb_data = pd.DataFrame(columns=["case1", "case2"])
    comb_data["case1"] = filter_force_data["pred_5"] + \
        filter_force_data["pred_8"]
    comb_data["case2"] = filter_force_data["pred_2"] + \
        filter_force_data["pred_3"] + filter_force_data["pred_7"]

    # individual sensor plotting

    for p in np.arange(1, 9):

        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("RH-%s walk%s %s sensor_%s" % (str(RH_num),
                                                    str(walk_num), str(R_or_L), str(p)), fontsize=15, **prop_title)

        bx1 = plt.subplot(3, 1, 1)
        for axis in ['top', 'bottom', 'left', 'right']:
            bx1.spines[axis].set_linewidth(2)
        plt.scatter(time_data, sumforce_data)
        plt.xlabel("time [s]", **prop_tick)
        plt.ylabel("Net pressure [kPa]", **prop_tick)
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

        bx2 = plt.subplot(3, 1, 2, sharex=bx1)
        for axis in ['top', 'bottom', 'left', 'right']:
            bx2.spines[axis].set_linewidth(2)
        plt.scatter(time_data, filter_force_data["pred_%s" % (str(p))])
        plt.xlabel("time [s]", **prop_tick)
        plt.ylabel("Pressure_%s [kPa]" % (str(int(p))), **prop_tick)
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

        bx3 = plt.subplot(3, 1, 3, sharex=bx1)
        for axis in ['top', 'bottom', 'left', 'right']:
            bx3.spines[axis].set_linewidth(2)
        plt.scatter(time_data, filter_volt_data["V%s" % (str(p))])
        plt.xlabel("time [s]", **prop_tick)
        plt.ylabel("voltage_%s [kPa]" % (str(int(p))), **prop_tick)
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

        plt.tight_layout()

        save_path = "C:/Users/minhee/OneDrive - SNU/AFO_exp_data/clinical data analysis/sensor_visualization/RH%s_%s_walk%s" % (
            str(RH_num), str(R_or_L), str(walk_num))

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            pass

        plt.savefig(str(save_path)+"/RH-%s walk%s %s sensor_%s.png" %
                    (str(RH_num), str(walk_num), str(R_or_L), str(p)))

    # 8-sensor total plotting

    fig = plt.figure(figsize=(13, 19))
    fig.suptitle("RH-%s walk%s %s 8 pressures total" % (str(RH_num),
                                                        str(walk_num), str(R_or_L)), fontsize=15, **prop_title)

    bx1 = plt.subplot(9, 1, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx1.spines[axis].set_linewidth(2)
    plt.scatter(time_data, sumforce_data)
    plt.xlabel("time [s]", **prop_tick)
    plt.ylabel("Net pressure [kPa]", **prop_tick)
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    for p in np.arange(2, 10):

        bx2 = plt.subplot(9, 1, int(p), sharex=bx1)
        for axis in ['top', 'bottom', 'left', 'right']:
            bx2.spines[axis].set_linewidth(2)
        plt.scatter(time_data, filter_force_data["pred_%s" % (str(int(p-1)))])
        plt.xlabel("time [s]", **prop_tick)
        plt.ylabel("Pressure_%s [kPa]" % (str(int(p-1))), **prop_tick)
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

    plt.tight_layout()

    plt.savefig(str(save_path)+"/RH-%s walk%s %s 8 pressures total.png" %
                (str(RH_num), str(walk_num), str(R_or_L)))

    # 8-voltage total plotting

    fig = plt.figure(figsize=(13, 19))
    fig.suptitle("RH-%s walk%s %s 8 voltages total" % (str(RH_num),
                                                       str(walk_num), str(R_or_L)), fontsize=15, **prop_title)

    bx1 = plt.subplot(9, 1, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx1.spines[axis].set_linewidth(2)
    plt.scatter(time_data, sumforce_data)
    plt.xlabel("time [s]", **prop_tick)
    plt.ylabel("Net pressure [kPa]", **prop_tick)
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    for p in np.arange(2, 10):

        bx2 = plt.subplot(9, 1, int(p), sharex=bx1)
        for axis in ['top', 'bottom', 'left', 'right']:
            bx2.spines[axis].set_linewidth(2)
        plt.scatter(time_data, filter_volt_data["V%s" % (str(int(p-1)))])
        plt.xlabel("time [s]", **prop_tick)
        plt.ylabel("Voltage_%s [V]" % (str(int(p-1))), **prop_tick)
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')

    plt.tight_layout()

    plt.savefig(str(save_path)+"/RH-%s walk%s %s 8 voltages total.png" %
                (str(RH_num), str(walk_num), str(R_or_L)))

    # sensor combination plotting

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("RH-%s walk%s %s sensor_combination" % (str(RH_num),
                                                         str(walk_num), str(R_or_L)), fontsize=15, **prop_title)

    bx1 = plt.subplot(3, 1, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx1.spines[axis].set_linewidth(2)
    plt.scatter(time_data, sumforce_data)
    plt.xlabel("time [s]", **prop_tick)
    plt.ylabel("Net pressure [kPa]", **prop_tick)
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    bx2 = plt.subplot(3, 1, 2, sharex=bx1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx2.spines[axis].set_linewidth(2)
    plt.scatter(time_data, comb_data["case1"])
    plt.xlabel("time [s]", **prop_tick)
    plt.ylabel("Pressure_5+8 [kPa]", **prop_tick)
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    bx3 = plt.subplot(3, 1, 3, sharex=bx1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx3.spines[axis].set_linewidth(2)
    plt.scatter(time_data, comb_data["case2"])
    plt.xlabel("time [s]", **prop_tick)
    plt.ylabel("Pressure_2+3+7 [kPa]", **prop_tick)
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    plt.tight_layout()

    plt.savefig(str(save_path)+"/RH-%s walk%s %s sensor_combination.png" %
                (str(RH_num), str(walk_num), str(R_or_L)))


def gait_phase_preprocessing(path, counter_path, WINDOWS, gait_cycle_tol, RH_num, walk_num, R_or_L, walk_force_labeling, save_path):

    ###########################################################################
    # for gait detection
    ###########################################################################

    ref_gait_index = pd.DataFrame(columns=[
                                  "RH_num", "walk_num", "R_or_L", "start_index", "end_index", "start_time", "end_time"])
    counter_gait_index = pd.DataFrame(columns=[
                                      "RH_num", "walk_num", "R_or_L", "start_index", "end_index", "start_time", "end_time"])

    for (d_num, data_path) in enumerate([path, counter_path]):

        gait_index_time = pd.DataFrame(columns=[
                                       "RH_num", "walk_num", "R_or_L", "start_index", "end_index", "start_time", "end_time"])

        data = pd.read_csv(data_path, header=0)

        force_plate_data = pd.DataFrame()
        force_plate_data["force_plate"] = data.iloc[:, 2]
        force_plate_data["time"] = data.time
        force_plate_data["filtered_force_plate"] = force_plate_data[["force_plate"]].rolling(
            window=WINDOWS, min_periods=1, center=True, axis=0).mean()

        force_plate_index = np.where(
            force_plate_data.force_plate <= (gait_cycle_tol))[0]

        true_index = np.delete(force_plate_data.index, force_plate_index)
        true_index = np.array(true_index)

        index_tmp = np.array(true_index, dtype=float)
        index_tmp = np.insert(index_tmp, 0, np.nan)
        index_tmp = np.delete(index_tmp, -1)

        gait_index = pd.DataFrame()
        gait_index["current"] = true_index
        gait_index["previous"] = index_tmp
        gait_index["del"] = gait_index["current"] - gait_index["previous"]
        gait_index.reset_index(drop=True, inplace=True)

        start_end_index = pd.DataFrame(columns=["start_index", "end_index"])
        # cycle_start_index = np.zeros((1,10))
        # cycle_end_index = np.zeros((1,10))

        start_end_index["end_index"] = np.array(gait_index.loc[np.where(
            abs(gait_index["del"]) > gait_cycle_tol*2)[0], "previous"], dtype=float)
        start_end_index.loc[len(start_end_index["end_index"]),
                            "end_index"] = float(true_index[-1])
        start_end_index.loc[1:, "start_index"] = np.array(gait_index.loc[np.where(
            abs(gait_index["del"]) > gait_cycle_tol*2)[0], "current"], dtype=float)
        start_end_index.loc[0, "start_index"] = float(true_index[0])
        # cycle_end_index = np.array(gait_index.loc[np.where(abs(gait_index["del"]) >gait_cycle_tol)[0],"previous"], dtype=float)
        # cycle_end_index = np.append(cycle_end_index, true_index[-1])
        # cycle_start_index = np.array(gait_index.loc[np.where(abs(gait_index["del"]) > gait_cycle_tol)[0],"current"], dtype=float)
        # cycle_start_index = np.insert(cycle_start_index, 0, true_index[0])

        # print("cycle_start_index")
        # print(cycle_start_index)
        # print(len(cycle_start_index))
        # print("cycle_end_index")
        # print(cycle_end_index)
        # print(len(cycle_end_index))

        # print(cycle_end_index)
        # print(cycle_start_index)
        # gait_index_time.loc[:,"start_index"] = cycle_start_index.tolist()
        # gait_index_time.loc[:,"end_index"] = cycle_end_index.tolist()

        # print(len(start_end_index))

        # print(start_end_index["start_index"])
        # print(start_end_index["end_index"])

        drop_ind = []
        for ind in np.arange(0, len(start_end_index)):

            # print("end -> start index")
            # print(start_end_index["end_index"])
            # print(start_end_index["start_index"])
            if (start_end_index.loc[int(ind), "end_index"] - start_end_index.loc[int(ind), "start_index"]) <= 300:

                drop_ind.append(int(ind))
                # start_end_index = start_end_index.drop(index=int(ind))
                # start_end_index.reset_index(drop=True, inplace=True)
            # cycle_start_index = np.delete(cycle_start_index, int(ind))
            # cycle_end_index = np.delete(cycle_end_index, int(ind))

        # print(start_end_index[["start_index"]])
        # print(start_end_index[["end_index"]])

        start_end_index = start_end_index.drop(index=drop_ind)
        start_end_index.reset_index(drop=True, inplace=True)

        gait_index_time["start_index"] = start_end_index["start_index"]
        gait_index_time["end_index"] = start_end_index["end_index"]
        for i in np.arange(0, len(gait_index_time)):
            gait_index_time.loc[int(i), "start_index"] = int(
                gait_index_time.loc[int(i), "start_index"])
            gait_index_time.loc[int(i), "end_index"] = int(
                gait_index_time.loc[int(i), "end_index"])
            gait_index_time.loc[int(i), "start_time"] = force_plate_data.loc[int(
                gait_index_time.loc[int(i), "start_index"]), "time"]
            gait_index_time.loc[int(i), "end_time"] = force_plate_data.loc[int(
                gait_index_time.loc[int(i), "end_index"]), "time"]
        # print(int(gait_index_time["start_index"]))
        # print(int(gait_index_time["end_index"]))
        gait_index_time["RH_num"] = str(RH_num)
        gait_index_time["walk_num"] = str(walk_num)

        # print(gait_index_time)

        if d_num == 0:

            ref_gait_index = gait_index_time
            ref_gait_index.loc[:, "R_or_L"] = str(R_or_L)

        else:

            counter_gait_index = gait_index_time
            counter_gait_index.loc[:, "R_or_L"] = str(
                walk_force_labeling[str(RH_num)][int(walk_num)][1])

        # print(gait_index_time)

        # for h in np.arange(1,NUM_AVG+1,1):
        #     tmp = force_plate_data.filtered_force_plate.values.tolist()
        #     for p in np.arange(0,h,1):
        #         tmp = np.insert(tmp,int(p),np.nan)
        #         tmp = np.delete(tmp,-1)

        #     force_plate_data["pre_force_plate_%s" %(str(h))] = tmp

        # force_plate_data["del_force_plate"] = force_plate_data["filtered_force_plate"] - force_plate_data["pre_force_plate_%s" %(NUM_AVG)]
        # force_plate_data = force_plate_data.dropna(axis=0)

        # data = data.loc[force_plate_data.index,:]
        # data.reset_index(drop=True, inplace=True)
        # force_plate_data.reset_index(drop=True, inplace=True)

        # # gait index
        # cycle_index = np.where((abs(force_plate_data.del_force_plate)>= gait_cycle_tol) & (force_plate_data.filtered_force_plate >= 10.0))[0]
        # # print(walk_num)
        # # plt.plot(force_plate_data.loc[cycle_index,"time"], force_plate_data.loc[cycle_index,"filtered_force_plate"])
        # # print(cycle_index)

        # index_tmp = np.array(cycle_index, dtype=float)
        # index_tmp = np.insert(index_tmp,0,np.nan)
        # index_tmp = np.delete(index_tmp,-1)

        # gait_index = pd.DataFrame()
        # gait_index["current"] = cycle_index
        # gait_index["previous"] = index_tmp
        # gait_index["del"] = gait_index["current"] - gait_index["previous"]
        # gait_index.reset_index(drop=True, inplace=True)
        # # plt.figure()
        # # plt.plot(gait_index["del"])

        # cycle_start_index = np.array(gait_index.loc[np.where(abs(gait_index["del"]) > 10.0)[0],"current"], dtype=float)
        # # print(cycle_start_index)
        # cycle_end_index = cycle_start_index - 1
        # cycle_start_index = np.delete(cycle_start_index, -1)
        # cycle_end_index = np.delete(cycle_end_index, 0)

        # if d_num == 0:

        #     for i, c_start in enumerate(cycle_start_index):

        #         ref_gait_index = ref_gait_index.append({"RH_num": str(RH_num), "walk_num": str(walk_num), "R_or_L": str(R_or_L), "start_index": c_start, "end_index": cycle_end_index[i], "start_time": data["time"].loc[c_start], "end_time": data["time"].loc[cycle_end_index[i]]}, ignore_index=True)

        # elif d_num == 1:

        #     for i, c_start in enumerate(cycle_start_index):

        #         counter_gait_index = counter_gait_index.append({"RH_num": str(RH_num), "walk_num": str(walk_num), "R_or_L": walk_force_labeling[str(RH_num)][int(walk_num)][1], "start_index": c_start, "end_index": cycle_end_index[i], "start_time": data["time"].loc[c_start], "end_time": data["time"].loc[cycle_end_index[i]]}, ignore_index=True)

    start_time = ref_gait_index.loc[0, "start_time"]
    start_index = int(ref_gait_index.loc[0, "start_index"])
    end_index = 0

    if counter_gait_index.loc[0, "start_time"] >= start_time:

        end_time = counter_gait_index.loc[0, "end_time"]
        counter_start_time = counter_gait_index.loc[0, "start_time"]
        end_index = int(counter_gait_index.loc[0, "end_index"])

    else:

        for num in np.arange(1, len(counter_gait_index), 1):

            if counter_gait_index.loc[num, "start_time"] >= start_time:

                end_time = counter_gait_index.loc[num, "end_time"]
                counter_start_time = counter_gait_index.loc[num, "start_time"]
                end_index = int(counter_gait_index.loc[num, "end_index"])
                pass

        # if end_index == 0:

        #     end_index =

    for num in np.arange(1, len(ref_gait_index), 1):

        if ref_gait_index.loc[num, "start_time"] <= counter_start_time:

            start_time = ref_gait_index.loc[num, "start_time"]
            start_index = int(ref_gait_index.loc[num, "start_index"])

    # print("RH"+RH_num+"walk"+walk_num)
    # print(start_time)
    # print(end_time)
    #######################################################################
    # 여기부터 고치기
    ########################################################################

    # gait cycle start time & end time
    # current_start_time = data.loc[cycle_index[0],"time"]
    # current_end_time = data.loc[cycle_index[-1], "time"]

    # # total gait cycle time
    # if pre_start_time is None:

    #     gait_start_time = float(current_start_time)
    #     gait_end_time = float(current_end_time)
    #     cur_gait_cycle_data = pd.read_csv(path, header=0)

    # else:

    #     if pre_start_time < current_start_time:
    #         gait_start_time = float(pre_start_time)
    #         gait_end_time = float(current_end_time)

    #     else:
    #         gait_start_time = float(current_start_time)
    #         gait_end_time = float(pre_end_time)

    #     pre_gait_cycle_data = pd.read_csv(pre_path, header=0)
    #     cur_gait_cycle_data = pd.read_csv(path, header=0)
    # ###########################################################################
    # start_index = np.where(cur_gait_cycle_data.time>= gait_start_time)[0][0]

    # end_index = np.where(cur_gait_cycle_data.time<= gait_end_time)[0][-1]

    # start_index = gait_start_index
    # end_index = gait_end_index

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        pass

    # if pre_path is None:

    #     final_gait_cycle_data = pd.DataFrame()

    #     gait_data = cur_gait_cycle_data

    #     final_time_data = gait_data.loc[start_index:end_index, "time"]
    #     final_start_time = gait_data.loc[start_index, "time"]
    #     final_end_time = gait_data.loc[end_index, "time"]
    #     final_percent_data = pd.DataFrame(columns = ["gait_cycle_%"])
    #     final_percent_data["gait_cycle_%"] = ((final_time_data.sub(final_start_time)).multiply(100)).divide(final_end_time-final_start_time)
    #     final_force_plate_data = gait_data.loc[start_index:end_index, gait_data.columns[2]]
    #     final_force_sum_data = gait_data.loc[start_index:end_index, "pred_sum"]
    #     final_force_indi_data = gait_data.loc[start_index:end_index, ["pred_1", "pred_2", "pred_3", "pred_4", "pred_5", "pred_6", "pred_7", "pred_8"]]
    #     final_volt_indi_data = gait_data.loc[start_index:end_index, ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]]

    #     final_force_plate_data = final_force_plate_data.rolling(window=WINDOWS, min_periods=1,center=True, axis=0).mean()
    #     final_force_sum_data = final_force_sum_data.rolling(window=WINDOWS, min_periods=1,center=True, axis=0).mean()
    #     final_force_indi_data = final_force_indi_data.rolling(window=WINDOWS, min_periods=1,center=True, axis=0).mean()
    #     final_volt_indi_data = final_volt_indi_data.rolling(window=WINDOWS, min_periods=1,center=True, axis=0).mean()

    #     final_gait_cycle_data = pd.concat([final_time_data, final_percent_data, final_force_plate_data, final_force_sum_data, final_force_indi_data, final_volt_indi_data], axis=1)

    #     final_gait_cycle_data.to_csv(str(save_path)+"/RH_%s walk%s %s sensors.csv" %(str(RH_num), str(walk_num), str(R_or_L)), sep = ",", index = False)

    # else:

    gait_cycle_data = pd.read_csv(path, header=0)
    counter_gait_cycle_data = pd.read_csv(counter_path, header=0)

    for (num, gait_data) in enumerate([gait_cycle_data, counter_gait_cycle_data]):
        final_gait_cycle_data = pd.DataFrame()

        final_time_data = gait_data.loc[start_index:end_index, "time"]
        final_start_time = gait_data.loc[start_index, "time"]
        final_end_time = gait_data.loc[end_index, "time"]
        final_percent_data = pd.DataFrame(columns=["gait_cycle_%"])
        final_percent_data["gait_cycle_%"] = ((final_time_data.sub(
            final_start_time)).multiply(100)).divide(final_end_time-final_start_time)
        final_force_plate_data = gait_data.loc[start_index:end_index,
                                               gait_data.columns[2]]
        final_force_sum_data = gait_data.loc[start_index:end_index, "pred_sum"]
        final_force_indi_data = gait_data.loc[start_index:end_index, [
            "pred_1", "pred_2", "pred_3", "pred_4", "pred_5", "pred_6", "pred_7", "pred_8"]]
        final_volt_indi_data = gait_data.loc[start_index:end_index, [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]]

        final_force_plate_data = final_force_plate_data.rolling(
            window=WINDOWS, min_periods=1, center=True, axis=0).mean()
        final_force_sum_data = final_force_sum_data.rolling(
            window=WINDOWS, min_periods=1, center=True, axis=0).mean()
        final_force_indi_data = final_force_indi_data.rolling(
            window=WINDOWS, min_periods=1, center=True, axis=0).mean()
        final_volt_indi_data = final_volt_indi_data.rolling(
            window=WINDOWS, min_periods=1, center=True, axis=0).mean()

        final_gait_cycle_data = pd.concat([final_time_data, final_percent_data, final_force_plate_data,
                                           final_force_sum_data, final_force_indi_data, final_volt_indi_data], axis=1)

        if num == 0:
            sensor_dir = str(R_or_L)
        else:
            if str(R_or_L) == "R":
                sensor_dir = "L"
            elif str(R_or_L) == "L":
                sensor_dir = "R"

        final_gait_cycle_data.to_csv(str(save_path)+"/RH_%s walk%s %s sensors.csv" % (
            str(RH_num), str(walk_num), str(sensor_dir)), sep=",", index=False)

    return ref_gait_index, counter_gait_index

# def mean_std_for_multiple_series()

# def gait_phase_visualization(path, WINDOWS, NUM_AVG, gait_cycle_tol, RH_num, walk_num, R_or_L, save_path, pre_path=None, pre_start_time=None, pre_end_time=None):


def gait_phase_visualization(R_path, R_path_name, walk_force_labeling, desired_col_num, desired_info_name, save_path):

    R_data_first = pd.DataFrame(columns=["path", "name"])
    L_data_first = pd.DataFrame(columns=["path", "name"])
    for i, (cycle_R_data, cycle_R_name) in enumerate(zip(R_path, R_path_name)):

        RH_num = str(cycle_R_name[3:5])
        walk_num = int(cycle_R_name[10:12])

        if walk_force_labeling[RH_num][int(walk_num)][0] == 'R':
            R_data_first = R_data_first.append(
                {"path": cycle_R_data, "name": cycle_R_name}, ignore_index=True)
        elif walk_force_labeling[RH_num][int(walk_num)][0] == 'L':
            L_data_first = L_data_first.append({"path": str(
                cycle_R_data[:-13])+"L sensors.csv", "name": str(cycle_R_name[:-13])+"L sensors.csv"}, ignore_index=True)
        else:
            pass

    # preprocessing for interpolation
    max_num_R = 0
    gait_num_R = 0
    for p, (R_first_data, R_first_name) in enumerate(zip(R_data_first["path"], R_data_first["name"])):

        data_R = pd.read_csv(R_first_data, header=0)
        data_R_len = data_R[["gait_cycle_%"]]

        if max_num_R <= len(data_R_len):
            max_num_R = len(data_R_len)
            gait_num_R = p

    max_num_L = 0
    gait_num_L = 0
    for q, (L_first_data, L_first_name) in enumerate(zip(L_data_first["path"], L_data_first["name"])):

        data_L = pd.read_csv(L_first_data, header=0)
        data_L_len = data_L[["gait_cycle_%"]]

        if max_num_L <= len(data_L_len):
            max_num_L = len(data_L_len)
            gait_num_L = q

    if desired_col_num == int(2):
        desired_info_name = "Force_plate"
    else:
        desired_info_name = desired_info_name

    # print("R")
    # print(R_data_first)
    # print("L")
    # print(L_data_first)

    #########################################
    # Right first data set
    ref_gait_R = pd.read_csv(R_data_first.loc[gait_num_R, "path"], header=0)
    desired_info_R = pd.DataFrame(ref_gait_R[["gait_cycle_%"]])
    for p, (R_first_data, R_first_name) in enumerate(zip(R_data_first["path"], R_data_first["name"])):

        data_R = pd.read_csv(R_first_data, header=0)
        data_L = pd.read_csv(R_first_data[:-13]+"L sensors.csv", header=0)
        data_R_info = data_R.loc[:, ["gait_cycle_%",
                                     data_R.columns[int(desired_col_num)]]]
        data_L_info = data_L.loc[:, ["gait_cycle_%",
                                     data_L.columns[int(desired_col_num)]]]

        # interp
        desired_info_R[str(desired_info_name)+"_R_%s" % (str(R_first_name[6:12]))] = np.interp(
            desired_info_R["gait_cycle_%"], data_R_info["gait_cycle_%"], data_R_info[data_R.columns[int(desired_col_num)]])
        desired_info_R[str(desired_info_name)+"_L_%s" % (str(R_first_name[6:12]))] = np.interp(
            desired_info_R["gait_cycle_%"], data_L_info["gait_cycle_%"], data_L_info[data_L.columns[int(desired_col_num)]])

    # to_csv()
    R_col = np.where(desired_info_R.columns.str.contains("R_walk"))[0]
    L_col = np.where(desired_info_R.columns.str.contains("L_walk"))[0]

    desired_info_R[str(desired_info_name)+"_R_mean"] = np.mean(
        desired_info_R.loc[:, desired_info_R.columns[R_col]], axis=1)
    desired_info_R[str(desired_info_name)+"_R_std"] = np.std(
        desired_info_R.loc[:, desired_info_R.columns[R_col]], axis=1)

    desired_info_R[str(desired_info_name)+"_L_mean"] = np.mean(
        desired_info_R.loc[:, desired_info_R.columns[L_col]], axis=1)
    desired_info_R[str(desired_info_name)+"_L_std"] = np.std(
        desired_info_R.loc[:, desired_info_R.columns[L_col]], axis=1)

    # plotting: desired_info_R("gait_cycle_%", str(desired_info_name)+"_R_mean", str(desired_info_name)+"_R_std", str(desired_info_name)+"_L_mean", str(desired_info_name)+"_L_std")

    # #########################################
    # # Left first data set
    # ref_gait_L = pd.read_csv(L_data_first.loc[gait_num_L, "path"], header=0)
    # desired_info_L = pd.DataFrame(ref_gait_L[["gait_cycle_%"]])
    # for q, (L_first_data, L_first_name) in enumerate(zip(L_data_first["path"], L_data_first["name"])):

    #     data_L = pd.read_csv(L_first_data, header=0)
    #     data_R = pd.read_csv(L_first_data[:-13]+"R sensors.csv", header=0)
    #     data_L_info = data_L[["gait_cycle_%", data_L.columns[int(desired_col_num)]]]
    #     data_R_info = data_R[["gait_cycle_%", data_R.columns[int(desired_col_num)]]]

    #     # interp
    #     desired_info_L[str(desired_info_name)+"_L_%s" %(str(L_first_name[6:12]))] = np.interp(desired_info_L["gait_cycle_%"], data_L_info["gait_cycle_%"], data_L_info[data_L.columns[int(desired_col_num)]])
    #     desired_info_L[str(desired_info_name)+"_R_%s" %(str(L_first_name[6:12]))] = np.interp(desired_info_L["gait_cycle_%"], data_R_info["gait_cycle_%"], data_R_info[data_R.columns[int(desired_col_num)]])

    # # to_csv()
    # L_col = np.where(desired_info_L.columns.str.contains("L_walk"))[0]
    # R_col = np.where(desired_info_L.columns.str.contains("R_walk"))[0]

    # desired_info_L[str(desired_info_name)+"_L_mean"] = np.mean(desired_info_L.loc[:,desired_info_L.columns[L_col]], axis=1)
    # desired_info_L[str(desired_info_name)+"_L_std"] = np.std(desired_info_L.loc[:,desired_info_L.columns[L_col]], axis=1)

    # desired_info_L[str(desired_info_name)+"_R_mean"] = np.mean(desired_info_L.loc[:,desired_info_L.columns[R_col]], axis=1)
    # desired_info_L[str(desired_info_name)+"_R_std"] = np.std(desired_info_L.loc[:,desired_info_L.columns[R_col]], axis=1)

    # # plotting: desired_info_L("gait_cycle_%", str(desired_info_name)+"_L_mean", str(desired_info_name)+"_L_std", str(desired_info_name)+"_R_mean", str(desired_info_name)+"_R_std")

    ###########################################
    if desired_col_num == int(2):
        y_range_max = 1000
        y_range = np.arange(0, y_range_max*(11/10), y_range_max*(1/10))
    elif desired_info_name.startswith("pred_sum") == 1:
        y_range_max = 100
        y_range = np.arange(0, y_range_max*(11/10), y_range_max*(1/10))
    else:
        y_range = []

    # plotting for right data first
    fig = plt.figure(figsize=(15, 12))
    # fig.suptitle("RH-%s_%s_right first" %(str(RH_num), str(desired_info_name)), fontsize=15, **prop_title)
    fig.suptitle("RH-%s_%s" % (str(RH_num), str(desired_info_name)),
                 fontsize=15, **prop_title)

    bx1 = plt.subplot(2, 1, 1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx1.spines[axis].set_linewidth(2)
    plt.errorbar(desired_info_R["gait_cycle_%"], desired_info_R[str(desired_info_name)+"_R_mean"],
                 desired_info_R[str(desired_info_name)+"_R_std"], fmt='-o', ecolor="lightskyblue")
    plt.errorbar(desired_info_R["gait_cycle_%"], desired_info_R[str(
        desired_info_name)+"_L_mean"], desired_info_R[str(desired_info_name)+"_L_std"], fmt='-o', ecolor="orange")
    plt.xlabel("Gait cycle [%]", **prop_tick)
    plt.ylabel("%s" % (str(desired_info_name)), **prop_tick)
    plt.xticks(np.arange(0, 110, 10), fontsize=15, fontweight='bold')
    plt.yticks(y_range, fontsize=15, fontweight='bold')
    plt.grid(True)

    bx2 = plt.subplot(2, 1, 2, sharex=bx1)
    for axis in ['top', 'bottom', 'left', 'right']:
        bx2.spines[axis].set_linewidth(2)
    plt.errorbar(desired_info_R["gait_cycle_%"], desired_info_R[str(desired_info_name)+"_R_mean"],
                 desired_info_R[str(desired_info_name)+"_R_std"], fmt='-o', ecolor="lightskyblue")
    plt.errorbar(desired_info_R["gait_cycle_%"], desired_info_R[str(
        desired_info_name)+"_L_mean"], desired_info_R[str(desired_info_name)+"_L_std"], fmt='-o', ecolor="red")
    plt.xlabel("Gait cycle [%]", **prop_tick)
    plt.ylabel("%s" % (str(desired_info_name)), **prop_tick)
    plt.xticks(np.arange(0, 110, 10), fontsize=15, fontweight='bold')
    plt.yticks(y_range, fontsize=15, fontweight='bold')
    plt.grid(True)

    plt.savefig(str(save_path)+"RH-%s %s Right first_%s_test.png" %
                (str(RH_num), str(walk_num), str(desired_info_name)))

    # # plotting for left data first
    # fig = plt.figure(figsize=(15,8))
    # fig.suptitle("RH-%s_%s_left first" %(str(RH_num), str(desired_info_name)), fontsize=15, **prop_title)

    # cx1 = plt.subplot(2,1,1)
    # for axis in ['top','bottom','left','right']:
    #     cx1.spines[axis].set_linewidth(2)
    # plt.errorbar(desired_info_L["gait_cycle_%"], desired_info_L[str(desired_info_name)+"_R_mean"], desired_info_L[str(desired_info_name)+"_R_std"], fmt='-o', ecolor="lightskyblue")
    # plt.xlabel("Gait cycle [%]", **prop_tick)
    # plt.ylabel("[Right] %s" %(str(desired_info_name)), **prop_tick)
    # plt.xticks(np.arange(0,110,10),fontsize=15, fontweight='bold')
    # plt.yticks(y_range,fontsize=15,fontweight='bold')
    # plt.grid(True)

    # cx2 = plt.subplot(2,1,2, sharex = cx1)
    # for axis in ['top','bottom','left','right']:
    #     cx2.spines[axis].set_linewidth(2)
    # plt.errorbar(desired_info_L["gait_cycle_%"], desired_info_L[str(desired_info_name)+"_L_mean"], desired_info_L[str(desired_info_name)+"_L_std"], fmt='-o', ecolor="lightskyblue")
    # plt.xlabel("Gait cycle [%]", **prop_tick)
    # plt.ylabel("[Left] %s" %(str(desired_info_name)), **prop_tick)
    # plt.xticks(np.arange(0,110,10),fontsize=15, fontweight='bold')
    # plt.yticks(y_range,fontsize=15,fontweight='bold')
    # plt.grid(True)

    # plt.savefig(str(save_path)+"RH-%s %s Left first_%s.png" %(str(RH_num), str(walk_num), str(desired_info_name)))

    return desired_info_R
# , desired_info_L

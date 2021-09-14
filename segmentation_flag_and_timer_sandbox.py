# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:33:30 2021

@author: 6degt

segmentaion with flag and timer:

start area =  flag + WAITING_TIME
stop area = start areas on reversed data

IDEA: working with svd(gyro)
[1] Find point for "leaving the None"  - pointer
[2] Check from the pointer for WAITING_TIME 
    (1) is there 'big' value (> PEAKS_FACTOR*MAX)? 
    (2) No value Less then THR
[3] Reverse data, find ends (area)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode, median 

import os
from os import listdir
from os.path import isfile, join  # isfile - (file or directory (guru99))


# Hyperparams for segmentation_with_flag_and_timer
WAITING_TIME = 20
SLEEP_TIME = 80  # avg time per motion
GYRO_SVD_FACTOR = 1.0
PEAKS_FACTOR = 0.2

THR = 0.13


# additional constants:
gravity_constant = 10.20

# Dimensions and titles:
coord_triple = ['x', 'y', 'z']
accel_triple = ['a' + i for i in coord_triple]
gyro_triple = ['g' + i for i in coord_triple]
magn_triple = ['m' + i for i in coord_triple]


imu_titles_6dof = accel_triple + gyro_triple
imu_titles_9dof = imu_titles_6dof + magn_triple

# Motions:
list_of_motions = ['N', 'U', 'D', 'L', 'R', 'T', 'C']
zipped_li = list(zip(list_of_motions, range(7)))
char_to_num_dicti = {key: val for (key, val) in zipped_li}
num_char_dicti = {val: key for key, val in char_to_num_dicti.items()}




def list_of_files(directory):
    li = [f for f in listdir(directory) if isfile(join(directory, f))]
    return li


def file_name_parser(file_name):
    # assumption:
    # file_name ='imu_output_recording_10_01_2021_14_42_motion_T.csv'
    file_title = file_name.split('.')[-2]
    li = file_title.split('_')
    motion = li[-1]
    return motion


def dictionary_of_motions_files(li_of_files):
    dictionary_of_motions_files = {}
    for file in li_of_files:
        motion = file_name_parser(file)
        dictionary_of_motions_files[motion] = file
    return dictionary_of_motions_files


def svd_reduction(data_matrix):
    data_matrix_avg = np.mean(data_matrix, axis=0)
    data_matrix_cap = data_matrix - data_matrix_avg
    corr_matrix = np.dot(data_matrix_cap.T, data_matrix_cap)
    (U, S, Vh) = np.linalg.svd(corr_matrix)
    # S=np.diag(S)
    main_direction = U[:, 0]
    # main direction according to abs(!!!)(U[:,0])
    main_direction_amplitude = np.dot(U[:, 0].T, data_matrix_cap.T)
    # projection without OFFSET!!!

    return main_direction_amplitude, main_direction

def add_gyro_svd(df):
    data_matrix = df[gyro_triple].to_numpy()
    df['gyro svd'] = svd_reduction(data_matrix)[0]


def add_accel_norm(df):
    
    df['gyro norm'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    df['accel norm'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df[f'accel norm g = {gravity_constant} reduced'] = df['accel norm'] - gravity_constant


# PLOTS:
    
def plot(df, choosed_channels, labels_li_as_nums, title, file_name,
         start, stop):
    df[choosed_channels][start:stop].plot()
    plt.plot(labels_li_as_nums, label='label')
    plt.title(title)
    plt.xlabel(file_name)
    plt.legend()
    plt.show()

def labels_list_converter(list_of_labels, di):
    return [di[el] for el in list_of_labels]


def plotter(csv_file_path):

    df = pd.read_csv(csv_file_path)
    file_name = csv_file_path.split('/')[-1]
    parsed_name = file_name.split('_')
    motion = parsed_name[-1].split('.')[0]
    title = (f'{user_name}, motion {motion}')
    start = 0
    stop = -1
    add_gyro_svd(df)
    list_of_labels = list(df['label'])
    labels_li_as_nums = labels_list_converter(list_of_labels,
                                              char_to_num_dicti)
    # plot accel data
    # channels = accel_triple + ['accel norm']
    # plot(df, choosed_channels, title, file_name, start, stop)

    # plot gyro data
    choosed_channels = gyro_triple + ['gyro svd']  # + ['gyro norm']
    plot(df, choosed_channels, labels_li_as_nums, title,
         file_name, start, stop)

    # plot magn data
    # choosed_channels = gyro_triple  # + ['gyro norm']
    # plot(df, choosed_channels, title, file_name, start, stop)
    
    
def clean_figs_run_plt_params():
    plt.close(fig='all')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt. rcParams["legend.loc"] = 'lower right'
    # line style:
    plt.rcParams['lines.linewidth'] = 0.2
    plt.rcParams['lines.markersize'] = 1.0
    plt.rcParams['lines.marker'] = 'o'


def motion_cloud_plot(motion_df, starts, stops):
    add_gyro_svd(df)
    add_accel_norm(df)
    plt.figure()
    for start, stop in zip(starts, stops):
        df['accel norm'].plot()
        df['gyro svd g = 9.8 reduced'].plot()


def threshold_crossed(ind, svd_gyro_abs):
    if svd_gyro_abs[ind+1] > THR and svd_gyro_abs[ind] < THR:
        return True
    else:
        return False


def peak_ahead(ind, gyro_svd_abs):
    ahead_samples = gyro_svd_abs[ind: ind + WAITING_TIME]
    if any(ahead_samples > LEVEL):
        return True
    else:
        return False


def monotonic_increasing(ind, gyro_svd_abs):
    forward_samples = gyro_svd_abs[ind+1: ind + 4]
    # print(forward_samples, THR)
    THR = none_gyro_svd_abs_max_scaled
    if all(forward_samples > THR):
        return True
    else:
        return False


# Flag and timer approach:
def flag_and_timer(gyro_svd):
    starts = []
    ind = 0
    while ind < len(gyro_svd)-1:
        if (threshold_crossed(ind, gyro_svd_abs)
                and peak_ahead(ind, gyro_svd_abs)
                and monotonic_increasing(ind, gyro_svd_abs)):
            starts.append(ind)
            ind = ind + SLEEP_TIME
        else:
            ind += 1
    return starts


# def sleep_for(SLEEP_TIME, ind):
#     wake_up_ind = ind + SLEEP_TIME
#     return wake_up_ind


clean_figs_run_plt_params()
# PART 1. PRINT THE FILES:
# user_name = 'stas1007'
user_name = 'stas'
# user_name = 'aya_1'
# user_name = 'aya'
dir_path = f'data/user_{user_name}'
li = list_of_files(dir_path)

di = dictionary_of_motions_files(li)
print(f'Files for segmentation, USER: {user_name}:')
for key, val in di.items():
    print(f'{key}: {val}')

# PART 2. WORK WITH NONE:
csv_file_name = di['N']
data_name = csv_file_name[:-13]
print(f'\nDATA NAMED BY None file: {data_name}')
csv_file_path = dir_path + '//' + csv_file_name
plotter(csv_file_path)
df_none = pd.read_csv(csv_file_path)
add_gyro_svd(df_none)

# Threshold = Scale*None: (Scale = 1.0 - 2.5)
none_gyro_svd = df_none['gyro svd'].to_numpy()

none_gyro_svd_abs = np.abs(none_gyro_svd)
none_gyro_svd_abs_rounded = np.round(none_gyro_svd_abs,2)
none_gyro_svd_df = pd.DataFrame(none_gyro_svd)
print('None variance(stats): ', none_gyro_svd_df.var())
none_gyro_svd_abs_max = np.max(none_gyro_svd_abs)
none_gyro_svd_abs_max_scaled = GYRO_SVD_FACTOR*none_gyro_svd_abs_max
print(f'\nThreshold defined by {GYRO_SVD_FACTOR}*None:' +
      f'\n{none_gyro_svd_abs_max_scaled}')

samples_forward = 3
val_li = []
# print(f'{samples_forward} samples forward: ')
# For every single motion:
for motion in list_of_motions[1:]:
    csv_file_name = di[motion]
    file_path = dir_path + '//' + csv_file_name

    curr_df = pd.read_csv(file_path)
    # print(f'motion {motion} header: {curr_df.columns}')
    gyro_data = curr_df[gyro_triple]
    
    # add svd(gyro), norm(accel), norm(accel) - gravity:
    add_accel_norm(curr_df)
    add_gyro_svd(curr_df)
    
    # choose svd gyro:
    gyro_svd = curr_df['gyro svd'].to_numpy()

    # THR:
    gyro_svd_abs = np.abs(gyro_svd)
    gyro_svd_abs_max = max(gyro_svd_abs)
    LEVEL = PEAKS_FACTOR * gyro_svd_abs_max
    print(f'Peak threshold for {motion} : {LEVEL}')

    starts = flag_and_timer(gyro_svd)
    # three_samples_forward_output = [abs(gyro_svd[start + samples_forward])
    #                                 for start in starts]
    # three_samples_forward_output_min = min(three_samples_forward_output)
    # print(f'motion {motion}: {three_samples_forward_output_min}')
    # val_li.append(three_samples_forward_output_min)
    plt.figure()
    plt.plot(gyro_svd_abs, label='abs gyro svd')
    ax = plt.gca()

    curr_df[gyro_triple].plot(ax=ax)
    plt.title(f'SVD REDUCTION ABS, {motion}')
    ymin, ymax = plt.ylim()
    plt.vlines(starts, ymin, ymax, colors='red', lw=0.5,
                label='starts with peak ahead and continuation')
    # plt.vlines(monotonic_starts, 0, 4.0, colors='k', lw=0.5, label='monotonic starts')
    # plt.legend()
    # plt.figure()
    # curr_df['t'].plot()
    # plt.legend()
    # val_li_arr = np.array(val_li)
    # m = val_li_arr.mean()

# print(f'average per {samples_forward} samples forward is {m}')

# gyro_svd = curr_df['gyro svd'].to_numpy()
# accel_proper_norm =\
#     curr_df[f'accel norm g = {gravity_constant} reduced'].to_numpy()
# gyro_svd_integral = np.cumsum(gyro_svd)

    
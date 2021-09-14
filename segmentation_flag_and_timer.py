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

import os
from os import listdir
from os.path import isfile, join  # isfile - (file or directory (guru99))


# Hyperparams for segmentation_with_max_in_sliding_window
WINDOW_SIZE = 20
GYRO_SVD_FACTOR = 1.5
PEAKS_FACTOR = 0.3

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
    df['accel norm g = 9.8 reduced'] = df['accel norm'] - 9.8


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
    plt.rcParams['lines.linewidth'] = 0.2
    plt.rcParams['lines.markersize'] = 1.0
    plt.rcParams['lines.marker'] = 'o'


def motion_cloud_plot(motion_df, starts, stops):
    add_gyro_svd(df)
    add_accel_norm(df)
    plt.figure()
    for  start, stop in zip(starts, stops):
        df['accel norm'].plot()
        df['gyro svd g = 9.8 reduced'].plot()


def threshold_crossed(ind, svd_gyro_abs):
    if svd_gyro_abs[ind+1]>THR and svd_gyro_abs[ind] < THR:
        return True
    else:
        return False
    
def peak_ahead(ind, gyro_svd_abs):
    ahead_samples = gyro_svd_abs[ind: ind + WAITING_TIME]

    if  any(ahead_samples>LEVEL):
        return True
    else:
        return False
    
def monotonic_increasing(ind, gyro_svd_abs):
    ahead_samples = gyro_svd_abs[ind: ind + 3]
    if  all(ahead_samples>THR):
        return True
    else:
        return False
    

WAITING_TIME = 20

# PART 1. PRINT THE FILES:
# user_name = 'stas1007'
user_name = 'stas'
dir_path = f'data/user_{user_name}'
li = list_of_files(dir_path)

di = dictionary_of_motions_files(li)
print(f'Files for segmentation {user_name}:')
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
none_gyro_svd_abs_max = np.max(none_gyro_svd_abs)
none_gyro_svd_abs_max_scaled = GYRO_SVD_FACTOR*none_gyro_svd_abs_max
print(f'\nThreshold defined by {GYRO_SVD_FACTOR}*None:' +
      f'\n{none_gyro_svd_abs_max_scaled}')



# For single motion:
motion = 'U'
csv_file_name = di[motion]
file_path = dir_path + '//' + csv_file_name
print(f'USER: {user_name}')
print('file choosed:', csv_file_name)
curr_df = pd.read_csv(file_path)
gyro_data = curr_df[gyro_triple]
curr_df['gyro svd'] = svd_reduction(gyro_data)[0]
clean_figs_run_plt_params()

# SVD REDUCTION: 
choosed_channels = gyro_triple + ['gyro svd']
curr_df[choosed_channels].plot(marker='o', ms=1.0, lw=.2)
plt.title(f'SVD REDUCTION, {motion}')
plt.legend(ncol=4)

# ABS SVD GYRO:
gyro_svd_abs = np.abs(curr_df['gyro svd'].to_numpy())


gyro_svd_abs_max = max(gyro_svd_abs)
LEVEL = 0.3* gyro_svd_abs_max
THR = 0.13
starts = []
for ind in range(len(gyro_svd_abs)-1):
    if (threshold_crossed(ind, gyro_svd_abs)
        and peak_ahead(ind, gyro_svd_abs)
        and  monotonic_increasing(ind, gyro_svd_abs)):
        starts.append(ind)
        print(ind)
    
plt.figure()
plt.plot(gyro_svd_abs, label = 'abs gyro svd')
plt.title(f'SVD REDUCTION ABS, {motion}')
xmin, xmax  = plt.xlim()
ymin, ymax = plt.ylim()
plt.hlines(none_gyro_svd_abs_max_scaled, xmin, xmax, label='gyro_svd_abs_threshold')
plt.vlines(starts, 0, 4.0, colors = 'red' , lw = 1.0)
plt.legend()
    
    
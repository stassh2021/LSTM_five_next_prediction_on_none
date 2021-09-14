# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:33:30 2021

@author: 6degt
data assumption:
[t, accel_triple, gyro_triple, magn_triple, label]

segmentaion with flag and timer:

start pointer :=  3 checks (CROSS + PEAK + CONTINUE)
stop pointer := starts  in reversed data

IDEA: working with abs(svd(gyro)):
[1] Find point for "leaving the None"  - pointer (CAUSE 1)
[2] Check:
    (1) is there 'big' value ahead (> PEAKS_FACTOR*MAX)?
    (2) Inreasing continued (SAMPLES FORWARD)
[3] Reverse data, find ends

ISSUE: PLATO -----/



"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode, median

import os
from os import listdir
from os.path import isfile, join  # isfile - (file or directory (guru99))


# Hyperparams for segmentation_with_flag_and_timer

SLEEP_TIME = 100  # avg time per motion, start found --> SLEEP,
# ALSO time for peak searching

# Start conditions:
GYRO_SVD_FACTOR = 1.3  # for exiting from None ('flag')
# for peaks forward
PEAKS_FACTOR = 0.3  # 20% of MAX cosidered a peak


SAMPLES_FORWARD = 3  # for increasing check
THR = 0.13  # for splashes in None


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


# adding features:
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
    df[f'accel norm g = {GRAVITY_CONSTANT} reduced'] =\
        df['accel norm'] - GRAVITY_CONSTANT


# PLOTS:
def plot(df, choosed_channels, title, file_name,
         start, stop):
    df[choosed_channels][start:stop].plot()
    plt.title(title)
    plt.xlabel(file_name)
    plt.legend()
    plt.show()


def plotter(csv_file_path):

    df = pd.read_csv(csv_file_path)
    file_name = csv_file_path.split('/')[-1]
    parsed_name = file_name.split('_')
    motion = parsed_name[-1].split('.')[0]
    title = (f'{user_name}, motion {motion}')
    start = 0
    stop = -1
    add_gyro_svd(df)

    # plot accel data
    # channels = accel_triple + ['accel norm']
    # plot(df, choosed_channels, title, file_name, start, stop)

    # plot gyro data
    choosed_channels = gyro_triple + ['gyro svd']  # + ['gyro norm']
    plot(df, choosed_channels, title,
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


# Flag and timer approach:
def threshold_crossed(ind, svd_gyro_abs, none_gyro_svd_threshold):
    if (svd_gyro_abs[ind+1] > none_gyro_svd_abs_max_scaled
            and svd_gyro_abs[ind] < none_gyro_svd_abs_max_scaled):
        return True
    else:
        return False


def peak_ahead(ind, gyro_svd_abs, peak_reached_threshold):
    ahead_samples = gyro_svd_abs[ind: ind + SLEEP_TIME]
    if any(ahead_samples > peak_reached_threshold):
        return True
    else:
        return False


def monotonic_increasing(ind, gyro_svd_abs, none_gyro_svd_threshold):
    next_ind = ind+1
    forward_samples = gyro_svd_abs[next_ind: next_ind + SAMPLES_FORWARD]
    # print(forward_samples, THR)
    THR = none_gyro_svd_abs_max_scaled
    if all(forward_samples > THR):
        return True
    else:
        return False



def flag_and_timer(gyro_svd):
    gyro_svd_abs = np.abs(gyro_svd)
    starts = []
    ind = 0
    while ind < len(gyro_svd)-1:
        if (threshold_crossed(ind, gyro_svd_abs, none_gyro_svd_threshold)
                and peak_ahead(ind, gyro_svd_abs, peak_reached_threshold)
                and monotonic_increasing(ind, gyro_svd_abs,
                                         none_gyro_svd_abs_max_scaled)):
            starts.append(ind)
            ind = ind + SLEEP_TIME
        else:
            ind += 1
    return np.array(starts)


def correct_labels(df, motion, starts, stops):
    labels_li = ['N']*len(df)
    for start, stop in zip(starts, stops):
        # print(start, stop)
        labels_li[start:stop] = [motion]*(stop-start)
    df['label'] = labels_li


# PART 1. PRINT THE FILES:
# user_name = 'stas1007'
user_name = 'stas'
# user_name = 'aya_1'
# user_name = 'aya'
dir_path = f'data/user_{user_name}'
train_files_dir_path = dir_path + '/' + 'train'
test_files_dir_path = dir_path + '/' + 'test'

# curr_dir_path = train_files_dir_path
curr_dir_path = test_files_dir_path

# li = list_of_files(train_files_dir_path)
li = list_of_files(curr_dir_path)

di = dictionary_of_motions_files(li)
print(f'Files for segmentation, {curr_dir_path}:')
for key, val in di.items():
    print(f'{key}: {val}')

# PART 2. WORK WITH NONE:
csv_file_name = di['N']
data_name = csv_file_name[:-13]
print(f'\nResuling DATA WILL NAMED BY None file: {data_name}')
csv_file_path = curr_dir_path + '/' + csv_file_name

clean_figs_run_plt_params()
plotter(csv_file_path)
df_none = pd.read_csv(csv_file_path)
add_gyro_svd(df_none)

# Threshold = Scale*None: (Scale = 1.0 - 2.5)
none_gyro_svd = df_none['gyro svd'].to_numpy()

none_gyro_svd_abs = np.abs(none_gyro_svd)
none_gyro_svd_abs_rounded = np.round(none_gyro_svd_abs, 2)

none_gyro_svd_df = pd.DataFrame(none_gyro_svd)
print('\nSOME STATISTICS FOR "NONE"')
print('None variance: ', none_gyro_svd_df.var())
none_gyro_svd_std = none_gyro_svd_df.std().to_numpy()[0]
print('None std: ', none_gyro_svd_std)
none_gyro_svd_abs_max = np.max(none_gyro_svd_abs)
none_gyro_svd_abs_max_scaled = GYRO_SVD_FACTOR*none_gyro_svd_abs_max
none_gyro_svd_threshold = none_gyro_svd_abs_max_scaled
none_gyro_svd_threshold = 4*none_gyro_svd_std


print(f'\nNone crossing threshold defined by 4*None_STD:' +
      f'\n{none_gyro_svd_threshold}')

# PART 3. SEGMENT THE MOTIONS FILES:

# For every single motion:
res_df = pd.DataFrame(columns=accel_triple+gyro_triple +['label'])
report_li = []

for motion in list_of_motions[1:]:
    # choose file:
    try:
        csv_file_name = di[motion]
    except KeyError:
        print(f'{motion} data not found')
    else:
        file_path = curr_dir_path + '/' + csv_file_name
        file_b_name = csv_file_name.split('.')[0]

        # read the data:
        curr_df = pd.read_csv(file_path)
        # add gyro svd:
        add_gyro_svd(curr_df)

        # choose svd gyro:
        gyro_svd = curr_df['gyro svd'].to_numpy()

        # define peaks thresholds:
        gyro_svd_abs = np.abs(gyro_svd)
        gyro_svd_abs_max = max(gyro_svd_abs)
        peak_reached_threshold = PEAKS_FACTOR * gyro_svd_abs_max
        print(f'Peak threshold ie {PEAKS_FACTOR}*NoneMax for {motion}' +
              f': {peak_reached_threshold}')

        # STARTS AND STOPS;
        # Find the starts:
        starts = flag_and_timer(gyro_svd)

        # Find the stops (reversed data starts):
        # Reverse:
        gyro_svd_reversed = gyro_svd[:: -1]
        gyro_svd_reversed_abs = np.abs(gyro_svd_reversed)

        # Find the reversed starts
        stops_ = flag_and_timer(gyro_svd_reversed)
        stops = len(gyro_svd_reversed) - stops_
        stops = stops[::-1]

        # Correct the labels:
        correct_labels(curr_df, motion, starts, stops)


        curr_df['gyro svd reversed'] = gyro_svd_reversed
        curr_df['gyro svd reversed abs'] = gyro_svd_reversed_abs
        curr_df['gyro svd abs'] = gyro_svd_abs

        # "DESICION PLOT":
        # channels = ['gyro svd abs']
        # curr_df[channels].plot()
        # plt.title(f'ABS(SVD(gyro)), MOTION {motion}')
        # ymin, ymax = plt.ylim()
        # xmin, xmax = plt.xlim()
        # plt.vlines(starts, ymin, ymax, colors='green',
        #            lw=0.5, label='starts by flag with timer')
        # plt.vlines(stops, ymin, ymax, colors='red', lw=0.5,
        #            label='reversed starts (stops)')
        # thr = round(none_gyro_svd_threshold, 2)
        # plt.hlines(thr, xmin, xmax, lw=0.5,
        #            label=f'None threshold: {thr}')
        # plt.legend()

        # "RESULT PLOT":
        curr_df[gyro_triple].plot()
        li = list(curr_df['label'])
        ax = plt.gca()
        ax1 = ax.twinx()
        ax1.plot(li)
        plt.title(f'SEGMENTED BY ABS(SVD(gyro)), MOTION {motion}')
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.vlines(starts, ymin, ymax, colors='green', lw=0.5,
                    label='starts by flag with timer')
        plt.vlines(stops, ymin, ymax, colors='red', lw=0.5,
                    label='reversed starts (stops)')
        thr = round(none_gyro_svd_threshold,2)
        plt.hlines(thr, xmin, xmax, lw=0.5,
                    label=f'None threshold: {thr}')
        plt.xlabel(csv_file_name)
        plt.legend()

        dst_file_name = file_b_name +'_corr_lbls.csv'
        dst_csv_file_path =\
            curr_dir_path + '/' + 'segmented' + '/' + dst_file_name
        curr_df.to_csv(dst_csv_file_path, index=False)
        report_li.append(f'{motion} corrected labels file saved to: \{dst_csv_file_path}')

print('\n')
for msg in report_li:
    print(msg)
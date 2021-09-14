# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:54:09 2020

@author: 6degt

viewer: plot the imu_ooutputs in the given folder
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('default')

coord_triple = ['x', 'y', 'z']
accel_triple = ['a' + i for i in coord_triple]
gyro_triple = ['g' + i for i in coord_triple]
magn_triple = ['m' + i for i in coord_triple]

imu_titles_9dof = accel_triple + gyro_triple + magn_triple


def clean_figs_run_plt_params():
    plt.close(fig='all')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 24
    plt. rcParams["legend.loc"] = 'best'


def list_files(dir_path):
    li_of_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    return li_of_files


def add_accel_gyro_norms(df):
    # Norms:
    df['accel norm'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gyro norm'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)
    df['accel norm g = 9.8 reduced'] = df['accel norm'] - 9.8


def plotter(csv_file_path, user_name):

    df = pd.read_csv(csv_file_path)
    file_name = csv_file_name.split('/')[-1]
    parsed_name = csv_file_path.split('_')
    motion = parsed_name[-1].split('.')[0]
    title = (f'{user_name}, motion {motion}')
    start = 0
    stop = -1
    add_accel_gyro_norms(df)
    # plot accel data
    # channels = accel_triple + ['accel norm']
    # df[channels][start:stop].plot(marker='o', ms=1.0, lw=.2)
    # plt.title(title)
    # plt.xlabel(file_name)
    # plt.legend()
    # plt.show()

    # plot gyro data
    choosed_channels = gyro_triple  # + ['gyro norm']
    df[choosed_channels][start:stop].plot(marker='o', ms=1.0, lw=.2)
    plt.title(title)
    plt.xlabel(file_name)
    plt.legend()
    plt.show()

    # plot magn data
    # df[magn_triple][start:stop].plot(marker='o', ms=1.0, lw=.2)
    # plt.title(title)
    # plt.xlabel(file_name)
    # plt.legend()
    # plt.show()

    # # histograms
    # choosed_channels = gyro_triple  # + ['gyro norm']
    # # choosed_channels = accel_triple + ['accel norm']
    # plt.figure()
    # for ch in choosed_channels:
    #     x = df[ch].to_numpy()
    #     plt.hist(x, density=True, bins=1000, label=ch)
    #     plt.title(title + ',' + ch)
    # plt.legend()


user_name = 'stas'
# user_name = 'stas'
print(os.getcwd())
dir_path = f'data/user_{user_name}'
csv_files_names = list_files(dir_path)

print(csv_files_names)

clean_figs_run_plt_params()

choosed_motions = ['N', 'U', 'D', 'L', 'R', 'T', 'C']
# choosed_motions = ['N']

# choosed_motions = ['table', 'IdleHand', 'UpDown10reps']
# choosed_motions = choosed_motions[5]
for csv_file_name in csv_files_names:
    parsed_name = csv_file_name.split('_')
    motion = parsed_name[-1].split('.')[0]
    if motion in choosed_motions:
        csv_file_path = dir_path + '\\' + csv_file_name
        plotter(csv_file_path, user_name)




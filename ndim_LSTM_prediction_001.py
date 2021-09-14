# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:47:34 2021

@author: 6degt

RECEVE THE DATA

data assumption:
    [t, ax, ay, az, gx, gy, gz, mx, my, mz]
    
Experiment 1:
    Predict Motion Plot
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split


# overwriting the model graph:
import tensorflow as tf
tf.compat.v1.reset_default_graph()

import datetime
import os

coord_triple = ['x', 'y', 'z']
accel_triple = ['a' + i for i in coord_triple]
gyro_triple = ['g' + i for i in coord_triple]
magn_triple = ['m' + i  for i in coord_triple]

WINDOW_SIZE = 10
steps_for_prediction = 3


def sliding_window(df):
    channels = accel_triple + gyro_triple
    nd_array = df[channels].to_numpy()
    windows_li = []
    labels_li = []
    for t in range(len(nd_array) - WINDOW_SIZE - steps_for_prediction):
        window = nd_array[t: t+WINDOW_SIZE]
        label = nd_array[t+WINDOW_SIZE + steps_for_prediction]
        windows_li.append(window)
        labels_li.append(label)
    return windows_li, labels_li


def model_constructor(n_features = 6, n_neurons = 2*WINDOW_SIZE):
    print('Hyperparams: ...')
    model = Sequential()
    model.add(LSTM(n_neurons,
              batch_input_shape=(n_batch, None, n_features), stateful=False))
    model.add(Dense(6, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()
    return model

def return_curr_time():
    now = datetime.datetime.now()  # 'now'  object
    date_time = now.strftime("%d_%m_%Y_%H_%M")  # 05_01_2019_14_45
    return date_time

def save_struct_and_weights(model):
    model_json = model.to_json()
    curr_date_time = return_curr_time()
  
    with open(f'models/model_structure_{curr_date_time}.json', 'w') as f:
        f.write(model_json)
    model.save_weights(f'models/weights_{curr_date_time}.h5')
    print(f'Weights, net saved to models directory')

def add_axis_for_prediction(windows_li):
    windows_tensors_li = []
    for window in windows_li:
        window_arr = np.array(window)
        window_tensor = window_arr[np.newaxis, :]
        windows_tensors_li.append(window_tensor)
    return windows_tensors_li

def clean_figs_run_plt_params():
    plt.close(fig='all')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 24
    plt.rcParams["legend.loc"] = 'best'
    plt.rcParams['lines.linewidth'] = 1.0

    
# Data:   
user_name = 'stasNone'
dir_path = 'data' + '/' + f'user_{user_name}' 
# csv_file_name_train ='imu_output_recording_12_07_2020_10_56_motion_U_corr_lbls.csv'
# csv_file_name_test = 'imu_output_recording_09_09_2020_20_23_motion_U_corr_lbls.csv'
csv_file_name_train = 'imu_recording_quat_ypr_09_02_2021_17_43_motion_N1.csv'
csv_file_name_test = 'imu_recording_quat_ypr_09_02_2021_17_44_motion_N2.csv'
csv_file_path_train = dir_path + '/' + 'train/segmented' + '/' + csv_file_name_train
csv_file_path_test = dir_path + '/' + 'test/segmented' + '/' + csv_file_name_test 


csv_file_path_train = dir_path +  '/' + csv_file_name_train
csv_file_path_test = dir_path +  '/' + csv_file_name_test 

train_df = pd.read_csv(csv_file_path_train)
test_df = pd.read_csv(csv_file_path_test)

n_features = 6
n_epoch = 20
n_batch = 1
windows_li, labels_li = sliding_window(train_df)

X = np.array(windows_li)
y = np.array(labels_li)
x_train, x_test, y_train, y_test = train_test_split(X, y)


# Model:
model = model_constructor()
history = model.fit(x_train, y_train,
                    epochs=n_epoch,
                    batch_size=n_batch,
                    verbose=2)

save_struct_and_weights(model)


windows_li1, labels_l1i = sliding_window(train_df)
x_for_prediction = np.array(windows_li1)
windows_tensors_li = add_axis_for_prediction(windows_li)

print(windows_tensors_li[0].shape)
predictions_li = []
for window_tensor in windows_tensors_li: 
    prediction = model.predict(window_tensor)
    predictions_li.append(prediction)

sample_ = x_test[0]
sample = sample_[np.newaxis, :]
pr = model.predict(sample)

print(len(predictions_li))
predictions_li = [np.array([[0, 0, 0, 0, 0, 0]])]*(WINDOW_SIZE+steps_for_prediction) + predictions_li
print(predictions_li[0].shape)

predictions_arr = np.vstack(predictions_li)


clean_figs_run_plt_params()
# ax
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 0], label = 'prediction ax')
chnls = ['ax']
train_df[chnls].plot(ax=ax)
plt.show()

# ay
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 1], label = 'prediction ay')
chnls = ['ay']
train_df[chnls].plot(ax=ax)
plt.show()

# az
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 2], label = 'prediction az')
chnls = ['az']
train_df[chnls].plot(ax=ax)
plt.show()

# gx
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 3], label = 'prediction gx')
chnls = ['gx']
train_df[chnls].plot(ax=ax)
plt.show()

# gy
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 4], label = 'prediction gy')
chnls = ['gy']
train_df[chnls].plot(ax=ax)
plt.show()

# gz
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 5], label = 'prediction gz')
chnls = ['gz']
train_df[chnls].plot(ax=ax)
plt.show()


windows_li1, labels_l1i = sliding_window(test_df)
x_for_prediction = np.array(windows_li1)
windows_tensors_li = add_axis_for_prediction(windows_li1)

print(windows_tensors_li[0].shape)
predictions_li = []
for window_tensor in windows_tensors_li: 
    prediction = model.predict(window_tensor)
    predictions_li.append(prediction)

# sample_ = x_test[0]
# sample = sample_[np.newaxis, :]
# # x = np.array([1, x_test[0]])
# pr = model.predict(sample)

print(len(predictions_li))
predictions_li = [np.array([[0, 0, 0, 0, 0, 0]])]*(WINDOW_SIZE + steps_for_prediction) + predictions_li


predictions_arr = np.vstack(predictions_li)
print(predictions_arr[0].shape)

# ax
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 0], label = 'prediction ax')
chnls = ['ax']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()

# ay
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 1], label = 'prediction ay')
chnls = ['ay']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()

# az
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 2], label = 'prediction az')
chnls = ['az']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()

# gx
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 3], label = 'prediction gx')
chnls = ['gx']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()

# gy
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 4], label = 'prediction gy')
chnls = ['gy']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()

# gz
fig, ax = plt.subplots()
ax.plot(predictions_arr[:, 5], label = 'prediction gz')
chnls = ['gz']
test_df[chnls].plot(ax=ax)
plt.legend()
plt.show()


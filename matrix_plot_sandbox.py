# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 14:26:54 2021

@author: 6degt
"""


import matplotlib.pyplot as plt
import numpy as np

arr_to_plot = np.array([[1.0, 6.0, 11.0],
                        [2.0, 7.0, 12.0],
                        [3.0, 8.0, 13.0],
                        [4.0, 9.0, 14.0],
                        [5.0, 10.0, 15.0]])

plt.figure()
plt.title('plot draw columns')
plt.plot(arr_to_plot)
plt.legend(['col1', 'col2', 'col3'])
plt.grid()

# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:45:54 2024

@author: user
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pickle
import scipy.stats as st

with open('all_y', 'rb') as fp:
    all_y = pickle.load(fp)
#all_y = np.array(all_y_all[:][0]) 
len_item = len(all_y[0])

def derivative(arr):
    return arr[1:] - arr[:-1], (arr[1:] - arr[:-1]).mean()

log_origin = np.abs(all_y[0])
log_origin = np.log(log_origin)
plt.figure()
#plt.plot(log_origin)
plt.plot(log_origin)
slope, b, _, _, _ = st.linregress(range(len_item), log_origin)
y_predict = slope * (np.array(range(len_item))) + b
plt.plot(range(len_item), y_predict)
abs_diff_mean = np.mean(np.absolute(log_origin - y_predict))

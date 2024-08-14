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
import matplotlib
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.patches import ConnectionPatch

font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

with open('all_y', 'rb') as fp:
    all_y = pickle.load(fp)
#all_y = np.array(all_y_all[:][0]) 
len_item = len(all_y[0])

def derivative(arr):
    return arr[1:] - arr[:-1], (arr[1:] - arr[:-1]).mean()

fig, ax = plt.subplots(1,2,figsize=(10,3))
plt.subplots_adjust(top=0.832,
bottom=0.087,
left=0.042,
right=0.997,
hspace=0.19,
wspace=0.149)
#score = np.log(np.abs(all_y[0]))
ax[0].set_title('(a) Recommendation Score', fontsize=25)
ax[1].set_title('(b) First Derivative', fontsize=25)

axins0 = ax[0].inset_axes((0.3, 0.1, 0.5, 0.35))
axins0.set_xticks([])
axins0.set_yticks([])

axins1 = ax[1].inset_axes((0.3, 0.1, 0.5, 0.35))
axins1.set_xticks([])
axins1.set_yticks([])


for i in range(180):
    score = all_y[i]
    ax[0].plot(score)
    axins0.plot(score)
    
    first_derivative, _ = derivative(score)
    ax[1].plot(first_derivative)
    axins1.plot(first_derivative)
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0].grid(linestyle='--',alpha=0.5)

axins0.set_xlim(-30, 700)
axins0.set_ylim(-1, 10)
sx = [-30, 700, 700, -30, -30]
sy = [-1, -1, 10, 10, -1]
ax[0].plot(sx, sy, "black")

con = ConnectionPatch(xyA=(-30,10), xyB=(-30,-1), coordsA="data", coordsB="data", axesA=axins0, axesB=ax[0])
axins0.add_artist(con)
con = ConnectionPatch(xyA=(700,10), xyB=(700,-1), coordsA="data", coordsB="data", axesA=axins0, axesB=ax[0])
axins0.add_artist(con)

ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[1].grid(linestyle='--',alpha=0.5)

axins1.set_xlim(-30, 700)
axins1.set_ylim(-0.2, 0.025)
sx = [-30, 700, 700, -100, -100]
sy = [-0.2, -0.2, 0.025, 0.025, -0.2]
ax[1].plot(sx, sy, "black")

con = ConnectionPatch(xyA=(-30,0.025), xyB=(-30,-0.2), coordsA="data", coordsB="data", axesA=axins1, axesB=ax[1])
axins1.add_artist(con)
con = ConnectionPatch(xyA=(700,0.025), xyB=(700,-0.2), coordsA="data", coordsB="data", axesA=axins1, axesB=ax[1])
axins1.add_artist(con)




plt.subplot_tool()
plt.show()

# fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# # plot a 3D wireframe like in the example mplot3d/wire3d_demo
# X = np.arange(0, len(all_y[0]))
# Y = np.arange(0, len(all_y[:2]))
# Y, X = np.meshgrid(X, Y)
# Z = np.array(all_y[:2])
# ax.plot_wireframe(X,Z,Y, rstride=10, cstride=10)

# plt.show()

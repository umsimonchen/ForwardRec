# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:20:25 2023

@author: simon
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:04:02 2023

@author: simon
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#plt.rcParams['font.sans-serif'] = ['SimHei']

font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

# =============================================================================
# labels = ['1', '2', '3', '4', '5']
# prec = [0.1732, 0.1953, 0.1787, 0.1763, 0.1676]
# rec = [0.1817, 0.2004, 0.1844, 0.1834, 0.1693]
# f1 = [0.1773, 0.1978, 0.1815, 0.1798, 0.1685]
# ndcg = [0.2265, 0.2532, 0.2305, 0.227, 0.2135]
# 
# x = np.arange(len(labels))  # the label locations
# fig, ax = plt.subplots(figsize=(8, 8))
# plt.subplots_adjust(top=0.779,
#     bottom=0.075,
#     left=0.125,
#     right=0.984,
#     hspace=0.205,
#     wspace=0.214)
# line1 = ax.plot(x, prec, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line2 = ax.plot(x, rec, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line3 = ax.plot(x, f1, label='F1', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line4 = ax.plot(x, ndcg, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Layer')
# ax.set_ylabel('Performance', fontsize=35)
# ax.set_title('LastFM', fontsize=35)
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.grid(linestyle='--',alpha=0.5)
# 
# 
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()
# =============================================================================

# =============================================================================
# labels = ['10', '20', '30', '40', '50']
# p_fm = [0.1915, 0.1842, 0.1953, 0.1869, 0.1842]
# r_fm = [0.1950, 0.1947, 0.2004, 0.1918, 0.1886]
# f_fm = [0.1932, 0.1893, 0.1978, 0.1893, 0.1864]
# n_fm = [0.2493, 0.2436, 0.2532, 0.2451, 0.2398]
# 
# p_fl = [0.0036, 0.0033, 0.0036, 0.0032, 0.0035]
# r_fl = [0.0041, 0.0032, 0.0047, 0.0038, 0.0040]
# f_fl = [0.0038, 0.0032, 0.0041, 0.0035, 0.0037]
# n_fl = [0.0048, 0.0043, 0.0051, 0.0043, 0.0044]
# 
# p_ye = [0.0064, 0.0063, 0.0066, 0.0064, 0.0060]
# r_ye = [0.0242, 0.0231, 0.0244, 0.0241, 0.0227]
# f_ye = [0.0102, 0.0099, 0.0103, 0.0101, 0.0095]
# n_ye = [0.0154, 0.0147, 0.0153, 0.0152, 0.0143] 
# 
# x = np.arange(len(labels))  # the label locations
# fig, ax = plt.subplots(1,3,figsize=(15, 5))
# plt.subplots_adjust(top=0.715,
#                     bottom=0.065,
#                     left=0.070,
#                     right=0.99,
#                     hspace=0.170,
#                     wspace=0.225)
# line1 = ax[0].plot(x, p_fm, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line2 = ax[0].plot(x, r_fm, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line3 = ax[0].plot(x, f_fm, label='F1', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line4 = ax[0].plot(x, n_fm, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# 
# line5 = ax[1].plot(x, p_fl, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line6 = ax[1].plot(x, r_fl, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line7 = ax[1].plot(x, f_fl, label='F1', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line8 = ax[1].plot(x, n_fl, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# 
# line9 = ax[2].plot(x, p_ye, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line10 = ax[2].plot(x, r_ye, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line11 = ax[2].plot(x, f_ye, label='F1', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line12 = ax[2].plot(x, n_ye, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[0].set_ylabel('Performance', fontsize=35)
# ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0].set_title('LastFM', fontsize=35)
# ax[0].set_xticks(x)
# ax[0].set_xticklabels(labels)
# ax[0].grid(linestyle='--',alpha=0.5)
# ax[0].set_ylim([0.15,0.3])
# 
# ax[1].set_title('Flickr', fontsize=35)
# ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1].set_xticks(x)
# ax[1].set_xticklabels(labels)
# ax[1].grid(linestyle='--',alpha=0.5)
# ax[1].set_ylim([0.0,0.008])
# 
# ax[2].set_title('Yelp', fontsize=35)
# ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[2].set_xticks(x)
# ax[2].set_xticklabels(labels)
# ax[2].grid(linestyle='--',alpha=0.5)
# ax[2].set_ylim([0.0,0.03])
# 
# handles, labels = ax[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()
# =============================================================================

labels = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1']
h_fm = [0.30611,0.32024,0.32297,0.3247,0.32544,0.32551,0.32452,0.3247,0.32284,0.32204]
p_fm = [0.13282,0.13895,0.14013,0.14089,0.14121,0.14124,0.14081,0.14089,0.14008,0.13973]
r_fm = [0.29698,0.311,0.314,0.31647,0.31692,0.31732,0.31589,0.31658,0.31508,0.31432]
n_fm = [0.28793,0.29959,0.30161,0.30451,0.30443,0.30543,0.30446,0.30453,0.30433,0.30276]

h_ab = [0.14955,0.15384,0.1558,0.15512,0.15608,0.15505,0.15429,0.15348,0.15394,0.1528]
p_ab = [0.01692,0.01741,0.01763,0.01755,0.01766,0.01754,0.01746,0.01737,0.01742,0.01729]
r_ab = [0.14084,0.14609,0.14741,0.14824,0.14864,0.14854,0.14839,0.14702,0.14857,0.14805]
n_ab = [0.07843,0.0811,0.08176,0.08163,0.0821,0.08219,0.08152,0.08112,0.08126,0.08052]

h_db = [0.13909,0.147,0.14567,0.14376,0.14168,0.1405,0.13935,0.13796,0.1366,0.13644]
p_db = [0.07784,0.08226,0.08152,0.08045,0.07928,0.07863,0.07798,0.0772,0.07644,0.07635]
r_db = [0.18153,0.18553,0.1851,0.18426,0.18164,0.18274,0.18143,0.18182,0.17839,0.18093]
n_db = [0.15685,0.16226,0.15985,0.15852,0.15651,0.15694,0.15521,0.15469,0.15222,0.15438]

h_ye = [0.05992,0.0628,0.06267,0.0625,0.06238,0.06224,0.06202,0.0616,0.06173,0.0612]
p_ye = [0.03067,0.03214,0.03207,0.03199,0.03193,0.03185,0.03174,0.03153,0.03159,0.03132]
r_ye = [0.06882,0.07127,0.07119,0.07076,0.0707,0.07057,0.07023,0.0698,0.07011,0.06927]
n_ye = [0.05544,0.05789,0.05795,0.05768,0.05761,0.05755,0.05733,0.0569,0.05704,0.05657]

# x = np.arange(len(labels))  # the label locations
# fig, ax = plt.subplots(2,8,figsize=(30,8))
# plt.subplots_adjust(top=0.760,
#                     bottom=0.065,
#                     left=0.040,
#                     right=0.99,
#                     hspace=0.170,
#                     wspace=0.225)
# line1 = ax[0,0].plot(x, h_fm, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line2 = ax[0,1].plot(x, p_fm, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line3 = ax[1,0].plot(x, r_fm, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line4 = ax[1,1].plot(x, n_fm, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

# line5 = ax[0,2].plot(x, h_ab, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line6 = ax[0,3].plot(x, p_ab, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line7 = ax[1,2].plot(x, r_ab, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line8 = ax[1,3].plot(x, n_ab, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

# line9 = ax[0,4].plot(x, h_db, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# line10 = ax[0,5].plot(x, p_db, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line11 = ax[1,4].plot(x, r_db, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line12 = ax[1,5].plot(x, n_db, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

# line13 = ax[0,6].plot(x, h_ye, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(2))
# Line14 = ax[0,7].plot(x, p_ye, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(3))
# line15 = ax[1,6].plot(x, r_ye, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line16 = ax[1,7].plot(x, n_ye, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax[0,0].set_ylabel('Performance', fontsize=30)
# ax[1,0].set_ylabel('Performance', fontsize=30)
# ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# #ax[0,0].set_title('LastFM', fontsize=35)
# ax[0,0].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# #ax[0,0].set_xticklabels(labels)
# ax[0,0].grid(linestyle='--',alpha=0.5)
# #ax[0].set_ylim([0.15,0.3])

# #ax[0,1].set_title('Amazon-Beauty', fontsize=35)
# ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,1].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# #ax[0,1].set_xticklabels(labels)
# ax[0,1].grid(linestyle='--',alpha=0.5)
# #ax[1].set_ylim([0.0,0.008])

# #ax[0,2].set_title('Douban-Book', fontsize=35)
# ax[0,2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,2].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# #ax[0,2].set_xticklabels(labels)
# ax[0,2].grid(linestyle='--',alpha=0.5)
# #ax[2].set_ylim([0.0,0.03])

# #ax[0,3].set_title('Yelp2018', fontsize=35)
# ax[0,3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,3].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# #ax[0,3].set_xticklabels(labels)
# ax[0,3].grid(linestyle='--',alpha=0.5)
# #ax[2].set_ylim([0.0,0.03])

# ax[0,4].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,4].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[0,4].grid(linestyle='--',alpha=0.5)

# ax[0,5].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,5].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[0,5].grid(linestyle='--',alpha=0.5)

# ax[0,6].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,6].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[0,6].grid(linestyle='--',alpha=0.5)

# ax[0,7].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,7].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[0,7].grid(linestyle='--',alpha=0.5)

# ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,0].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,0].grid(linestyle='--',alpha=0.5)

# ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,1].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,1].grid(linestyle='--',alpha=0.5)

# ax[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,2].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,2].grid(linestyle='--',alpha=0.5)

# ax[1,3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,3].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,3].grid(linestyle='--',alpha=0.5)

# ax[1,4].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,4].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,4].grid(linestyle='--',alpha=0.5)

# ax[1,5].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,5].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,5].grid(linestyle='--',alpha=0.5)

# ax[1,6].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,6].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,6].grid(linestyle='--',alpha=0.5)

# ax[1,7].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,7].set_xticks(x, ['0.01','','0.03','','0.05','','0.07','','0.09',''])
# ax[1,7].grid(linestyle='--',alpha=0.5)

# fig.text(0.13,0.8,"LastFM",fontsize=30)
# fig.text(0.35,0.8,"Amazon-Beauty",fontsize=30)
# fig.text(0.60,0.8,"Douban-Book",fontsize=30)
# fig.text(0.85,0.8,"Yelp2018",fontsize=30)

# handles_h, labels_h = ax[0,0].get_legend_handles_labels()
# handles_p, labels_p = ax[0,1].get_legend_handles_labels()
# handles_r, labels_r = ax[1,0].get_legend_handles_labels()
# handles_n, labels_n = ax[1,1].get_legend_handles_labels()

# fig.legend(handles_h+handles_p+handles_r+handles_n, labels_h+labels_p+labels_r+labels_n, loc='upper center', ncol=4, fontsize=20)
# plt.subplot_tool()
# #plt.savefig('tau.pdf', format='pdf')
# plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots(2,2,figsize=(13,9))
plt.subplots_adjust(top=0.87,
bottom=0.03,
left=0.085,
right=0.92,
hspace=0.25,
wspace=0.33)
# line1 = ax[0,0].plot(x, h_fm, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# line2 = ax[0,1].plot(x, h_ab, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# line3 = ax[1,0].plot(x, h_db, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
# line4 = ax[1,1].plot(x, h_ye, label='Hit Rate', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

# ax0 = ax[0,0].twinx()
# ax1 = ax[0,1].twinx()
# ax2 = ax[1,0].twinx()
# ax3 = ax[1,1].twinx()
# line5 = ax0.plot(x, n_fm, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line6 = ax1.plot(x, n_ab, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line7 = ax2.plot(x, n_db, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
# line8 = ax3.plot(x, n_ye, label='NDCG', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))

line1 = ax[0,0].plot(x, p_fm, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line2 = ax[0,1].plot(x, p_ab, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line3 = ax[1,0].plot(x, p_db, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))
line4 = ax[1,1].plot(x, p_ye, label='Precision', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(5))

ax0 = ax[0,0].twinx()
ax1 = ax[0,1].twinx()
ax2 = ax[1,0].twinx()
ax3 = ax[1,1].twinx()
line5 = ax0.plot(x, r_fm, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line6 = ax1.plot(x, r_ab, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line7 = ax2.plot(x, r_db, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))
line8 = ax3.plot(x, r_ye, label='Recall', marker='8', linewidth=4, markersize=10, color=plt.cm.Set3(4))

# ax[0,0].set_ylabel('Hit Rate', fontsize=30)
# ax[1,0].set_ylabel('Hit Rate', fontsize=30)
# ax1.set_ylabel('NDCG', fontsize=30)
# ax3.set_ylabel('NDCG', fontsize=30)

ax[0,0].set_ylabel('Precision', fontsize=30)
ax[1,0].set_ylabel('Precision', fontsize=30)
ax1.set_ylabel('Recall', fontsize=30)
ax3.set_ylabel('Recall', fontsize=30)


ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0,0].set_title('LastFM', fontsize=35)
ax[0,0].set_xticks(x, ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1'])
#ax[0].set_xticklabels(labels)
ax[0,0].grid(linestyle='--',alpha=0.5)
#ax[0].set_ylim([0.15,0.3])

ax[0,1].set_title('Amazon-Beauty', fontsize=35)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0,1].set_xticks(x, ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1'])
#ax[0,1].set_xticklabels(labels)
ax[0,1].grid(linestyle='--',alpha=0.5)
#ax[1].set_ylim([0.0,0.008])

ax[1,0].set_title('Douban-Book', fontsize=35)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1,0].set_xticks(x, ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1'])
#ax[0,2].set_xticklabels(labels)
ax[1,0].grid(linestyle='--',alpha=0.5)
#ax[2].set_ylim([0.0,0.03])

ax[1,1].set_title('Yelp2018', fontsize=35)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1,1].set_xticks(x, ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1'])
#ax[0,3].set_xticklabels(labels)
ax[1,1].grid(linestyle='--',alpha=0.5)
#ax[2].set_ylim([0.0,0.03])

handles_h, labels_h = ax[0,0].get_legend_handles_labels()
handles_d, labels_d = ax0.get_legend_handles_labels()
fig.legend(handles_h+handles_d, labels_h+labels_d, loc='upper center', ncol=4, fontsize=20)
plt.subplot_tool()
#plt.savefig('tau.pdf', format='pdf')
plt.show()


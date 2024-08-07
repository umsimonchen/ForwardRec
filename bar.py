# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:52:22 2023

@author: simon
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:31:47 2023

@author: simon
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

#plt.rcParams['font.sans-serif'] = ['SimHei']

font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

labels = ['LastFM','Beauty', 'Douban', 'Yelp2018']
lightgcn_tpe = [0.62,2.95,5.34,24.61]
ltgnn_tpe = [1.09,3.16,10.42,30.78]
mixgcf_tpe = [0.67,2.73,5.29,22.84]
sgl_tpe = [1.38,4.67,12.64,55.91]
ff_tpe = [1.17,3.79,10.27,44.05]

lightgcn_e = [424,257,270,173]
ltgnn_e = [960,777,1081,1197]
mixgcf_e = [602,412,413,341]
sgl_e = [122,87,86,69]
ff_e = [114,71,76,61]

lightgcn_tt = [263,758,1442,4258]
ltgnn_tt = [1046,2455,11264,36844]
mixgcf_tt = [403,1125,2185,7788]
sgl_tt = [168,406,1087,3858]
ff_tt = [133,269,781,2687]

# lightgcn_tpe = [np.log10(0.62),np.log10(2.95),np.log10(5.34),np.log10(24.61)]
# ltgnn_tpe = [np.log10(1.09),np.log10(3.16),np.log10(10.42),np.log10(30.78)]
# mixgcf_tpe = [np.log10(0.67),np.log10(2.73),np.log10(5.29),np.log10(22.84)]
# sgl_tpe = [np.log10(1.38),np.log10(4.67),np.log10(12.64),np.log10(55.91)]
# ff_tpe = [np.log10(1.17),np.log10(3.79),np.log10(10.27),np.log10(44.05)]

# lightgcn_e = [np.log10(424),np.log10(257),np.log10(270),np.log10(173)]
# ltgnn_e = [np.log10(960),np.log10(777),np.log10(1081),np.log10(1197)]
# mixgcf_e = [np.log10(602),np.log10(412),np.log10(413),np.log10(341)]
# sgl_e = [np.log10(122),np.log10(87),np.log10(86),np.log10(69)]
# ff_e = [np.log10(114),np.log10(71),np.log10(76),np.log10(61)]

# lightgcn_tt = [np.log10(263),np.log10(758),np.log10(1442),np.log10(4258)]
# ltgnn_tt = [np.log10(1046),np.log10(2455),np.log10(11264),np.log10(36844)]
# mixgcf_tt = [np.log10(403),np.log10(1125),np.log10(2185),np.log10(7788)]
# sgl_tt = [np.log10(168),np.log10(406),np.log10(1087),np.log10(3858)]
# ff_tt = [np.log10(133),np.log10(269),np.log10(781),np.log10(2687)]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars
fig1, ax1 = plt.subplots(1,3,figsize=(15, 6))
plt.subplots_adjust(top=0.715,
                    bottom=0.065,
                    left=0.070,
                    right=0.99,
                    hspace=0.170,
                    wspace=0.330)
rects1 = ax1[0].bar(x - width/3*4, lightgcn_tpe, width/3*2, label='LightGCN', color=plt.cm.Set3(6))
rects2 = ax1[0].bar(x - width/3*2, ltgnn_tpe, width/3*2, label='LTGNN', color=plt.cm.Set3(2))
rects3 = ax1[0].bar(x, mixgcf_tpe, width/3*2, label='MixGCF', color=plt.cm.Set3(3))
rects4 = ax1[0].bar(x + width/3*2, sgl_tpe, width/3*2, label='SGL', color=plt.cm.Set3(4))
rects5 = ax1[0].bar(x + width/3*4, ff_tpe, width/3*2, label='ForwardRec', color=plt.cm.Set3(5))

rects6 = ax1[1].bar(x - width/3*4, lightgcn_e, width/3*2, label='LightGCN', color=plt.cm.Set3(6))
rects7 = ax1[1].bar(x - width/3*2, ltgnn_e, width/3*2, label='LTGNN', color=plt.cm.Set3(2))
rects8 = ax1[1].bar(x, mixgcf_e, width/3*2, label='MixGCF', color=plt.cm.Set3(3))
rects9 = ax1[1].bar(x + width/3*2, sgl_e, width/3*2, label='SGL', color=plt.cm.Set3(4))
rects10 = ax1[1].bar(x + width/3*4, ff_e, width/3*2, label='ForwardRec', color=plt.cm.Set3(5))

rects11 = ax1[2].bar(x - width/3*4, lightgcn_tt, width/3*2, label='LightGCN', color=plt.cm.Set3(6))
rects12 = ax1[2].bar(x - width/3*2, ltgnn_tt, width/3*2, label='LTGNN', color=plt.cm.Set3(2))
rects13 = ax1[2].bar(x, mixgcf_tt, width/3*2, label='MixGCF', color=plt.cm.Set3(3))
rects14 = ax1[2].bar(x + width/3*2, sgl_tt, width/3*2, label='SGL', color=plt.cm.Set3(4))
rects15 = ax1[2].bar(x + width/3*4, ff_tt, width/3*2, label='ForwardRec', color=plt.cm.Set3(5))

# def exp_format(x,pos):
#     return f'{x:.1e}'
# ax1[0].yaxis.set_major_formatter(FuncFormatter(exp_format))
# ax1[1].yaxis.set_major_formatter(FuncFormatter(exp_format))
# ax1[2].yaxis.set_major_formatter(FuncFormatter(exp_format))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1[0].set_ylabel('Time (seconds)', fontsize=30)
ax1[0].set_title('Time per Epoch', fontsize=30)
ax1[0].set_xticks(x)
ax1[0].set_xticklabels(labels)
ax1[0].set_yscale('log')
ax1[0].grid(linestyle='--',alpha=0.5)

ax1[1].set_ylabel('Epoch Number', fontsize=30)
ax1[1].set_title('Epoch', fontsize=30)
ax1[1].set_xticks(x)
ax1[1].set_xticklabels(labels)
ax1[1].set_yscale('log')
ax1[1].grid(linestyle='--',alpha=0.5)

ax1[2].set_ylabel('Time (seconds)', fontsize=30)
ax1[2].set_title('Total Time', fontsize=30)
ax1[2].set_xticks(x)
ax1[2].set_xticklabels(labels)
ax1[2].set_yscale('log')
ax1[2].grid(linestyle='--',alpha=0.5)

handles, labels = ax1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', ncol=5, fontsize=20)

plt.subplot_tool()
plt.show()

# =============================================================================
# labels = ['P@10','R@10', 'F1@10', 'N@10']
# att_fm = [0.1953, 0.2004, 0.1978, 0.2532]
# avg_fm = [0.1806, 0.1860, 0.1833, 0.2359]
# max_fm = [0.1652, 0.1691, 0.1672, 0.2115]
# cat_fm = [0.1105, 0.1132, 0.1187, 0.1481]
# 
# att_fl = [0.0036,	0.0047,	0.0041,	0.0051]
# avg_fl = [0.0033,	0.0039,	0.0036,	0.0043]
# max_fl = [0.0029,	0.0034,	0.0031,	0.0038]
# cat_fl = [0.0030,	0.0039,	0.0034,	0.0042]
# 
# att_ye = [0.0066,	0.0244,	0.0103,	0.0153]
# avg_ye = [0.0062,	0.0230,	0.0098,	0.0146]
# max_ye = [0.0061,	0.0244,	0.0098,	0.0153]
# cat_ye = [0.0057,	0.0223,	0.0090,	0.0140]
# 
# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
# fig1, ax1 = plt.subplots(1,3,figsize=(15, 5))
# plt.subplots_adjust(top=0.715,
#                     bottom=0.065,
#                     left=0.070,
#                     right=0.99,
#                     hspace=0.170,
#                     wspace=0.225)
# rects1 = ax1[0].bar(x - 3*width/2, att_fm, width, label='Attention', color=plt.cm.Set3(2))
# rects2 = ax1[0].bar(x - width/2, avg_fm, width, label='Mean', color=plt.cm.Set3(3))
# rects3 = ax1[0].bar(x + width/2, max_fm, width, label='Max', color=plt.cm.Set3(4))
# rects4 = ax1[0].bar(x + 3*width/2, cat_fm, width, label='Concatenation', color=plt.cm.Set3(5))
# 
# rects5 = ax1[1].bar(x - 3*width/2, att_fl, width, label='Attention', color=plt.cm.Set3(2))
# rects6 = ax1[1].bar(x - width/2, avg_fl, width, label='Mean', color=plt.cm.Set3(3))
# rects7 = ax1[1].bar(x + width/2, max_fl, width, label='Max', color=plt.cm.Set3(4))
# rects8 = ax1[1].bar(x + 3*width/2, cat_fl, width, label='Concatenation', color=plt.cm.Set3(5))
# 
# rects9 = ax1[2].bar(x - 3*width/2, att_ye, width, label='Attention', color=plt.cm.Set3(2))
# rects10 = ax1[2].bar(x - width/2, avg_ye, width, label='Mean', color=plt.cm.Set3(3))
# rects11 = ax1[2].bar(x + width/2, max_ye, width, label='Max', color=plt.cm.Set3(4))
# rects12 = ax1[2].bar(x + 3*width/2, cat_ye, width, label='Concatenation', color=plt.cm.Set3(5))
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax1[0].set_ylabel('Performance', fontsize=35)
# ax1[0].set_title('LastFM', fontsize=35)
# ax1[0].set_xticks(x)
# ax1[0].set_xticklabels(labels)
# ax1[0].grid(linestyle='--',alpha=0.5)
# 
# ax1[1].set_title('Flickr', fontsize=35)
# ax1[1].set_xticks(x)
# ax1[1].set_xticklabels(labels)
# ax1[1].grid(linestyle='--',alpha=0.5)
# 
# ax1[2].set_title('Yelp', fontsize=35)
# ax1[2].set_xticks(x)
# ax1[2].set_xticklabels(labels)
# ax1[2].grid(linestyle='--',alpha=0.5)
# 
# handles, labels = ax1[0].get_legend_handles_labels()
# fig1.legend(handles, labels, loc='upper center', ncol=5, fontsize=20)
# 
# plt.subplot_tool()
# plt.show()
# 
# =============================================================================





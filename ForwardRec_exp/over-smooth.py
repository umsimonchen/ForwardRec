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
import pickle

#plt.rcParams['font.sans-serif'] = ['SimHei']

font = {'family' : 'Calibri',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

LightGCN = {'lastfm':[[0.30072,0.30711,0.30742,0.3094], [0.27498,0.28364,0.28376,0.28756]], 
            'amazon-beauty': [[0.12975,0.13743,0.13988,0.14182], [0.07005,0.07351,0.07574,0.07643]], 
            'douban-book': [[0.10415,0.11097,0.11164,0.11272], [0.11481,0.12644,0.12605,0.12928]], 
            'yelp2018': [[0.05054,0.05185,0.05266,0.05311], [0.04682,0.04819,0.0491,0.04942]],
            'gowalla': [[0.14606,0.15265,0.15656,0.15802], [0.14015,0.14741,0.15057,0.15199]]}

LGCN = {'lastfm':[[0.28691,0.28542,0.28542,0.28121,], [0.26135,0.25885,0.26019,0.25765,]] , 
        'amazon-beauty': [[0.11881,0.11866,0.11795,0.11783,], [0.06199,0.06291,0.0632,0.06344,]], 
        'douban-book': [[0.09875,0.09985,0.10035,0.10011,], [0.10705,0.10742,0.10828,0.10726,]],
        'yelp2018': [[0.04625,0.04673,0.04723,0.04769,], [0.04229,0.0426,0.04313,0.04371,]], 
        'gowalla': [[0.13675,0.13759,0.1371,0.13515,], [0.12912,0.13027,0.12974,0.12749,]]}
                  
DHCF = {'lastfm': [[0.28226,0.29131,0.29354,0.29447,], [0.25584,0.26365,0.26769,0.26792,]],
            'amazon-beauty': [[0.12181,0.12564,0.12609,0.12249,], [0.06243,0.064,0.06475,0.06349,]], 
            'douban-book': [[0.09349,0.09569,0.09683,0.09619,], [0.10332,0.10664,0.11029,0.1082,]], 
            'yelp2018': [[0.04444,0.04443,0.04487,0.04595,], [0.04066,0.04096,0.04099,0.04214,]], 
            'gowalla': [[0.12797,0.12819,0.1272,0.12705,], [0.12022,0.1214,0.11943,0.11919,]]}
                  
SGL = {'lastfm': [[0.30661,0.31107,0.31665,0.31677,], [0.28309,0.28761,0.2913,0.29243,]],
        'amazon-beauty': [[0.14839,0.14965,0.15117,0.15215,], [0.07884,0.08046,0.08052,0.08139,]], 
        'douban-book': [[0.12414,0.12544,0.12626,0.12705,], [0.14749,0.15001,0.15143,0.15226,]], 
        'yelp2018': [[0.05631,0.05773,0.05846,0.05849,], [0.05219,0.05387,0.05469,0.05474,]], 
        'gowalla': [[0.1619,0.16975,0.17102,0.17134,], [0.15308,0.16199,0.16292,0.16359,]]}
                  
MixGCF = {'lastfm': [[0.30413,0.30791,0.30909,0.30723,], [0.28004,0.284,0.28382,0.28391,]],
            'amazon-beauty': [[0.13363,0.13786,0.14164,0.13967,], [0.07104,0.07439,0.07543,0.07493,]], 
            'douban-book': [[0.10589,0.11,0.11196,0.11241,], [0.1154,0.12109,0.12383,0.12422,]], 
            'yelp2018': [[0.05195,0.05411,0.05444,0.05488,], [0.0481,0.05034,0.05063,0.05092,]], 
            'gowalla': [[0.15065,0.15743,0.16046,0.16192,], [0.14294,0.1496,0.15277,0.15423,]]}

ForwardRec = {'lastfm': [[0.3190, 0.3274, 0.3290, 0.3314]],
            'amazon-beauty': [[0.1584, 0.1614, 0.1634, 0.1674]], 
            'douban-book': [[0.1292, 0.1456, 0.1513, 0.1528]], 
            'yelp2018': [[0.0611, 0.0643, 0.0661, 0.0669]], 
            'gowalla': [[0.1694, 0.1865, 0.1892, 0.1902]]}

#-------------------------------------------------------------------------------------------------------------------
#Layer 1
# =============================================================================
# fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# plt.subplots_adjust(top=0.87,
# bottom=0.03,
# left=0.085,
# right=0.92,
# hspace=0.25,
# wspace=0.33)
# line1 = ax[0,0].plot(np.arange(len(training_record_FF[0][0][:199])), training_record_FF[0][0][:199], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax0 = ax[0,0].twinx()
# line2 = ax0.plot(np.arange(len(training_record_FF[0][1][:199])), training_record_FF[0][1][:199], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# line3 = ax[0,1].plot(np.arange(len(training_record_FF[1][0][:87])), training_record_FF[1][0][:87], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax1 = ax[0,1].twinx()
# line4 = ax1.plot(np.arange(len(training_record_FF[1][1][:87])), training_record_FF[1][1][:87], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# 
# line5 = ax[1,0].plot(np.arange(len(training_record_FF[2][0][:87])), training_record_FF[2][0][:87], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax2 = ax[1,0].twinx()
# line6 = ax2.plot(np.arange(len(training_record_FF[2][1][:87])), training_record_FF[2][1][:87], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# line7 = ax[1,1].plot(np.arange(len(training_record_FF[3][0][:75])), training_record_FF[3][0][:75], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax3 = ax[1,1].twinx()
# line8 = ax3.plot(np.arange(len(training_record_FF[3][1][:75])), training_record_FF[3][1][:75], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# 
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel('Layer')
# ax[0,0].set_ylabel('Hit Rate', fontsize=30)
# ax[1,0].set_ylabel('Hit Rate', fontsize=30)
# ax1.set_ylabel('Loss', fontsize=30)
# ax3.set_ylabel('Loss', fontsize=30)
# ax[0,0].set_title('LastFM', fontsize=30)
# ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[0,1].set_title('Amazon-Beauty', fontsize=30)
# ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,0].set_title('Douban-Book', fontsize=30)
# ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,1].set_title('Yelp2018', fontsize=30)
# ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# 
# #ax.set_xticks(x)
# #ax.set_xticklabels(labels)
# ax[0,0].grid(linestyle='--',alpha=0.5)
# ax[0,1].grid(linestyle='--',alpha=0.5)
# ax[1,0].grid(linestyle='--',alpha=0.5)
# ax[1,1].grid(linestyle='--',alpha=0.5)
# 
# 
# handles_h, labels_h = ax[0,0].get_legend_handles_labels()
# handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = fig.legend(handles_h+handles_l, labels_h+labels_l, loc='upper center', ncol=6, fontsize=20)
# for obj in leg.legendHandles:
#     obj.set_linewidth(6.0)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()
# =============================================================================

#------------------------------------------------------------------------------------------------------------
#Layer 2
# fig, ax = plt.subplots(2, 2, figsize=(13, 9))
# plt.subplots_adjust(top=0.87,
# bottom=0.03,
# left=0.085,
# right=0.92,
# hspace=0.25,
# wspace=0.33)

# line1 = ax[0,0].plot(np.arange(len(training_record_FF[0][0][199:])), training_record_FF[0][0][199:], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax0 = ax[0,0].twinx()
# line2 = ax0.plot(np.arange(len(training_record_FF[0][1][199:])), training_record_FF[0][1][199:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# line3 = ax[0,1].plot(np.arange(len(training_record_FF[1][0][87:])), training_record_FF[1][0][87:], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax1 = ax[0,1].twinx()
# line4 = ax1.plot(np.arange(len(training_record_FF[1][1][87:])), training_record_FF[1][1][87:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# line5 = ax[1,0].plot(np.arange(len(training_record_FF[2][0][88:])), training_record_FF[2][0][88:], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax2 = ax[1,0].twinx()
# line6 = ax2.plot(np.arange(len(training_record_FF[2][1][88:])), training_record_FF[2][1][88:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
# line7 = ax[1,1].plot(np.arange(len(training_record_FF[3][0][75:])), training_record_FF[3][0][75:], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
# ax3 = ax[1,1].twinx()
# line8 = ax3.plot(np.arange(len(training_record_FF[3][1][75:])), training_record_FF[3][1][75:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel('Layer')
# ax[0,0].set_ylabel('Hit Rate', fontsize=30)
# ax[1,0].set_ylabel('Hit Rate', fontsize=30)
# ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax1.set_ylabel('Loss', fontsize=30)
# ax3.set_ylabel('Loss', fontsize=30)
# ax[0,0].set_title('LastFM', fontsize=30)
# ax[0,1].set_title('Amazon-Beauty', fontsize=30)
# ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,0].set_title('Douban-Book', fontsize=30)
# ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1,1].set_title('Yelp2018', fontsize=30)
# ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# #ax.set_xticks(x)
# #ax.set_xticklabels(labels)
# ax[0,0].grid(linestyle='--',alpha=0.5)
# ax[0,1].grid(linestyle='--',alpha=0.5)
# ax[1,0].grid(linestyle='--',alpha=0.5)
# ax[1,1].grid(linestyle='--',alpha=0.5)


# handles_h, labels_h = ax[0,0].get_legend_handles_labels()
# handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = fig.legend(handles_h+handles_l, labels_h+labels_l, loc='upper center', ncol=6, fontsize=20)
# for obj in leg.legendHandles:
#     obj.set_linewidth(6.0)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()

#with compared methods 
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
plt.subplots_adjust(top=0.702,
bottom=0.162,
left=0.051,
right=0.995,
hspace=0.195,
wspace=0.215)

x = np.arange(1, 5)
line1_ForwardRec= ax[0].plot(x, ForwardRec['lastfm'][0], label='ForwardRec', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(6))
line1_LightGCN= ax[0].plot(x, LightGCN['lastfm'][0], label='LightGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(1))
line1_LGCN = ax[0].plot(x, LGCN['lastfm'][0], label='LGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(4))
line1_DHCF = ax[0].plot(x, DHCF['lastfm'][0], label='DHCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(3))
line1_SGL = ax[0].plot(x, SGL['lastfm'][0], label='SGL', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(2))
line1_MixGCF = ax[0].plot(x, MixGCF['lastfm'][0], label='MixGCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(7))
#ax0 = ax[0].twinx()
#line2 = ax0.plot(np.arange(len(training_record_FF[0][1])-199), training_record_FF[0][1][199:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

line3_ForwardRec= ax[1].plot(x, ForwardRec['amazon-beauty'][0], label='ForwardRec', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(6))
line3_LightGCN = ax[1].plot(x, LightGCN['amazon-beauty'][0], label='LightGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(1))
line3_LGCN = ax[1].plot(x, LGCN['amazon-beauty'][0], label='LGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(4))
line3_DHCF = ax[1].plot(x, DHCF['amazon-beauty'][0], label='DHCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(3))
line3_SGL= ax[1].plot(x, SGL['amazon-beauty'][0], label='SGL', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(2))
line3_MixGCF = ax[1].plot(x, MixGCF['amazon-beauty'][0], label='MixGCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(7))
#ax1 = ax[1].twinx()
#line4 = ax1.plot(np.arange(len(training_record_FF[1][1])-87), training_record_FF[1][1][87:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

line5_ForwardRec= ax[2].plot(x, ForwardRec['douban-book'][0], label='ForwardRec', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(6))
line5_LightGCN = ax[2].plot(x, LightGCN['douban-book'][0], label='LightGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(1))
line5_LGCN = ax[2].plot(x, LGCN['douban-book'][0], label='LGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(4))
line5_DHCF = ax[2].plot(x, DHCF['douban-book'][0], label='DHCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(3))
line5_SGL = ax[2].plot(x, SGL['douban-book'][0], label='SGL', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(2))
line5_MixGCF = ax[2].plot(x, MixGCF['douban-book'][0], label='MixGCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(7))
#ax2 = ax[2].twinx()
#line6 = ax2.plot(np.arange(len(training_record_FF[2][1])-88), training_record_FF[2][1][88:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

line7_ForwardRec= ax[3].plot(x, ForwardRec['yelp2018'][0], label='ForwardRec', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(6))
line7_LightGCN = ax[3].plot(x, LightGCN['yelp2018'][0], label='LightGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(1))
line7_LGCN = ax[3].plot(x, LGCN['yelp2018'][0], label='LGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(4))
line7_DHCF = ax[3].plot(x, DHCF['yelp2018'][0], label='DHCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(3))
line7_SGL = ax[3].plot(x, SGL['yelp2018'][0], label='SGL', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(2))
line7_MixGCF = ax[3].plot(x, MixGCF['yelp2018'][0], label='MixGCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(7))
#ax3 = ax[3].twinx()
#line8 = ax3.plot(np.arange(len(training_record_FF[3][1])-75), training_record_FF[3][1][75:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

line9_ForwardRec= ax[4].plot(x, ForwardRec['gowalla'][0], label='ForwardRec', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(6))
line9_LightGCN = ax[4].plot(x, LightGCN['gowalla'][0], label='LightGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(1))
line9_LGCN = ax[4].plot(x, LGCN['gowalla'][0], label='LGCN', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(4))
line9_DHCF = ax[4].plot(x, DHCF['gowalla'][0], label='DHCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(3))
line9_SGL = ax[4].plot(x, SGL['gowalla'][0], label='SGL', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(2))
line9_MixGCF = ax[4].plot(x, MixGCF['gowalla'][0], label='MixGCF', linestyle='-.', linewidth=2, marker='8', markersize=10, color=plt.cm.Set1(7))
#ax4 = ax[4].twinx()
#lineA = ax4.plot(np.arange(len(training_record_FF[4][1])-87), training_record_FF[4][1][87:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel('Layer')
ax[0].set_ylabel('Hit Rate@20', fontsize=45)
#ax3.set_ylabel('Loss', fontsize=30)
ax[0].set_title('LastFM', fontsize=40)
ax[1].set_title('Amazon-Beauty', fontsize=40)
ax[2].set_title('Douban-Book', fontsize=40)
ax[3].set_title('Yelp2018', fontsize=40)
ax[4].set_title('Gowalla', fontsize=40)
# #ax.set_xticks(x)
# #ax.set_xticklabels(labels)
ax[0].grid(linestyle='--',alpha=0.5)
ax[1].grid(linestyle='--',alpha=0.5)
ax[2].grid(linestyle='--',alpha=0.5)
ax[3].grid(linestyle='--',alpha=0.5)
ax[4].grid(linestyle='--',alpha=0.5)

ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[4].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax[0].set_xlabel('Layer', fontsize=35)
ax[1].set_xlabel('Layer', fontsize=35)
ax[2].set_xlabel('Layer', fontsize=35)
ax[3].set_xlabel('Layer', fontsize=35)
ax[4].set_xlabel('Layer', fontsize=35)


# handles_h, labels_h = ax[0].get_legend_handles_labels()
# #handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = ax[0].legend(handles_h, labels_h, loc='upper center', ncol=1, fontsize=16)
# leg = ax[1].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[2].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[3].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[4].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# for obj in leg.legendHandles:
#     obj.set_linewidth(1.0)

handles_h, labels_h = ax[0].get_legend_handles_labels()
# handles_d, labels_d = ax1_1.get_legend_handles_labels()
fig.legend(handles_h, labels_h,loc='upper center', ncol=6, fontsize=30)
plt.subplot_tool()
#plt.savefig('a.pdf', format='pdf')
plt.show()

# =============================================================================
# loss with compared methods
# line11 = ax[0].plot(np.arange(min(len(training_record_FF[0][1])-199,200)), training_record_FF[0][1][199:199+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line12 = ax[0].plot(np.arange(min(len(training_record_SGL[0][1]),200)), training_record_SGL[0][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line13 = ax[0].plot(np.arange(min(len(training_record_LightGCN[0][1]),200)), training_record_LightGCN[0][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line14 = ax[0].plot(np.arange(min(len(training_record_LTGNN[0][1]),200)), training_record_LTGNN[0][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line15 = ax[0].plot(np.arange(min(len(training_record_MixGCF[0][1]),200)), training_record_MixGCF[0][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))
# 
# line21 = ax[1].plot(np.arange(min(len(training_record_FF[1][1])-87,200)), training_record_FF[1][1][87:87+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line22 = ax[1].plot(np.arange(min(len(training_record_SGL[1][1]),200)), training_record_SGL[1][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line23 = ax[1].plot(np.arange(min(len(training_record_LightGCN[1][1]),200)), training_record_LightGCN[1][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line24 = ax[1].plot(np.arange(min(len(training_record_LTGNN[1][1]),200)), training_record_LTGNN[1][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line25 = ax[1].plot(np.arange(min(len(training_record_MixGCF[1][1]),200)), training_record_MixGCF[1][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))
# 
# line31 = ax[2].plot(np.arange(min(len(training_record_FF[2][1])-88,200)), training_record_FF[2][1][88:88+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line32 = ax[2].plot(np.arange(min(len(training_record_SGL[2][1]),200)), training_record_SGL[2][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line33 = ax[2].plot(np.arange(min(len(training_record_LightGCN[2][1]),200)), training_record_LightGCN[2][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line34 = ax[2].plot(np.arange(min(len(training_record_LTGNN[2][1]),200)), training_record_LTGNN[2][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line35 = ax[2].plot(np.arange(min(len(training_record_MixGCF[2][1]),200)), training_record_MixGCF[2][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))
# 
# line41 = ax[3].plot(np.arange(min(len(training_record_FF[3][1])-75,200)), training_record_FF[3][1][75:200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line42 = ax[3].plot(np.arange(min(len(training_record_SGL[3][1]),200)), training_record_SGL[3][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line43 = ax[3].plot(np.arange(min(len(training_record_LightGCN[3][1]),200)), training_record_LightGCN[3][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line44 = ax[3].plot(np.arange(min(len(training_record_LTGNN[3][1]),200)), training_record_LTGNN[3][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line45 = ax[3].plot(np.arange(min(len(training_record_MixGCF[3][1]),200)), training_record_MixGCF[3][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))
# 
# 
# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel('Layer')
# ax[0].set_ylabel('Hit Rate', fontsize=30)
# # ax3.set_ylabel('Loss', fontsize=30)
# ax[0].set_title('LastFM', fontsize=30)
# ax[1].set_title('Amazon-Beauty', fontsize=30)
# ax[2].set_title('Douban-Book', fontsize=30)
# ax[3].set_title('Yelp2018', fontsize=30)
# # #ax.set_xticks(x)
# # #ax.set_xticklabels(labels)
# ax[0].grid(linestyle='--',alpha=0.5)
# ax[1].grid(linestyle='--',alpha=0.5)
# ax[2].grid(linestyle='--',alpha=0.5)
# ax[3].grid(linestyle='--',alpha=0.5)
# 
# 
# handles_h, labels_h = ax[0].get_legend_handles_labels()
# #handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = fig.legend(handles_h, labels_h, loc='upper center', ncol=6, fontsize=20)
# for obj in leg.legendHandles:
#     obj.set_linewidth(3.0)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()
# 
# 
# =============================================================================

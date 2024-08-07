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
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

training_record_FF = []
for dataset in ['lastfm','beauty','douban_new','yelp2018']:
    with open ('time_record/training_record_'+dataset, 'rb') as fp:
        tmp = pickle.load(fp)
    hr = []
    loss = []
    for elem in tmp[0]:
        hr.append(float(elem[0].split(':')[1]))
    loss = tmp[1]
    training_record_FF.append([hr,loss])
    
training_record_SGL = []
for dataset in ['lastfm','beauty','douban','yelp2018']:
    with open ('time_record/training_record_'+dataset+'_SGL_rec', 'rb') as fp:
        tmp = pickle.load(fp)
    hr = []
    loss = []
    for elem in tmp[0]:
        hr.append(float(elem[0].split(':')[1]))
    loss = tmp[1]
    training_record_SGL.append([hr,loss])

training_record_LightGCN = []
for dataset in ['lastfm','beauty','douban','yelp2018']:
    with open ('time_record/training_record_'+dataset+'_LightGCN', 'rb') as fp:
        tmp = pickle.load(fp)
    hr = []
    loss = []
    for elem in tmp[0]:
        hr.append(float(elem[0].split(':')[1]))
    loss = tmp[1]
    training_record_LightGCN.append([hr,loss])
    
training_record_LTGNN = []
for dataset in ['lastfm','beauty','douban','yelp2018']:
    with open ('time_record/training_record_'+dataset+'_LTGNN', 'rb') as fp:
        tmp = pickle.load(fp)
    hr = []
    loss = []
    for elem in tmp[0]:
        hr.append(elem['hit'][0])
    for elem in tmp[1]:
        loss.append(float(elem[4:9]))
    training_record_LTGNN.append([hr,loss])

training_record_MixGCF = []
for dataset in ['lastfm','beauty','douban','yelp2018']:
    with open ('time_record/training_record_'+dataset+'_MixGCF', 'rb') as fp:
        tmp = pickle.load(fp)
    hr = []
    loss = []
    for elem in tmp[0]:
        hr.append(float(elem[0].split(':')[1]))
    loss = tmp[1]
    training_record_MixGCF.append([hr,loss])

#-------------------------------------------------------------------------------------------------------------------
#Layer 1
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
plt.subplots_adjust(top=0.87,
bottom=0.03,
left=0.085,
right=0.92,
hspace=0.25,
wspace=0.33)
line1 = ax[0,0].plot(np.arange(len(training_record_FF[0][0][:199])), training_record_FF[0][0][:199], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
ax0 = ax[0,0].twinx()
line2 = ax0.plot(np.arange(len(training_record_FF[0][1][:199])), training_record_FF[0][1][:199], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
line3 = ax[0,1].plot(np.arange(len(training_record_FF[1][0][:87])), training_record_FF[1][0][:87], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
ax1 = ax[0,1].twinx()
line4 = ax1.plot(np.arange(len(training_record_FF[1][1][:87])), training_record_FF[1][1][:87], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

line5 = ax[1,0].plot(np.arange(len(training_record_FF[2][0][:87])), training_record_FF[2][0][:87], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
ax2 = ax[1,0].twinx()
line6 = ax2.plot(np.arange(len(training_record_FF[2][1][:87])), training_record_FF[2][1][:87], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))
line7 = ax[1,1].plot(np.arange(len(training_record_FF[3][0][:75])), training_record_FF[3][0][:75], label='Hit Rate', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(4))
ax3 = ax[1,1].twinx()
line8 = ax3.plot(np.arange(len(training_record_FF[3][1][:75])), training_record_FF[3][1][:75], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))


# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_xlabel('Layer')
ax[0,0].set_ylabel('Hit Rate', fontsize=30)
ax[1,0].set_ylabel('Hit Rate', fontsize=30)
ax1.set_ylabel('Loss', fontsize=30)
ax3.set_ylabel('Loss', fontsize=30)
ax[0,0].set_title('LastFM', fontsize=30)
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[0,1].set_title('Amazon-Beauty', fontsize=30)
ax[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1,0].set_title('Douban-Book', fontsize=30)
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax[1,1].set_title('Yelp2018', fontsize=30)
ax[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

#ax.set_xticks(x)
#ax.set_xticklabels(labels)
ax[0,0].grid(linestyle='--',alpha=0.5)
ax[0,1].grid(linestyle='--',alpha=0.5)
ax[1,0].grid(linestyle='--',alpha=0.5)
ax[1,1].grid(linestyle='--',alpha=0.5)


handles_h, labels_h = ax[0,0].get_legend_handles_labels()
handles_l, labels_l = ax0.get_legend_handles_labels()
leg = fig.legend(handles_h+handles_l, labels_h+labels_l, loc='upper center', ncol=6, fontsize=20)
for obj in leg.legendHandles:
    obj.set_linewidth(6.0)
plt.subplot_tool()
#plt.savefig('a.pdf', format='pdf')
plt.show()

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

# with compared methods 
# fig, ax = plt.subplots(1, 4, figsize=(25, 5))
# plt.subplots_adjust(top=0.877,
# bottom=0.107,
# left=0.046,
# right=0.99,
# hspace=0.195,
# wspace=0.16)

# line1 = ax[0].plot(np.arange(len(training_record_FF[0][0])-199), training_record_FF[0][0][199:], label='ForwardRec', linestyle='-.', linewidth=2, color=plt.cm.Set1(1))
# line1_SGL = ax[0].plot(np.arange(len(training_record_FF[0][0])-199), training_record_SGL[0][0][:len(training_record_FF[0][0])-199], label='SGL', linestyle='-.', linewidth=2, color=plt.cm.Set1(4))
# line1_LightGCN = ax[0].plot(np.arange(len(training_record_FF[0][0])-199), training_record_LightGCN[0][0][:len(training_record_FF[0][0])-199], label='LightGCN', linestyle='-.', linewidth=2, color=plt.cm.Set1(3))
# line1_MixGCF = ax[0].plot(np.arange(len(training_record_FF[0][0])-199), training_record_MixGCF[0][0][:len(training_record_FF[0][0])-199], label='MixGCF', linestyle='-.', linewidth=2, color=plt.cm.Set1(2))
# line1_LTGNN = ax[0].plot(np.arange(len(training_record_FF[0][0])-199), training_record_LTGNN[0][0][:len(training_record_FF[0][0])-199], label='LTGNN', linestyle='-.', linewidth=2, color=plt.cm.Set1(7))
# #ax0 = ax[0].twinx()
# #line2 = ax0.plot(np.arange(len(training_record_FF[0][1])-199), training_record_FF[0][1][199:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# line3 = ax[1].plot(np.arange(len(training_record_FF[1][0])-87), training_record_FF[1][0][87:], label='ForwardRec', linestyle='-.', linewidth=2,  color=plt.cm.Set1(1))
# line3_SGL = ax[1].plot(np.arange(len(training_record_FF[1][0])-87), training_record_SGL[1][0][:len(training_record_FF[1][0])-87], label='SGL', linestyle='-.', linewidth=2, color=plt.cm.Set1(4))
# line3_LightGCN = ax[1].plot(np.arange(len(training_record_FF[1][0])-87), training_record_LightGCN[1][0][:len(training_record_FF[1][0])-87], label='LightGCN', linestyle='-.', linewidth=2, color=plt.cm.Set1(3))
# line3_MixGCF= ax[1].plot(np.arange(len(training_record_FF[1][0])-87), training_record_MixGCF[1][0][:len(training_record_FF[1][0])-87], label='MixGCF', linestyle='-.', linewidth=2, color=plt.cm.Set1(2))
# line3_LTGNN = ax[1].plot(np.arange(len(training_record_FF[1][0])-87), training_record_LTGNN[1][0][:len(training_record_FF[1][0])-87], label='LTGNN', linestyle='-.', linewidth=2, color=plt.cm.Set1(7))
# #ax1 = ax[1].twinx()
# #line4 = ax1.plot(np.arange(len(training_record_FF[1][1])-87), training_record_FF[1][1][87:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# line5 = ax[2].plot(np.arange(len(training_record_FF[2][0])-88), training_record_FF[2][0][88:], label='ForwardRec', linestyle='-.', linewidth=2, color=plt.cm.Set1(1))
# line5_SGL = ax[2].plot(np.arange(len(training_record_FF[2][0])-88), training_record_SGL[2][0][:len(training_record_FF[2][0])-88], label='SGL', linestyle='-.', linewidth=2, color=plt.cm.Set1(4))
# line5_LightGCN = ax[2].plot(np.arange(len(training_record_FF[2][0])-88), training_record_LightGCN[2][0][:len(training_record_FF[2][0])-88], label='LightGCN', linestyle='-.', linewidth=2, color=plt.cm.Set1(3))
# line5_MixGCF = ax[2].plot(np.arange(len(training_record_FF[2][0])-88), training_record_MixGCF[2][0][:len(training_record_FF[2][0])-88], label='MixGCF', linestyle='-.', linewidth=2, color=plt.cm.Set1(2))
# line5_LTGNN = ax[2].plot(np.arange(len(training_record_FF[2][0])-88), training_record_LTGNN[2][0][:len(training_record_FF[2][0])-88], label='LTGNN', linestyle='-.', linewidth=2, color=plt.cm.Set1(7))
# #ax2 = ax[2].twinx()
# #line6 = ax2.plot(np.arange(len(training_record_FF[2][1])-88), training_record_FF[2][1][88:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# line7 = ax[3].plot(np.arange(len(training_record_FF[3][0])-75), training_record_FF[3][0][75:], label='ForwardRec', linestyle='-.', linewidth=2, color=plt.cm.Set1(1))
# line7_SGL = ax[3].plot(np.arange(len(training_record_FF[3][0])-75), training_record_SGL[3][0][:len(training_record_FF[3][0])-75], label='SGL', linestyle='-.', linewidth=2, color=plt.cm.Set1(4))
# line7_LighGCN = ax[3].plot(np.arange(len(training_record_FF[3][0])-75), training_record_LightGCN[3][0][:len(training_record_FF[3][0])-75], label='LightGCN', linestyle='-.', linewidth=2, color=plt.cm.Set1(3))
# line7_MixGCF = ax[3].plot(np.arange(len(training_record_FF[3][0])-75), training_record_MixGCF[3][0][:len(training_record_FF[3][0])-75], label='MixGCF', linestyle='-.', linewidth=2, color=plt.cm.Set1(2))
# line7_LTGNN = ax[3].plot(np.arange(len(training_record_FF[3][0])-75), training_record_LTGNN[3][0][:len(training_record_FF[3][0])-75], label='LTGNN', linestyle='-.', linewidth=2, color=plt.cm.Set1(7))
# #ax3 = ax[3].twinx()
# #line8 = ax3.plot(np.arange(len(training_record_FF[3][1])-75), training_record_FF[3][1][75:], label='Loss', marker='8', linewidth=1, markersize=1, color=plt.cm.Set3(3))

# # Add some text for labels, title and custom x-axis tick labels, etc.
# #ax.set_xlabel('Layer')
# ax[0].set_ylabel('Hit Rate', fontsize=30)
# #ax3.set_ylabel('Loss', fontsize=30)
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


# handles_h, labels_h = ax[0].get_legend_handles_labels()
# #handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = ax[0].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[1].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[2].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# leg = ax[3].legend(handles_h, labels_h, loc='lower right', ncol=1, fontsize=16)
# for obj in leg.legendHandles:
#     obj.set_linewidth(2.0)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()

# loss with compared methods
# line11 = ax[0].plot(np.arange(min(len(training_record_FF[0][1])-199,200)), training_record_FF[0][1][199:199+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line12 = ax[0].plot(np.arange(min(len(training_record_SGL[0][1]),200)), training_record_SGL[0][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line13 = ax[0].plot(np.arange(min(len(training_record_LightGCN[0][1]),200)), training_record_LightGCN[0][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line14 = ax[0].plot(np.arange(min(len(training_record_LTGNN[0][1]),200)), training_record_LTGNN[0][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line15 = ax[0].plot(np.arange(min(len(training_record_MixGCF[0][1]),200)), training_record_MixGCF[0][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))

# line21 = ax[1].plot(np.arange(min(len(training_record_FF[1][1])-87,200)), training_record_FF[1][1][87:87+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line22 = ax[1].plot(np.arange(min(len(training_record_SGL[1][1]),200)), training_record_SGL[1][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line23 = ax[1].plot(np.arange(min(len(training_record_LightGCN[1][1]),200)), training_record_LightGCN[1][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line24 = ax[1].plot(np.arange(min(len(training_record_LTGNN[1][1]),200)), training_record_LTGNN[1][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line25 = ax[1].plot(np.arange(min(len(training_record_MixGCF[1][1]),200)), training_record_MixGCF[1][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))

# line31 = ax[2].plot(np.arange(min(len(training_record_FF[2][1])-88,200)), training_record_FF[2][1][88:88+200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line32 = ax[2].plot(np.arange(min(len(training_record_SGL[2][1]),200)), training_record_SGL[2][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line33 = ax[2].plot(np.arange(min(len(training_record_LightGCN[2][1]),200)), training_record_LightGCN[2][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line34 = ax[2].plot(np.arange(min(len(training_record_LTGNN[2][1]),200)), training_record_LTGNN[2][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line35 = ax[2].plot(np.arange(min(len(training_record_MixGCF[2][1]),200)), training_record_MixGCF[2][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))

# line41 = ax[3].plot(np.arange(min(len(training_record_FF[3][1])-75,200)), training_record_FF[3][1][75:200], label='ForwardRec', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(4))
# line42 = ax[3].plot(np.arange(min(len(training_record_SGL[3][1]),200)), training_record_SGL[3][1][:200], label='SGL', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(2))
# line43 = ax[3].plot(np.arange(min(len(training_record_LightGCN[3][1]),200)), training_record_LightGCN[3][1][:200], label='LightGCN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(3))
# line44 = ax[3].plot(np.arange(min(len(training_record_LTGNN[3][1]),200)), training_record_LTGNN[3][1][:200], label='LTGNN', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(5))
# line45 = ax[3].plot(np.arange(min(len(training_record_MixGCF[3][1]),200)), training_record_MixGCF[3][1][:200], label='MixGCF', marker='8',linewidth=1,markersize=1,color=plt.cm.Set3(6))


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


# handles_h, labels_h = ax[0].get_legend_handles_labels()
# #handles_l, labels_l = ax0.get_legend_handles_labels()
# leg = fig.legend(handles_h, labels_h, loc='upper center', ncol=6, fontsize=20)
# for obj in leg.legendHandles:
#     obj.set_linewidth(3.0)
# plt.subplot_tool()
# #plt.savefig('a.pdf', format='pdf')
# plt.show()


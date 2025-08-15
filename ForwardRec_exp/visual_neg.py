# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 18:54:49 2025

@author: user
"""

import os
import pickle
from numpy import linalg
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 16}

plt.rc('font', **font)

# canvas initialization
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
plt.subplots_adjust(top=0.882,
bottom=0.162,
left=0.061,
right=0.985,
hspace=0.18,
wspace=0.175)

#Load file
emb_dict = {}
PATH = 'neg_visual/'
for name in ['RNS','MixGCF','AHNS','ForwardRec']:
    with open(PATH+name,'rb') as fp:
        emb_dict[name.split('.')[0]] = pickle.load(fp)
        
#Calculate TSNE
TSNE_dict = {}
for i, name in enumerate(list(emb_dict.keys())):
    tsne = TSNE(n_components=2,random_state=0, perplexity=emb_dict[name][3].shape[0]-1)
    tsne_res_pos = tsne.fit_transform(emb_dict[name][2])
    tsne_res_neg = tsne.fit_transform(emb_dict[name][3])
    # n2_user = linalg.norm(tsne_res_user, axis=1)
    # for i in range(len(n2_user)):
    #     tsne_res_user[i,0]/=n2_user[i]
    #     tsne_res_user[i,1]/=n2_user[i]
    # x_user = tsne_res_user[:,0]
    # y_user = tsne_res_user[:,1]
    # angle_user = np.arctan2(y_user,x_user)
    # TSNE_dict[name] = [x_user, y_user, angle_user]
    diff = tsne_res_neg-tsne_res_pos
    d = np.sum(np.sqrt(np.sum(np.power(diff,2), axis=1)))
    
    data = pd.DataFrame({'x': diff[:,0], 'y': diff[:,1], 'label': emb_dict[name][1]})
    
    ax[i].set_title(r'%s, $d$=%.2f'%(name, d), fontsize=35)
    if 'Positive' in emb_dict[name][1]:
        sns.scatterplot(ax=ax[i], data=data, x='x', y='y', alpha=0.6, hue='label', hue_order=['Negative', 'Positive'], palette=['forestgreen','lightcoral'], s=500)
    else:
        sns.scatterplot(ax=ax[i], data=data, x='x', y='y', alpha=0.6, hue='label', palette=['forestgreen','lightcoral'], s=500)
    ax[i].set_xlabel("Dimension 1", fontsize=45)
    if i==0:
        ax[i].set_ylabel("Dimension 2", fontsize=45)
    else:
        ax[i].set_ylabel("", fontsize=45)
    ax[i].grid(linestyle='--',alpha=0.5)
    sns.move_legend(ax[i], "upper right", fontsize=25, markerscale=3)
    ax[i].legend_.set_title(None)
    
    
    
    
    
    
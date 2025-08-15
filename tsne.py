# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:41:44 2024

@author: user
"""

import os
import pickle
from numpy import linalg
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

with open('lightgcn_emb', 'rb') as fp:
    all_emb = pickle.load(fp)
all_emb = np.concatenate([all_emb[0], all_emb[1]], axis=0)

with open('binary_adj', 'rb') as fp:
    binary_adj = pickle.load(fp).A
user_num, item_num = binary_adj.shape
user_degree = binary_adj.sum(axis=1)
item_degree = binary_adj.sum(axis=0)

tsne = TSNE(n_components=2,random_state=0)
tsne_emb = tsne.fit_transform(all_emb)

user_colors = user_degree / max(user_degree)
item_colors = item_degree / max(item_degree)

fig, ax = plt.subplots(1, 2,figsize=(10,5))

user_scatter = ax[0].scatter(tsne_emb[:user_num,0], tsne_emb[:user_num,1], c=user_colors, cmap='Blues', s=4, label='User Colors')
user_legend = ax[0].legend(*user_scatter.legend_elements(), title='User Colors')
item_scatter = ax[1].scatter(tsne_emb[user_num:,0], tsne_emb[user_num:,1], c=item_colors, cmap='Oranges', s=4, label='Item Colors')
item_legend = ax[1].legend(*item_scatter.legend_elements(), title='Item Colors')

ax[0].gca().add_artist(user_legend)
ax[0].gca().add_artist(item_legend)

plt.show()


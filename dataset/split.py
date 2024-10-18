# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:13:28 2024

@author: user
"""

import pandas as pd

data = pd.read_csv('ratings.dat', sep='\t', header=None)
data = data[[0,1,2]]

train = data.sample(frac=0.8).sort_values(by=[0,1])
test = data.drop(train.index)

train.to_csv('train.txt', sep=' ', index=None, header=None)
test.to_csv('test.txt', sep=' ', index=None, header=None)
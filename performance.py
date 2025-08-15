# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:39:08 2024

@author: user
"""

import re
import numpy as np

with open('performance.txt','r') as fp:
    l = fp.readlines()

all_data = []

for elem in l:
    data = []
    elem = re.split('{|}', elem)
    try:
        res = re.split(':|,', elem[1])
    except:
        continue
    for index in [1,3,5,7]:
        try:
            data.append(float(res[index].strip()))
        except:
            break
    all_data.append(data)
    
all_data = np.array(all_data)
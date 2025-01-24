# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:30:43 2024

@author: user
"""

import pickle
import scipy as sp

with open('gowalla', 'rb') as fp:
    norm_adj = pickle.load(fp)
    
e, v = sp.sparse.linalg.eigs(norm_adj, k=100, which='SM', tol=1)

with open('eigen', 'wb') as fp:
    pickle.dump([e,v], fp)
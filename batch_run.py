# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:08 2024

@author: user
"""

import os

if __name__ == '__main__':
    for n_layer in [1,2,3,4]:
        with open('performance.txt','a') as fp:
            fp.write('When layer=%d:'%(n_layer))
        os.system("python update_conf.py -n_layer %d"%(n_layer))
        os.system("python main.py")
            
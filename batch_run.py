# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:08 2024

@author: user
"""

import os

if __name__ == '__main__':
    for lamda in [0.05,0.1,0.5,1.0]:
        for eps in [0.0,0.1,0.2,0.3,0.4,0.5]:
            with open('performance.txt','a') as fp:
                fp.write('When lamda=%f,eps=%f:'%(lamda,eps))
            os.system("python update_conf.py -lamda %f -eps %f"%(lamda,eps))
            os.system("python main.py")
            
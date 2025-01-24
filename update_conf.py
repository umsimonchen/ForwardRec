# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:44 2024

@author: user
"""

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n_layer", type=int)
args = parser.parse_args()

with open('conf/LGCN.conf', 'r') as fp:
    lines = fp.readlines()

lines[10] = "LGCN=-n_layer %d -frequency 100"%(args.n_layer)+"\n"

with open('conf/LGCN.conf', 'w') as fp:
    fp.writelines(lines)
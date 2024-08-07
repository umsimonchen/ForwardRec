# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:44 2024

@author: user
"""

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-lamda", type=float)
parser.add_argument("-eps", type=float)
args = parser.parse_args()

with open('conf/SGL.conf', 'r') as fp:
    lines = fp.readlines()

lines[10] = "SGL=-n_layer 4 -temp 0.2 -augtype 1 -lambda %f -droprate %f"%(args.lamda, args.eps)+"\n"

with open('conf/SGL.conf', 'w') as fp:
    fp.writelines(lines)
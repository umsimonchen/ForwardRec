# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:18:44 2024

@author: user
"""

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str)
parser.add_argument("-alpha", type=float)
args = parser.parse_args()

with open('conf/DINS.conf', 'r') as fp:
    lines = fp.readlines()

lines[0] = "training.set=./dataset/%s/train.txt"%(args.dataset)+"\n"
lines[1] = "test.set=./dataset/%s/test.txt"%(args.dataset)+"\n"
lines[10] = "DINS=-n_layer 2 -alpha %f -candidate 64"%(args.alpha)+"\n"

with open('conf/DINS.conf', 'w') as fp:
    fp.writelines(lines)
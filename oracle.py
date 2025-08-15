# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:56:26 2024

@author: user
"""

all_test_records = {}
with open('dataset/amazon-beauty/test.txt', 'r') as fp:
    all_test = fp.readlines()

for line in all_test:
    line = line.strip().split('\t')
    try:
        all_test_records[line[0]].append(line[1])
    except:
        all_test_records[line[0]] = [line[1]]

K = 20
precision = 0.0
for user in all_test_records:
    precision += len(all_test_records[user])/K
print("Oracle of Precision: ", precision/len(all_test_records))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:31:18 2019

@author: tianyu
"""

import numpy as np
mat = np.zeros((124,16281))
mat[0,:] = 1
label = np.zeros((16281,1))

f=open("a9a_test.txt", "r")
f1 =f.readlines()
print(len(f1))
for i in range(0,len(f1)):
    print(i)
    x = f1[i]
    l = len(x)
    jind = []
    bind = []
    ind = []
    for j in range(0,l):
        if x[j] == ':':
            jind.append(j)    
        if x[j] == ' ':
            bind.append(j)
    for k in range(0,len(jind)):
        ind.append(int(x[bind[k]+1:jind[k]],10))
    for p in range(0,len(ind)):
        mat[ind[p],i] = 1
    if x[0] == '+':
        label[i,0] = 1
    else:
        label[i,0] = -1

print(label[0:10,0])

np.save('test_data.npy',mat)
np.save('test_label.npy',label)    


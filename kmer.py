#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:18:49 2018

"""

from itertools import *
import numpy as np
import matplotlib.pyplot as plt

import time

query='GCF_002973605.1_ASM297360v1_genomic.fna'


def k_mer(fileName,k):
    start = time.time()
    words=list(product('ATGC', repeat = k))
    with open(fileName,'r') as genomeFile:
        lines = genomeFile.readlines()
    genom = ''
    for line in lines:
        if not '>' in line: 
            genom += line
    genom = genom.replace('\n','')
    genomLength = len(genom)
    result = {}
    for wo in words:
        w = ''
        for i in range(k):
            w += wo[i]
        result[w] = 0
    for i in range(genomLength-k+1):
        result[genom[i:i+k]]+=1
    end = time.time()
    return (result, end-start)
    
        
result = k_mer(query,10)

#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.bar(list(result[0].keys()), result[0].values(), color='g')
#plt.show()

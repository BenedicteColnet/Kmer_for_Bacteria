#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:18:49 2018

"""

from itertools import *
import numpy as np
import matplotlib.pyplot as plt
import time

def read_genome(fileName):
    
    """
    ENTRY adress of fasta file with genome (str)
    
    FUNCTION read the file and extract genome
    
    EXIT query only (str)
    
    """
    with open(fileName,'r') as genomeFile:
        lines = genomeFile.readlines()
    genom_sequence = ''
    for line in lines:
        if not '>' in line: 
            genom_sequence += line
    genom_sequence = genom_sequence.replace('\n','')
    
    return genom_sequence




def k_mer_naive(genom,k):
            
    """
    ENTRY genom sequence (str) and size of kmer asked (int)
    
    FUNCTION read the genom and count the number of kmer
    
    EXIT dictionnary ({kmer:number})
    
    """    
    genomLength = len(genom)
    result = {}
    words=list_of_possible_kmer('ATGC',k)
    for wo in words:
        w = ''
        for i in range(k):
            w += wo[i]
        result[w] = 0
    for i in range(genomLength-k+1):
        result[genom[i:i+k]]+=1

    return result



def list_of_possible_kmer(letters,k):
    
    """
    ENTRY letters possible for e.g 'ATGC' (str) and size of kmer (int)
    
    FUNCTION create a list of all possible words of size k
    
    EXIT list of possible words of size k (list)
    
    """  
    words=list(product(letters, repeat = k))
    return words



query='GCF_002973605.1_ASM297360v1_genomic.fna'    
start = time.time()       
result = k_mer_naive(read_genome(query),3)
end = time.time()

plt.figure()
plt.bar(result.keys(), result.values(), color='g')
plt.plot()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:18:49 2018

"""

from itertools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import time

def read_genome(fileName):
    
    """
    ENTRY adress of fasta file with genome (str)
    
    FUNCTION read the file and extract the whole genome
    
    EXIT query only (str)
    
    """
    with open(fileName,'r') as genomeFile:
        lines = genomeFile.readlines()
    tmp = lines[0].split(' ')
    genomeName = tmp[1] + '_' + tmp[2]
    genom_sequence = ''
    for line in lines:
        if not '>' in line:
            for c in line:
                if c not in set('ACGT'):
                    line  = line.replace(c,'')
            genom_sequence += line
    genom_sequence = genom_sequence.replace('\n','')
    return (genomeName,genom_sequence)

def k_mer_naive_with_dict_initialization(genome,k):
            
    """
    ENTRY genom sequence (str) and size of kmer asked (int)
    
    FUNCTION read the genom and count the number of kmer with dictionary init
    
    EXIT dictionnary ({kmer:number})
    
    """    
    genomLength = len(genome)
    result = {}
    words=list_of_possible_kmer('ACGT',k)
    for wo in words:
        w = ''
        for i in range(k):
            w += wo[i]
        result[w] = 0
    for i in range(genomLength-k+1):
        result[genome[i:i+k]]+=1

    return result


def k_mer_naive_without_dict_initialization(genome,k):
            
    """
    ENTRY genom sequence (str) and size of kmer asked (int)
    
    FUNCTION read the genom and count the number of kmer without  dictionary init
    
    EXIT dictionnary ({kmer:number})
    
    """    
    genomLength = len(genome)
    result = {}
    for i in range(genomLength-k+1):
        kmer=genome[i:i+k]
        if kmer in result:
            result[kmer]+=1
        else:
            result[kmer]=1

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


def naive_time_xperiment():
    with_dic_initialization = []
    for k in range(1,14):
        print("Processing with_dict_init, k=",str(k))
        start = time.time()       
        result_with = k_mer_naive_with_dict_initialization(read_genome(query)[1],k)
        end = time.time()
        with_dic_initialization.append(end-start)

    without_dic_initialization = []
    for k in range(1,14):
        print("Processing without_dict_init, k=",str(k))
        start = time.time()       
        result_without = k_mer_naive_without_dict_initialization(read_genome(query)[1],k)
        end = time.time()
        without_dic_initialization.append(end-start)

    np.save('with_dic_initialization',with_dic_initialization)
    np.save('without_dic_initialization',without_dic_initialization)

    plt.plot(with_dic_initialization, color = 'r', label='With dictionary initialization')
    plt.plot(without_dic_initialization, color = 'b', label='Without dictionary initialization')
    plt.legend(loc='best')
    plt.show()

def dict_to_normalized_vector(dict):
    vec = np.array(list(dict.values()),dtype=object)
    return vec / sum(vec)

def homogeneity_matrix(genome,nDiv,k):
    seqSections = []
    seqSignature = []
    gLen = len(genome)
    l = int(np.round(gLen / nDiv))
    disMat = np.zeros((nDiv,nDiv))
    for i in range(nDiv):
        if((i+1)*l > gLen):
            break
        seqSections.append(genome[i*l:(i+1)*l])
    for sec in seqSections:
        sig = k_mer_naive_with_dict_initialization(sec,k)
        seqSignature.append(dict_to_normalized_vector(sig))
    
    for i,sigi in enumerate(seqSignature):
        for j,sigj in enumerate(seqSignature):
            disMat[i][j] = np.linalg.norm(sigi-sigj)
    return disMat

#comparing with global
def homogeneity_vector(genome,devLen,devShft,k):
    disVector = []
    gLen = len(genome)
    globalSig = dict_to_normalized_vector(k_mer_naive_with_dict_initialization(genome,k))
    for i in range(gLen-devLen+1):
        sec = genome[i*devShft:i*devShft+devLen]
        secSig = dict_to_normalized_vector(k_mer_naive_with_dict_initialization(sec,k))
        disVector.append(np.linalg.norm(globalSig-secSig))
    return disVector



"""
kList=[3,4,5,6]
nDivList = [50,100,200,500,1000]
devLenList = [1000,10000,20000,50000,100000]
devShiftPer = 0.7   
    
for k in kList:
    for nDiv in nDivList:
        distMat = homogeneity_matrix(read_genome(query)[1],nDiv,k)
        np.save('distMat' + str(nDiv) + str(k),distMat)
        fig, ax = plt.subplots()
        cax = ax.imshow(distMat, interpolation='nearest', cmap=cm.coolwarm)
        ax.set_title('Homogeneity Matrix, nDiv= ' + str(nDiv) + ', k= ' + str(k))
        cbar = fig.colorbar(cax, ticks=[0, 0, np.max(distMat)])
        plt.savefig('Homogeneity Matrix, nDiv= ' + str(nDiv) + ', k= ' + str(k))
    
    for devLen in devLenList:
        disVector = homogeneity_vector(read_genome(query)[1],devLen,int(np.round(devShiftPer*devLen)),k)
        np.save('distVector' + str(devLen) + str(k),disVector)
        fig, ax = plt.subplots()
        ax.plot(disVector)
        ax.set_title('Homogeneity Vector, devLen= ' + str(devLen) + ", k= " + str(k))
        ax.set_ylabel('Distance')
        ax.set_xlabel('Position')
        plt.savefig('Homogeneity Vector, devLen= ' + str(devLen) + ", k= " + str(k))
"""

def CreateProf(gType,gName,gSeq,kMax):
    prof = (gType,gName,)
    for k in range(1,kMax+1):
        signature = k_mer_naive_with_dict_initialization(gSeq,k)
        signature = dict_to_normalized_vector(signature)
        prof += (signature,)
    return prof

def CreateProfiles(gType,dbPath,kMax):
    profilesTest = []
    profilesTrain = []
    for filename in os.listdir(dbPath):
        gName,gSeq = read_genome(dbPath+"/"+filename)
        print("Processing ", gName)
        gLen = len(gSeq)
        np.random.seed(int(round(time.time())))

        #Test dataset (random positions)!
        steps = range(3,7)
        for s in steps:
            l = 5**s
            for i in range(10):
                winIndx = int(round(np.random.uniform(0,gLen-l)))    
                profilesTest.append(CreateProf(gType,gName,gSeq[winIndx:winIndx+l],kMax))
        
        #Train dataset
        l = 10000
        i=0
        while((i+1)*l < gLen):
            profilesTrain.append(CreateProf(gType,gName,gSeq[i*l:(i+1)*l],kMax))
            i+=1

    np.save("test_profiles", profilesTest)
    np.save("train_profiles_L"+str(l), profilesTrain)

CreateProfiles("bact","..//Database",5)
# -*- coding: utf-8 -*-
"""
Classification base on kmer vectors previously build
"""

import numpy as np
import pandas as pd
from itertools import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans


"""
OBJECT CONSIDERED
Tuple (type, name, 1-mer, 2-mer, 3-mer, and so on)

"""
# FUNCTIONS 


def list_of_possible_kmer(letters,k):
    
    """
    ENTRY letters possible for e.g 'ATGC' (str) and size of kmer (int)
    
    FUNCTION create a list of all possible words of size k
    
    OUT list of possible words of size k (list)
    
    """  
    words=list(product(letters, repeat = k))
    return words


def from_tuples_to_dataframe(data_as_tuples, list_of_k):
    """
    ENTRY list of tuples, list of Ks to concatenate
    OUT Dataframe
    """
    #Header
    kmer_vector=['Type','Name']
    for c in list_of_k:
        kmer_vector+=list_of_possible_kmer('ACGT',c)
        
    #Data    
    stock=[]
    for t in data_as_tuples:
        #item=[t[1]]
        item=[t[0],t[1]]
        for k in list_of_k:
            #item=item+t[k+1]
            for j in t[k+1]:
                item.append(j)
        stock.append(item)
    df = pd.DataFrame(stock, columns=kmer_vector)
    return df

def Principal_composant_analysis(data_as_df, number_of_component=2):
    """
    ENTRY Array with rows as items, and column as features (kmer concatenated)
    OUT number of principal components asked
    """
    pca = PCA(number_of_component)
    #X=data_as_df.loc[:,2:].asMatrix()
    X=data_as_df.values[:,2:]
    pca.fit(X)
    X_pca = pca.transform(X)
    return X_pca

def clustering_kmean(X,nb_of_clusters=3):
    """
    ENTRY X is the result of the PCA + number of cluster wanted
    OUT plot
    """
    km = KMeans(n_clusters=nb_of_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    return km

def from_Yasser_output_to_tuples(name_of_npy):
    data=np.load(name_of_npy)
    output=[]
    for item in data:
        prof = ('bact',)
        for i in item:
            if type(i)==str:
                prof+=(i,)
            else:
                prof += (list(i),)
        output.append(prof)
    return output

"""
Test area
"""


tuples_test=[('bact','a',[0.5,0,0.5,0]),('bact','b',[0,0.5,0,0.5]), ('bact','c',[0,0,0,1]), ('bact','d',[0,0.2,0,0.8]),
             ('bact','e',[0.9,0,0.1,0.]), ('bact','f',[0,0.1,0,0.9]), ('bact','g',[0,0.3,0.1,0.6]), ('bact','h',[0,0.2,0,0.8])
                    ]
#res=from_tuples_to_dataframe(tuples_test,[1])
#resPCA=Principal_composant_analysis(res)
#plt.scatter(resPCA[:, 0], resPCA[:, 1],marker='o',s=25, edgecolor='k')
#cluster_classifier=clustering_kmean(resPCA)
#plt.scatter(resPCA[:, 0], resPCA[:, 1], s=10, c=cluster_classifier.labels_)


data_tuples=from_Yasser_output_to_tuples("profiles.npy")
df=from_tuples_to_dataframe(data_tuples,[3])
resPCA=Principal_composant_analysis(df)
#plt.scatter(resPCA[:, 0], resPCA[:, 1],marker='o',s=25, edgecolor='k')
cluster_classifier=clustering_kmean(resPCA)
plt.scatter(resPCA[:, 0], resPCA[:, 1], s=10, c=cluster_classifier.labels_)
plt.savefig("one_two_3_mer_PCA_2.png")


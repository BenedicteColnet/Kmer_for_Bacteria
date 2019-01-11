# -*- coding: utf-8 -*-
"""
Classification base on kmer vectors previously build
"""


import pandas as pd
import numpy as np
"""
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
"""
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cluster
from sklearn.cluster import KMeans
from itertools import *
import matplotlib.pyplot as plt
import time
import pickle


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


def Tuples2DF(data_as_tuples, list_of_k):
    """
    ENTRY list of tuples, list of Ks to concatenate
    OUT Dataframe
    """
    #Header
    kmer_vector=['Type','Name','Length']
    for c in list_of_k:
        kmer_vector+=list_of_possible_kmer('ACGT',c)
        
    #Data    
    stock=[]
    for t in data_as_tuples:
        #item=[t[1]]
        item=[t[0],t[1],t[2]]
        for k in list_of_k:
            #item=item+t[k+1]
            for j in t[k+2]:
                item.append(j)
        stock.append(item)
    df = pd.DataFrame(stock, columns=kmer_vector)
    return df

def ApplyPCA(X, nComp=2):
    """
    ENTRY Array with rows as items, and column as features (kmer concatenated)
    OUT number of principal components asked
    """    
    pcaM = PCA(n_components=nComp)
    #X=data_as_df.values[:,2:]
    pcaM.fit(X)
    X_pca = pcaM.transform(X)
    return X_pca

def clustering_kmean(X,nb_of_clusters=3):
    """
    ENTRY X is the result of the PCA + number of cluster wanted
    OUT plot
    """
    km = KMeans(n_clusters=nb_of_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    return km

def PreProcessTuples(name_of_npy):
    """
    ENTRY Database name
    OUT data base in shape [('bact','name',[0-mer],[1-mer],[2-mer], etc...),(),()]
    """
    data=np.load(name_of_npy)
    output=[]
    for item in data:
        prof = ()
        for i in item:
            if type(i)==str:
                prof+=(i,)
            else:
                prof += (list(i),)
        output.append(prof)
    return output

def CrossValidation(model,X,Y,cross_val=10):
    startT = time.time()
    cvScores = cross_val_score(model, X, Y, cv=cross_val, scoring='accuracy')*100
    endT = time.time()
    return cvScores,endT-startT


def CrossValidationStratification(model,X,Y,cross_val=10):
    startT = time.time()
    skf = StratifiedKFold(n_splits=cross_val, random_state=7, shuffle=True)
    cvScores = []
    for train_index, test_index in skf.split(X,Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train,y_train)
        cvScores.append(model.score(X_test,y_test))
    endT = time.time()
    cvScores = np.array(cvScores)
    return cvScores,endT-startT


def AnalyzeNNMLP(X,y,pcaNC=False):
    if pcaNC != False:
        data = ApplyPCA(X,pcaNC)
    else:
        data = X
    modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-7,hidden_layer_sizes=(10, 5))
    #scoresMLP, totalDurMLP = CrossValidation(modelMLP,data,y,10)
    scoresMLP, totalDurMLP = CrossValidationStratification(modelMLP,data,y,10)
    print("MLP CV, Accuracy: %0.2f%% (+/- %0.2f%%), total time: %f s" 
          % (scoresMLP.mean()*100, scoresMLP.std()*100 * 2, totalDurMLP))

def AnalyzeNLSVM(X,y,pcaNC=False):
    if pcaNC != False:
        data = ApplyPCA(X,pcaNC)
    else:
        data = X
    for kern in ['linear', 'poly', 'rbf']:
        for cValue in [10e-9,0.1,10]:
            modelSVM = svm.SVC(gamma = 'auto', C=cValue, kernel=kern)
            #scoresSVM, totalDurSVM = CrossValidation(modelSVM,data,y,10)
            scoresSVM, totalDurSVM = CrossValidationStratification(modelSVM,data,y,10)
            print("SVM CV, Kernel: " + kern + ", C value: " + str(cValue) + ", Accuracy: %0.2f%% (+/- %0.2f%%), total time: %f s" 
                  % (scoresSVM.mean()*100, scoresSVM.std()*100 * 2, totalDurSVM))

def PreProcessKmerDataTrain(name_of_npy, list_of_k):
    #mapping=np.load("mapping.npy")
    tuples=PreProcessTuples(name_of_npy)
    df=Tuples2DF(tuples, list_of_k)
    df=df.sample(frac=1) #For suffle
    #df_numerized=df.replace({'Name': mapping})
    #Y=df_numerized["Name"]
    #X=df_numerized.drop(['Type', 'Name'], axis=1)
    Y = pd.factorize(df["Name"])[0]
    X=df.drop(['Type', 'Name', 'Length'], axis=1)    
    return X, Y

def PreProcessKmerDataTest(name_of_npy, list_of_k):
    groups = dict()
    tuples=PreProcessTuples(name_of_npy)
    df=Tuples2DF(tuples, list_of_k)
    df=df.sample(frac=1) #For suffle
    Y = pd.factorize(df["Name"])[0]
    for index, sample in df.iterrows():
        spcs = sample["Name"]
        if spcs not in groups.keys():
            groups[spcs] = {}
        l = sample["Length"]
        if l not in groups[spcs].keys():
            groups[spcs][l] = ([],[])
        groups[spcs][l][1].append(Y[index])
        groups[spcs][l][0].append(sample.drop(['Type', 'Name', 'Length']).values)   
    return groups

def SaveModel(folder,model,file):
    fHandler = open(folder+'/'+file,'wb')
    pickle.dump(model, fHandler)
    fHandler.close()

def LoadModel(fileName):
    fHandler = open(fileName,'rb')
    loaded_model = pickle.load(fHandler)
    fHandler.close()
    return loaded_model


#CROSS VALIDATION
"""
X, y = PreProcessKmerDataTrain("Profiles/train_profiles_L10000.npy", [2,3,4,5])

X_pca = ApplyPCA(X,2)
plt.scatter(X_pca[:, 0], X_pca[:, 1],marker="o", c=y,s=25, edgecolor="k")
plt.savefig('2ComponentsPlots/L10000_k2345')

AnalyzeNNMLP(X.values,y,50)
AnalyzeNLSVM(X.values,y,50)
"""


#PRACTICAL TEST
nPC = 50
"""
cValue = 10
kern = "linear"
X, y = PreProcessKmerDataTrain("Profiles/train_profiles_L5000.npy", [2,3,4,5])
X = ApplyPCA(X.values,nPC)
modelSVM = svm.SVC(gamma = 'auto', C=cValue, kernel=kern).fit(X,y)
SaveModel('Models',modelSVM,'model_profiles_L5000')
"""
modelSVM = LoadModel('Models/model_profiles_L5000')

groups = PreProcessKmerDataTest("Profiles/test_profiles.npy", [2,3,4,5])

results = []
for spcs in groups.keys():
    print(spcs)
    rslt = []
    for l, data in groups[spcs].items():
        x = ApplyPCA(np.array(data[0]),nPC)
        y = data[1]
        rslt.append(modelSVM.score(x,y))
    results.append(rslt)
results = np.array(results)

np.save('plotResults',results)
#results = np.load('plotResults.npy')

result_mean = []
result_std = []
for i in range(len(results[0])):
    result_mean.append(np.mean(results[:,i]))
    result_std.append(np.std(results[:,i]))

plt.errorbar(range(len(results[0])),result_mean,result_std,xerr=None, fmt='o', ecolor='r', capthick=2, barsabove=True, linestyle='-',color='b')
plt.xlabel('Length of sequence')
plt.ylabel('Accuracy')
plt.legend()    
plt.savefig('plotResults')


########################################################################################################################
"""
def neural_network_from_dataframe(X, y):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5)).fit(X,y)
    return model

def only_testing_neural_network(X_test,Y_test,model):
     #YTest_predicted = model.predict(X_test)   
     Score = model.score(X_test,Y_test)
     return Score
"""

"""
Test area
"""
#model=neural_network_from_dataframe(X, y)
#score = only_testing_neural_network(X_test,y_test,model)

"""
tuples_test=[('bact','a',[0.5,0,0.5,0]),('bact','b',[0,0.5,0,0.5]), ('bact','c',[0,0,0,1]), ('bact','d',[0,0.2,0,0.8]),
             ('bact','e',[0.9,0,0.1,0.]), ('bact','f',[0,0.1,0,0.9]), ('bact','g',[0,0.3,0.1,0.6]), ('bact','h',[0,0.2,0,0.8])
                    ]
"""
#res=Tuples2DF(tuples_test,[1])
#resPCA=Principal_composant_analysis(res)
#plt.scatter(resPCA[:, 0], resPCA[:, 1],marker='o',s=25, edgecolor='k')
#cluster_classifier=clustering_kmean(resPCA)
#plt.scatter(resPCA[:, 0], resPCA[:, 1], s=10, c=cluster_classifier.labels_)


#X, y= PreProcessKmer("profiles.npy", [1])


##Stratification:
"""

"""

## installing libraries
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# for tensorflow you first have to install it
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import pandas as pd

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## setting working directory, in the second line add your directory
print("Current Working Directory " , os.getcwd())
os.chdir("/Users/bejbcia/pythonprojects/Machine Learning/ML Challenge")

## loading the data
written_train = np.load("written_train.npy", allow_pickle=True)
written_test = np.load("written_test.npy", allow_pickle=True)
spoken_train = np.load("spoken_train.npy", allow_pickle=True)
spoken_test = np.load("spoken_test.npy", allow_pickle=True)
match_train = np.load("match_train.npy", allow_pickle=True)



# subsetting the data to get some test set
written_train_test = written_train[:4000,:]
written_train_train = written_train[4000:,:]
spoken_train_test = spoken_train[:4000]
spoken_train_train = spoken_train[4000:]
match_train_test = match_train[:4000] 
match_train_train = match_train[4000:]


# fucntion to determine how many true/false values are in target
def true_false(y):
    unique,counts=np.unique(y,return_counts=True)
    return dict(zip(unique,counts))

## Feature engineering ##

def spoken_features(data,functions):
    features=[]
    for example in data:
        feat=np.concatenate([fun(example,axis=0) for fun in functions])
        features.append(feat)
    return np.array(features)

#feature engineering on spoken
def feat_eng_spoken(spoken,summaries):
    X_s=spoken_features(spoken,summaries)
    #lengths of spoken
    lens=np.array([example.shape[0] for example in spoken])
    lens=lens.reshape(lens.shape[0],1)
    #concatenate all the parts
    X=np.concatenate((X_s,lens),axis=1)
    #standardise
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled

#define functions to be applied to spoken data
summaries=[np.mean, np.max, np.min, np.std, np.median, kurtosis, iqr, skew, np.sum, np.argmin, np.argmax]


# Preparing training data
written_train=written_train.astype('float32')
#normalise written
written_train/=255
#reshape it into 28 x 28 matrix
written_train_m=written_train.reshape(written_train.shape[0],28,28,1)

#apply functions on sproken
spoken_train_m=feat_eng_spoken(spoken_train,summaries)


# Perparing the data for the prediction
#written
written_test=written_test.astype('float32')
#normalise written
written_test/=255
#reshape it into 28 x 28 matrix
written_test_m=written_test.reshape(written_test.shape[0],28,28,1)

#apply functions on sproken
spoken_test_m=feat_eng_spoken(spoken_test,summaries)
print(spoken_test_m.shape)
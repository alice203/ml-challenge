## installing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis, iqr

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#load data
written_train = np.load("written_train.npy", allow_pickle = True)
written_test = np.load("written_test.npy", allow_pickle = True)
spoken_train = np.load("spoken_train.npy", allow_pickle = True)
spoken_test = np.load("spoken_test.npy", allow_pickle = True)
match_train = np.load("match_train.npy", allow_pickle = True)

#define y_train_full
y_train_full=match_train

# function to determine how many true/false values are in target
def true_false(y):
    unique,counts=np.unique(y,return_counts=True)
    return dict(zip(unique,counts))

#################################################### Feature engineering ############################

def spoken_features(data,functions):
    features=[]
    for example in data:
        feat=np.concatenate([fun(example,axis=0) for fun in functions])
        features.append(feat)
    return np.array(features)
    
def feat_eng(written,spoken,summaries):
    X_w=written
    #feature engineering on spoken
    X_s=spoken_features(spoken,summaries)
    #lengths of spoken
    lens=np.array([example.shape[0] for example in spoken])
    lens=lens.reshape(lens.shape[0],1)
    #concatenate all the parts
    X=np.concatenate((X_w,X_s,lens),axis=1)
    #standardise
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled

#feature engineering on spoken only
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
summaries=[np.mean, np.max, np.min, np.std,np.median, kurtosis, iqr, skew, np.sum, np.argmin, np.argmax]

#perparing the data for the model
#written
written_train=written_train.astype('float32')
#normalise written
written_train/=255
#reshape it into 28 x 28 matrix
written_train_m=written_train.reshape(written_train.shape[0],28,28,1)
#get test data
w_train, w_val, y_train, y_val = train_test_split(written_train_m, y_train_full, test_size=0.1111, random_state=999)
#spoken
#apply functions on sproken
spoken_train_m=feat_eng_spoken(spoken_train,summaries)

#################################################### Train-Test-Split ############################

w_train, w_val, y_train, y_val = train_test_split(written_train_m, y_train_full, test_size=0.1111, random_state=999)
s_train, s_val, y_train, y_val = train_test_split(spoken_train_m, y_train_full, test_size=0.1111, random_state=999)


## Group 29
## Contributors: Laura Gärtner, Alicja Ciuńczyk, Alicia Horsch
## CodaLab account: AlicjaCiunczyk

## Importing all the packages used ##
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import skew, kurtosis, iqr
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Input, Dense, Concatenate, concatenate, Conv2D, Flatten, MaxPooling2D, LeakyReLU

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## loading the data
written_train = np.load("written_train.npy", allow_pickle = True)
written_test = np.load("written_test.npy", allow_pickle = True)
spoken_train = np.load("spoken_train.npy", allow_pickle = True)
spoken_test = np.load("spoken_test.npy", allow_pickle = True)
match_train = np.load("match_train.npy", allow_pickle = True)

# fucntion to determine how many true/false values are in target
def true_false(y):
    unique,counts=np.unique(y,return_counts=True)
    return dict(zip(unique,counts))

## Feature engineering ##

# function used to compute different statistics
def spoken_features(data,functions):
    features=[]
    for example in data:
        feat=np.concatenate([fun(example,axis=0) for fun in functions])
        features.append(feat)
    return np.array(features)

#feature engineering on spoken
def feat_eng_spoken(spoken,summaries):
    #all the statistics
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

#statistics used
summaries=[np.mean, np.max, np.min, np.std, np.median, kurtosis, iqr, skew, np.sum, np.argmin, np.argmax]

## Preparing training data ##
written_train=written_train.astype('float32')
#normalise written
written_train/=255
#reshape it into 28 x 28 matrix
written_train_m=written_train.reshape(written_train.shape[0],28,28,1)

#apply functions on sproken
spoken_train_m=feat_eng_spoken(spoken_train,summaries)

## Perparing the data for the prediction ##
#written
written_test=written_test.astype('float32')
#normalise written
written_test/=255
#reshape it into 28 x 28 matrix
written_test_m=written_test.reshape(written_test.shape[0],28,28,1)

#apply functions on sproken
spoken_test_m=feat_eng_spoken(spoken_test,summaries)
print(spoken_test_m.shape)

## MODEL ##

# Multiple input model
# CNN for written
# Features just for spoken as defined above 

## written
input_written = Input(shape=(28,28,1))
layer1_w = Conv2D(32, kernel_size=4, activation='relu')(input_written)
layer2_w = LeakyReLU(alpha=0.1)(layer1_w)
layer3_w = MaxPooling2D(pool_size=(2, 2))(layer2_w)
layer4_w = Conv2D(16, kernel_size=4, activation='relu')(layer3_w)
layer5_w = MaxPooling2D(pool_size=(2, 2))(layer4_w)
flat = Flatten()(layer5_w)

## spoken 
input_spoken = Input(shape=(144,))
layer1_s = Dense(50, activation="relu")(input_spoken)
layer2_s = Dense(50, activation="relu")(layer1_s)

# combined
combined = concatenate([flat, layer2_s])
 
layer1_c=Dense(50, activation='relu')(combined)
layer2_c=Dense(50, activation='relu')(layer1_c)
output = Dense(1, activation="sigmoid")(layer2_c)
 
model = Model(inputs=[input_written,input_spoken], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Fitting the model, 50 epochs with batch size 10 worked the best on our own test data
model.fit([written_train_m, spoken_train_m], [match_train], epochs=50, batch_size=10)

# predict, this gives probabilities
y_pred = model.predict([written_test_m,spoken_test_m])

# transform probabilities in class values of 0 and 1
y_pred2=np.array([1 if e>=0.5 else 0 for e in y_pred])

# how many trues and false
true_false(y_pred2)

np.save("result", y_pred2)
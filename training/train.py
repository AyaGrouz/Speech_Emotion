# Provides a way of using operating system dependent functionality. 
import os

# LibROSA provides the audio analysis
import librosa
# Need to implictly import from librosa
import librosa.display

# Import the audio playback widget
import IPython.display as ipd
from IPython.display import Image

# Enable plot in the notebook
import matplotlib as plt

# These are generally useful to have around
import numpy as np
import pandas as pd


# To build Neural Network and Create desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split


# import required libraries
import os
import sys
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
np.random.seed(42) 

from utils.feature_extraction import get_features_dataframe
from utils.feature_extraction import get_audio_features 
 
# trainfeatures, trainlabel = get_features_dataframe(train_df, sampling_rate)
# testfeatures, testlabel = get_features_dataframe(test_df, sampling_rate)

# I have ran above 2 lines and get the featured dataframe. 
# and store it into pickle file to use it for later purpose.
# it take too much time to generate features(around 30-40 minutes). 

def train_test_split_preprocess (dataset) : 
    # Load the pickled files from the dataset
    trainfeatures = dataset.to_path() + "/trainfeatures.pkl"
    trainlabel = dataset.to_path() + "/trainlabel.pkl"
    testfeatures = dataset.to_path() + "/testfeatures.pkl"
    testlabel = dataset.to_path() + "/testlabel.pkl"
    
    train_features_df = pd.read_pickle(trainfeatures)
    train_label_df = pd.read_pickle(trainlabel)
    test_features_df = pd.read_pickle(testfeatures)
    test_label_df = pd.read_pickle(testlabel)

    train_features_df = train_features_df.fillna(0)
    test_features_df =test_features_df.fillna(0)  

# By using .ravel() : Converting 2D to 1D e.g. (512,1) -> (512,). To prevent DataConversionWarning

    X_train = np.array(train_features_df)
    y_train = np.array(trainlabel).ravel()
    X_test = np.array(test_features_df)
    y_test = np.array(testlabel).ravel() 

# One-Hot Encoding
    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))  

# Changing dimensions 

    x_traincnn =np.expand_dims(X_train, axis=2)
    x_testcnn= np.expand_dims(X_test, axis=2)  
    return  [x_traincnn , trainlabel , x_testcnn , testlabel ]


# Train the model, return the model
def train_model(dataset , param):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    l= train_test_split_preprocess (dataset) 
    x_traincnn = l[0]
    trainlabel = l[1]
    x_testcnn = l[2]
    testlabel = l[3]
 
    model = Sequential()

    model.add(Conv1D(256, 5,padding='same',
                 input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(trainlabel.shape[1]))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6) 

    model.compile(loss=param['loss'], optimizer=opt,metrics=param['metric']) 

    return model


# Evaluate the metrics for the model
def get_model_metrics(model, dataset): 
    l= train_test_split_preprocess (dataset) 
    x_traincnn = l[0]
    trainlabel = l[1]
    x_testcnn = l[2]
    testlabel = l[3]
   # """Construct a dictionary of metrics for the model"""
    predictions = model.predict(x_testcnn)
    fpr, tpr, thresholds = metrics.roc_curve(testlabel,predictions)
    model_metrics = {
        "auc": (
           metrics.auc(
                fpr, tpr))}
    print(model_metrics)
    return model_metrics 
import pandas as pd 
import numpy as np 
import csv 
import seaborn as sb
import warnings  
warnings.filterwarnings('ignore')
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import listdir

'''
LSTM
'''

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from tensorflow import keras
from keras import backend as K

'''
CNN
'''
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D
from keras.models import Model
import time
import sys
import sklearn 
from sklearn.preprocessing import LabelEncoder
import keras


def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")

    #dropping id column
    X_train = X_train.drop('id', axis = 1)
    X_test = X_test.drop('id', axis = 1)
    y_train = y_train.drop('id', axis = 1)
    
    #reshuffling data
    X_train['y'] = y_train
    X_train = X_train.sample(frac=1).reset_index(drop=True)
    y_train = X_train['y']
    X_train = X_train.drop('y', axis = 1)
    
    return X_train, X_test, y_train



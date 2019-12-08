# The utilities functions file for the EEG/EMG time series classification taks
# A bunch of imports:
import sys
import os

import pandas as pd
import numpy as np
import random as rn
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
import pyeeg

from scipy import signal
from scipy.signal import (welch, medfilt, wiener,savgol_filter)
from scipy.integrate import simps

import matplotlib.pyplot as plt

#Keras

#Sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.svm import (SVC, SVR)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import (StratifiedKFold, KFold)

from sklearn.metrics import (accuracy_score, make_scorer, balanced_accuracy_score, roc_auc_score, mean_squared_error)

from sklearn.model_selection import GridSearchCV

from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.metrics import confusion_matrix

from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from pystruct.learners import OneSlackSSVM
from sklearn.ensemble import VotingClassifier

# Signal statistics:
import numpy as np
import scipy
from scipy import integrate
import biosppy
from biosppy.signals import eeg, emg
from scipy.signal import find_peaks,peak_prominences,peak_widths,periodogram
from scipy.stats import kurtosis,skew

def load_data():
    print("Loading xtrain...")
    xtrain_eeg1 = pd.read_csv("train_eeg1.csv").drop("Id", axis = 1)
    xtrain_eeg2 = pd.read_csv("train_eeg2.csv").drop("Id", axis = 1)
    xtrain_emg = pd.read_csv("train_emg.csv").drop("Id", axis = 1)
    print("Shapes:", xtrain_eeg1.shape, xtrain_eeg2.shape, xtrain_emg.shape)

    print("Loading ytrain...")
    ytrain = pd.read_csv("train_labels.csv").drop("Id", axis = 1)
    print("Shape:", ytrain.shape)

    print("Loading xtest...")
    xtest_eeg1 = pd.read_csv("test_eeg1.csv").drop("Id", axis = 1)
    xtest_eeg2 = pd.read_csv("test_eeg2.csv").drop("Id", axis = 1)
    xtest_emg = pd.read_csv("test_emg.csv").drop("Id", axis = 1)
    print("Shapes:", xtest_eeg1.shape, xtest_eeg2.shape, xtest_emg.shape)
    return xtrain_eeg1, xtrain_eeg2, xtrain_emg, ytrain, xtest_eeg1, xtest_eeg2, xtest_emg

def simple_statistics(sig):
    """ Tested """
    # Check if it is a 1d array
    if (len(sig.shape) > 1) and (sig.shape[1]!=1):
        return [np.mean(sig, axis=1), np.median(sig, axis=1),
                    np.std(sig, axis=1), np.max(sig, axis=1),
                    np.min(sig, axis=1), kurtosis(sig, axis=0),
                    skew(sig, axis=0)]
    else:
        return [np.mean(sig), np.median(sig), np.std(sig),
                np.max(sig), np.min(sig), float(kurtosis(sig)),
                float(skew(sig))]

def process_EEG(eeg_sig, fs=128):
    # Statistical Features
    simple_stats = simple_statistics(eeg_sig)

    # if eeg_sig.shape < 2:
    #     eeg_sig = eeg_sig.reshape()
    # Power Features
    [_, theta, alpha_low,alpha_high,beta, gamma]= eeg.get_power_features(signal=eeg_sig, sampling_rate=fs)

    ts1 = simple_statistics(theta)
    al1 = simple_statistics(alpha_low)
    ah1 = simple_statistics(alpha_high)
    b1 = simple_statistics(beta)
    g1 = simple_statistics(gamma)

    return np.array([*simple_stats, *ts1, *al1, *ah1, *b1, *g1])

def process_EMG(emg_sig, fs=128):
    # Statistical Features
    simple_stats = simple_statistics(emg_sig)

    # EMG features from biosppy, not sure this is very helpful
    # For some reason, only works if I pass in double the sampling rate of 128Hz
    # [ts, filtered, onsets] = emg.emg(emg_sig, sampling_rate=2*128)
    # onset_simple_stats = simple_statistics(np.diff(onsets))
    # Did not yield very consistent onsets. I think the function requires a higher
    # sampling frequency, could upsample?

    # GEt the total power of signal
    sig_pow = np.sum(scipy.signal.periodigram(emg_sig)[1])

    # This might be good to use :
    # Peak Features
    # [peaks,_] = find_peaks(raw_signal)
    # pprom = peak_prominences(raw_signal,peaks)[0]
    # contour_heights = raw_signal[peaks] - pprom
    # pwid = peak_widths(raw_signal,peaks,rel_height=0.4)[0]
    # [ppmean,ppstd,_,ppmin] = simple_statistics(pprom)
    # [pwmean,pwstd,pwmax,pwmin] = simple_statistics(pwid)

    # return np.array([std,maxv,minv,maxHFD, kurt,sk,ppmean,ppstd,ppmin,pwmean,pwstd,pwmax,pwmin])
    return np.array([*simple_stats, sig_pow])

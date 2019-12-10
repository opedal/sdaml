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
import csv

from scipy import signal
from scipy.signal import (welch, medfilt, wiener,savgol_filter)

import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.svm import (SVC, SVR)
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import (StratifiedKFold, KFold)
from sklearn.metrics import (accuracy_score, make_scorer, balanced_accuracy_score, roc_auc_score, mean_squared_error)
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
from pystruct.learners import OneSlackSSVM

# Signal statistics:
import numpy as np
import scipy
from scipy import integrate
import biosppy
from biosppy.signals import eeg, emg
from scipy.signal import find_peaks,peak_prominences,peak_widths,periodogram
from scipy.stats import kurtosis,skew
import yasa
from sklearn.utils.class_weight import compute_class_weight

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

def make_submission(filename, predictions):
    SAMPLE_FILE_PATH = "sample.csv"
    sample =  pd.read_csv(SAMPLE_FILE_PATH)
    sample["y"] = predictions
    sample.to_csv(filename, index= False)

# TODO:
# 1. Test simple stats (input ndarray or dataframe, output ndarray) x
# 2. Test power features (input ndarray or dataframe, output ndarray) x
# 3. Run process_EEG with just those two, and run it on the CRF. See result. Should
# be equal to or better than SVM. Also run it quickly on SVM.
# 4. Add peak_features function to EEG
# 5. Add EMG Processing
# 6. Test with leave-one-subject-out cv

def simple_statistics(sig, fs=128):
    """ TESTED for (nxd) matrix input """
    # Check if it is not a 1d array
    if (len(sig.shape) > 1) and (sig.shape[1]!=1):
        return np.array([np.mean(sig, axis=1), np.median(sig, axis=1),
                    np.std(sig, axis=1), np.max(sig, axis=1),
                    np.min(sig, axis=1), kurtosis(sig, axis=1),
                    skew(sig, axis=1)]).T
    else:
        print("Not Tested with this input!")
        return np.array([np.mean(sig), np.median(sig), np.std(sig),
                np.max(sig), np.min(sig), float(kurtosis(sig)),
                float(skew(sig))])



def power_features(sig, fs=128):
    """ TESTED for (nxd) matrix input, returns (nxk) = (nx35) matrix output

     Conclusion: Passing in signals separately is the same as passing them in together.

     """
    [_, theta, alpha_low,alpha_high,beta, gamma]= eeg.get_power_features(signal=sig.T, sampling_rate=fs)
    ts1 = simple_statistics(theta.T)
    al1 = simple_statistics(alpha_low.T)
    ah1 = simple_statistics(alpha_high.T)
    b1 = simple_statistics(beta.T)
    g1 = simple_statistics(gamma.T)
    return np.concatenate((ts1, al1, ah1, b1, g1), axis=1)

def andreas_power_features(eeg_signal, fs=128):
    for i in (np.arange(eeg_signal.shape[0] / 100) + 1):
        if i == 1:
            df = yasa.bandpower(eeg_signal.iloc[0:int(100*i),:].values, sf=fs)
        else:
            df = df.append(yasa.bandpower(eeg_signal.iloc[int(100*(i-1)):int(100*i),:].values, sf=fs))

    df = df.set_index(np.arange(eeg_signal.shape[0]))
    df = df.drop(columns = ["FreqRes","Relative"], axis = 1)
    return np.array(df)

def peak_features(sig, fs=128):
    return

def process_EEG(eeg_sig, fs=128):
    """ # TODO: Properly join these three matrices, concat is not the proper way"""
    # Statistical Features
    simple_stats = simple_statistics(eeg_sig, fs=fs)
    # power_feats = power_features(eeg_sig, fs=fs)
    power_feats = andreas_power_features(eeg_sig, fs=fs)
    # peak_feats = peak_features(eeg_sig, fs=fs)
    return np.concatenate((simple_stats, power_feats), axis=1)

def process_EMG(emg_sig, fs=128):
    # Statistical Features
    # simple_stats = simple_statistics(emg_sig)

    # EMG features from biosppy, not sure this is very helpful
    # For some reason, only works if I pass in double the sampling rate of 128Hz
    # [ts, filtered, onsets] = emg.emg(emg_sig, sampling_rate=2*128)
    # onset_simple_stats = simple_statistics(np.diff(onsets))
    # Did not yield very consistent onsets. I think the function requires a higher
    # sampling frequency, could upsample?

    # GEt the total power of signal
    # sig_pow = np.sum(scipy.signal.periodogram(emg_sig)[1])

    # This might be good to use :
    # Peak Features
    # [peaks,_] = find_peaks(raw_signal)
    # pprom = peak_prominences(raw_signal,peaks)[0]
    # contour_heights = raw_signal[peaks] - pprom
    # pwid = peak_widths(raw_signal,peaks,rel_height=0.4)[0]
    # [ppmean,ppstd,_,ppmin] = simple_statistics(pprom)
    # [pwmean,pwstd,pwmax,pwmin] = simple_statistics(pwid)

    # return np.array([std,maxv,minv,maxHFD, kurt,sk,ppmean,ppstd,ppmin,pwmean,pwstd,pwmax,pwmin])
    return process_EEG(emg_sig, fs=fs)

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
from tqdm import tqdm
import itertools

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
from multipledispatch import dispatch

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

@dispatch(pd.core.frame.DataFrame)
def andreas_power_features(eeg_signal, fs=128):
    for i in (np.arange(eeg_signal.shape[0] / 100) + 1):
        if i == 1:
            df = yasa.bandpower(eeg_signal.iloc[0:int(100*i),:].values, sf=fs)
        else:
            df = df.append(yasa.bandpower(eeg_signal.iloc[int(100*(i-1)):int(100*i),:].values, sf=fs))

    df = df.set_index(np.arange(eeg_signal.shape[0]))
    df = df.drop(columns = ["FreqRes","Relative"], axis = 1)
    return np.array(df)

@dispatch(np.ndarray)
def andreas_power_features(eeg_signal, fs=128):
    for i in (np.arange(eeg_signal.shape[0] / 100) + 1):
        if i == 1:
            df = yasa.bandpower(eeg_signal[0:int(100*i),:], sf=fs)
        else:
            df = df.append(yasa.bandpower(eeg_signal[int(100*(i-1)):int(100*i),:], sf=fs))

    df = df.set_index(np.arange(eeg_signal.shape[0]))
    df = df.drop(columns = ["FreqRes","Relative"], axis = 1)
    return np.array(df)

def peak_features(sig, fs=128):
    return

def total_power(sig, fs=128):
    mse = ((sig - np.mean(sig, axis=1))**2).mean(axis=1)
    return mse

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
    eeg_ = process_EEG(emg_sig, fs=fs)
    return eeg_

def losocv(eeg1, eeg2, emg, y, model, fs=128):
    """Leave one subject out cross validation"""

    epochs = 21600
    num_sub = 3
    # Indices of the subjects
    sub_indices = [np.arange(0, epochs), np.arange(epochs, epochs*2),np.arange(epochs*2, epochs*3)]
    res = []

    for i in tqdm(range(len(sub_indices))):

        # For the ith iteration, select as trainin the sub_indices other than those at index i for train_index
        train_index = np.concatenate([sub_indices[(i+1)%num_sub], sub_indices[(i+2)%num_sub]])
        eeg1_train = eeg1[train_index]
        eeg2_train = eeg2[train_index]
        emg_train = emg[train_index]
        y_train = y[train_index]

        # The test subject is the one at index i
        test_index = sub_indices[i]
        eeg1_test = eeg1[test_index]
        eeg2_test = eeg2[test_index]
        emg_test = emg[test_index]
        y_test = y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(xtest)
        res.append(sklearn.metrics.balanced_accuracy_score(y_test, y_pred))
    return res

def losocv_CRF(eeg1, eeg2, emg, y, C=0.5, weight_shift=1.5, fs=128):
    """Leave one subject out cross validation for the CRF model becasuse it requires
    special datahandling. Input should be a Pandas Dataframe."""

    epochs = 21600
    num_sub = 3
    # Indices of the subjects
    sub_indices = [np.arange(0, epochs), np.arange(epochs, epochs*2),np.arange(epochs*2, epochs*3)]
    res = []

    for i in tqdm(range(len(sub_indices))):

        # For the ith iteration, select as trainin the sub_indices other than those at index i for train_index
        train_index = np.concatenate([sub_indices[(i+1)%num_sub], sub_indices[(i+2)%num_sub]])
        eeg1_train = eeg1.values[train_index]
        eeg2_train = eeg2.values[train_index]
        emg_train = emg.values[train_index]
        y_train = y.values[train_index]

        # The test subject is the one at index i
        test_index = sub_indices[i]
        eeg1_test = eeg1.values[test_index]
        eeg2_test = eeg2.values[test_index]
        emg_test = emg.values[test_index]
        y_test = y.values[test_index]

        # CRF Model Preprocessing
        eeg1_ = process_EEG(eeg1_train)
        eeg2_ = process_EEG(eeg2_train)
        emg_ = process_EMG(emg_train)
        xtrain_ = np.concatenate((eeg1_, eeg2_, emg_), axis=1)
        ytrain_classes = np.reshape(y_train, (y_train.shape[0],))
        ytrain_ = y_train

        eeg1_ = process_EEG(eeg1_test)
        eeg2_ = process_EEG(eeg2_test)
        emg_ = process_EEG(emg_test)
        xtest_ = np.concatenate((eeg1_, eeg2_, emg_), axis=1)
        ytest_ = y_test

        xtrain_crf = np.reshape(xtrain_, (2, -1, xtrain_.shape[1])) # Reshape so that it works with CRF
        ytrain_crf = np.reshape(ytrain_, (2, -1)) -1 # Reshape so that it works with CRF

        # CRF Model fitting:
        classes = np.unique(ytrain_)
        weights_crf = compute_class_weight("balanced", list(classes), list(ytrain_classes))
        weights_crf[0] = weights_crf[0]+weight_shift+1
        weights_crf[1] = weights_crf[1]+weight_shift

        model = ChainCRF(class_weight=weights_crf)
        ssvm = OneSlackSSVM(model=model, C=0.5, max_iter=2000)
        ssvm.fit(xtrain_crf, ytrain_crf)

        # Test on the third guy
        xtest_crf = np.reshape(xtest_, (1, -1, xtest_.shape[1]))
        ytest_crf = np.reshape(ytest_, (1, -1)) -1
        y_pred_crf = ssvm.predict(xtest_crf)
        y_pred_crf = np.asarray(y_pred_crf).reshape(-1) + 1

        resy = sklearn.metrics.balanced_accuracy_score(ytest_, y_pred_crf)
        res.append(resy)
        print("BMAC:", resy )
    return res

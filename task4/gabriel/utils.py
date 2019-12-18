# The utilities functions file for the EEG/EMG time series classification taks
# A bunch of imports:
import sys
import os
import pandas as pd
import numpy as np
import numpy
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
from numba import jit
import entropy
from math import log, floor

# TODO:
# 0. Vectorize DFA
# 1. Add proper EMG features (total power etc?)
# 2. Add cleaned peak features

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

def power_features(sig, fs=128):
    """ TESTED for (nxd) matrix input, returns (nxk) = (nx35) matrix output

     Conclusion: Passinzg in signals separately is the same as passing them in together.

     """
    [_, theta, alpha_low,alpha_high,beta, gamma]= eeg.get_power_features(signal=sig.T, sampling_rate=fs)
    ts1 = simple_statistics(theta.T)
    al1 = simple_statistics(alpha_low.T)
    ah1 = simple_statistics(alpha_high.T)
    b1 = simple_statistics(beta.T)
    g1 = simple_statistics(gamma.T)
    return np.concatenate((ts1, al1, ah1, b1, g1), axis=1)

@dispatch(np.ndarray)
def simple_statistics(sig, fs=128):
    """ TESTED for (nxd) matrix input """
    # Check if it is not a 1d array
    return np.array([np.mean(sig, axis=1), np.median(sig, axis=1),
                np.std(sig, axis=1), np.max(sig, axis=1),
                np.min(sig, axis=1), kurtosis(sig, axis=1),
                skew(sig, axis=1)]).T

@dispatch(pd.core.frame.DataFrame)
def advanced_statistics(signal, fs=128):
    return advanced_statistics(signal.values, fs=fs)

# @dispatch(np.ndarray)
@jit(nopython=True)
def advanced_statistics(signal, fs=128):
    K_boundary = 10         # to be tuned
    t_fisher = 12          # to be tuned
    d_fisher = 40          # to be tuned
    features_num = 11
    threshold =  0.0009
    advanced_stats = np.zeros((signal.shape[0],features_num))
    print("Gathering advanced statistics...")
    for i in tqdm((np.arange(signal.shape[0]))):
        feat_array = np.array([
                               pyeeg.fisher_info(signal[i,:], t_fisher, d_fisher),
                               pyeeg.pfd(signal[i,:]),
                               pyeeg.dfa(signal[i,:]),
                               pyeeg.hfd(signal[i,:], K_boundary),
                               np.sum((abs(signal[i,:]) ** (-0.3)) > 20),
                               np.sum((abs(signal[i,:])) > threshold),
                               np.std(abs(signal[i,:]) ** (0.05)),
                               np.sqrt(np.mean(np.power(np.diff(signal[i,:]), 2))),
                               np.mean(np.abs(np.diff(signal[i,:]))),
                               np.mean(signal[i,:] ** 5),
                               np.sum(signal[i,:] ** 2)
                               ])
        advanced_stats[i, :] = feat_array
    return advanced_stats

@dispatch(pd.core.frame.DataFrame)
def vectorized_adv_stat(signal, fs=128):
    return vectorized_adv_stat(signal.values, fs=128)

@dispatch(np.ndarray)
def vectorized_adv_stat(signal, fs=128):
    K_boundary = 10         # to be tuned
    t_fisher = 12          # to be tuned
    d_fisher = 40          # to be tuned
    features_num = 11
    threshold =  0.0009
    # Missing fisher info and dfa
    feat_array = np.array([
                           fisher_info(signal, t_fisher, d_fisher),
                           pfd(signal),
                           hfd(signal, K_boundary),
                           np.sum((np.power(np.abs(signal),(-0.3)) > 20), axis=1),
                           np.sum((np.abs(signal)) > threshold, axis=1),
                           np.std(np.power(np.abs(signal),(0.05)), axis=1),
                           np.sqrt(np.mean(np.power(np.diff(signal, axis=1), 2), axis=1)),
                           np.mean(np.abs(np.diff(signal, axis=1)), axis=1),
                           np.mean(np.power(signal, 5), axis=1),
                           np.sum(np.power(signal, 2), axis=1)
                           ]).T
    feat_array = np.concatenate((feat_array, matrix_dfa(signal)), axis=1) # Concatenate the lengthy dfa calculatoin
    return feat_array

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

@dispatch(pd.core.frame.DataFrame)
def simple_power_features(eeg_signal, fs=128):
    return simple_power_features(eeg_signal.values, fs=fs)

@dispatch(np.ndarray)
def simple_power_features(eeg_signal, fs=128):
    print("SIZE:", eeg_signal.shape)
    df = yasa.bandpower(eeg_signal, sf=fs)
    df = df.set_index(np.arange(eeg_signal.shape[0]))
    df = df.drop(columns = ["FreqRes","Relative"], axis = 1)
    return np.array(df)

@dispatch(np.ndarray)
def peak_statistics(sig, fs=128):
    Rprom_arr = np.zeros([])
    Rwidth_arr = np.array([])
    res = np.zeros((sig.shape[0], 7))
    for i in tqdm((np.arange(sig.shape[0]))):
        print("I:", i)
        RRpeaks = find_peaks(sig[i, :])[0]
        Rprom = peak_prominences(sig[i, :], RRpeaks)[0]
        Rprom = np.reshape(Rprom, (-1, 1))
        Rwidth = peak_widths(sig[i, :], RRpeaks, rel_height=0.4)[0]
        Rwidth = np.reshape(Rwidth, (-1, 1))
        import pdb; pdb.set_trace()
        res = np.vstack(res, np.concatenate((simple_statistics(Rprom_arr, fs=128), simple_statistics(Rwidth_arr, fs=128)), axis=1))
    return res[1:]

def total_power(sig, fs=128):
    # mse = ((sig - np.mean(sig, axis=1))**2).mean(axis=1)
    return np.mean(np.power(sig, 2), axis=1)

def process_EEG(eeg_sig, fs=128):
    """ # TODO: Properly join these three matrices, concat is not the proper way"""
    simple_stats = simple_statistics(eeg_sig, fs=fs)
    power_feats = andreas_power_features(eeg_sig, fs=fs)
    advanced_feats = vectorized_adv_stat(eeg_sig, fs=fs)
    return np.concatenate((simple_stats, power_feats, advanced_feats), axis=1)

def process_EMG(emg_sig, fs=128):
    # simple_stats = simple_statistics(emg_sig, fs=fs)
    # advanced_feats = vectorized_adv_stat(emg_sig, fs=fs)
    # return np.concatenate((simple_stats, advanced_feats), axis=1)
    return process_EEG(emg_sig, fs=fs)

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

    for i in range(len(sub_indices)):

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

        eeg1_t = process_EEG(eeg1_test)
        eeg2_t = process_EEG(eeg2_test)
        emg_t = process_EMG(emg_test)
        xtest_ = np.concatenate((eeg1_t, eeg2_t, emg_t), axis=1)
        ytest_ = y_test

        xtrain_crf = np.reshape(xtrain_, (2, -1, xtrain_.shape[1])) # Reshape so that it works with CRF
        ytrain_crf = np.reshape(ytrain_, (2, -1)) -1 # Reshape so that it works with CRF

        # CRF Model fitting:
        classes = np.unique(ytrain_)
        weights_crf = compute_class_weight("balanced", list(classes), list(ytrain_classes))
        weights_crf[0] = weights_crf[0]+2.5*weight_shift
        weights_crf[1] = weights_crf[1]+1.5*weight_shift

        model = ChainCRF(class_weight=weights_crf)
        ssvm = OneSlackSSVM(model=model, C=C, max_iter=2000)
        ssvm.fit(xtrain_crf, ytrain_crf)

        # Test on the third guy
        xtest_crf = np.reshape(xtest_, (1, -1, xtest_.shape[1]))
        ytest_crf = np.reshape(ytest_, (1, -1)) -1
        y_pred_crf = ssvm.predict(xtest_crf)
        y_pred_crf = np.asarray(y_pred_crf).reshape(-1) + 1

        resy = sklearn.metrics.balanced_accuracy_score(ytest_, y_pred_crf)
        print("Iteration, result:", i, resy)
        res.append(resy)
    return res

def CRF_submit(eeg1, eeg2, emg, y, eeg1test, eeg2test, emgtest, C=0.9, weight_shift=0, fs=128):

    # For the ith iteration, select as trainin the sub_indices other than those at index i for train_index
    eeg1_train = eeg1.values
    eeg2_train = eeg2.values
    emg_train = emg.values
    y_train = y.values

    # The test subject is the one at index i
    eeg1_test = eeg1test.values
    eeg2_test = eeg2test.values
    emg_test = emgtest.values

    # CRF Model Preprocessing
    eeg1_ = process_EEG(eeg1_train)
    eeg2_ = process_EEG(eeg2_train)
    emg_ = process_EMG(emg_train)
    xtrain_ = np.concatenate((eeg1_, eeg2_, emg_), axis=1)
    ytrain_classes = np.reshape(y_train, (y_train.shape[0],))
    ytrain_ = y_train

    eeg1_ = process_EEG(eeg1_test)
    eeg2_ = process_EEG(eeg2_test)
    emg_ = process_EMG(emg_test)
    xtest_ = np.concatenate((eeg1_, eeg2_, emg_), axis=1)

    xtrain_crf = np.reshape(xtrain_, (3, -1, xtrain_.shape[1])) # Reshape so that it works with CRF
    ytrain_crf = np.reshape(ytrain_, (3, -1)) -1 # Reshape so that it works with CRF

    # CRF Model fitting:
    classes = np.unique(ytrain_)
    weights_crf = compute_class_weight("balanced", list(classes), list(ytrain_classes))
    weights_crf[0] = weights_crf[0]+2.5*weight_shift
    weights_crf[1] = weights_crf[1]+1.5*weight_shift

    model = ChainCRF(class_weight=weights_crf)
    ssvm = OneSlackSSVM(model=model, C=C, max_iter=2000)
    ssvm.fit(xtrain_crf, ytrain_crf)

    # Test on the third guy
    xtest_crf = np.reshape(xtest_, (2, -1, xtest_.shape[1]))
    y_pred_crf = ssvm.predict(xtest_crf)
    y_pred_crf = np.asarray(y_pred_crf).reshape(-1) + 1
    return y_pred_crf

# Optimized Fractal and Entropy functions
@dispatch(np.ndarray, int)
def hfd(X, Kmax):
    """ VECTORIZED!!! TESTED: Matches the for loop output. Can test easily comparing
    to the pyeeg.hfd() function. X now a (nxd) matrix.
    Compute Higuchi Fractal Dimension of a time series X. kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = X.shape[1]
    for k in (range(1, Kmax)):
        # Lk = np.empty(shape=[0, ])
        Lk = np.empty(shape=[X.shape[0], 1])
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(numpy.floor((N - m) / k))):
                Lmk += np.abs(X[:, m + i * k] - X[:, m + i * k - k])
            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k
            Lmk = np.reshape(Lmk, (Lmk.shape[0], 1))
            Lk = np.hstack((Lk, Lmk))

        # Remove that first placeholder column of zeros in Lk:
        Lk = Lk[:, 1:]
        L.append(numpy.log(numpy.mean(Lk, axis=1)))
        x.append([numpy.log(float(1) / k), 1]) # Fix this!!!

    (p, _, _, _) = numpy.linalg.lstsq(x, L)
    return p[0]

@dispatch(np.ndarray)
def pfd(X):
    """VECTORIZED!!! TESTED, matches the 1d time series output. Now accepts (nxd) matrices as input
    Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)
    In case 1, D is computed using Numpy's difference function.
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    n = X.shape[1]
    diff = np.diff(X, axis=1)
    N_delta = np.sum(diff[:, 1:-1] * diff[:, 0:-2] < 0, axis=1)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

@dispatch(np.ndarray, int, int)
def fisher_info(X, Tau=3, DE=1, W=None):
    """ VECTORIZED, TESTED, gives approximate results but not exact. Compute SVD Entropy from either two cases below:
    """
    if W is None:
        Y = _embed(X, Tau, DE)
        W = numpy.linalg.svd(Y, compute_uv=0)
        W = np.divide(W, np.reshape(np.sum(W, axis=1), (-1, 1)))  # normalize singular values
    return -1 * np.sum(W * numpy.log(W), axis=1)

@dispatch(np.ndarray, int, int)
def _embed(x, order=3, delay=1):
    """ VECTORIZED!! Kind of TESTED. Time-delay embedding. x is an (nxd) matrix
    """
    N = x.shape[1]
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((x.shape[0], N - (order - 1) * delay, order))
    for i in range(order):
        Y[:, :, i] = x[:, i * delay:i * delay + Y.shape[1]]
    return Y

@jit(nopython=True)
def matrix_dfa(x):
    dfa_ = np.zeros((x.shape[0],), dtype=np.float64)
    for i in (np.arange(x.shape[0])):
        dfa_[i] = _dfa(np.asarray(x[i], dtype=np.float64))
    return np.reshape(dfa_, (-1, 1))

@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

@jit('i8[:](f8, f8, f8)', nopython=True)
def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

@jit('f8(f8[:])', nopython=True)
def _dfa(x):
    """
    Utility function for detrended fluctuation analysis
    """
    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[:N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        slope = np.empty(d_len)
        intercept = np.empty(d_len)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope[i], intercept[i] = _linear_regression(ran_n, d[i])
            y = np.zeros_like(ran_n)
            # Equivalent to np.polyval function
            for p in [slope[i], intercept[i]]:
                y = y * ran_n + p
            trend[i, :] = y
        # calculate standard deviation (fluctuation) of walks in d around trend
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
        # calculate mean fluctuation over all subsequences
        fluctuations[i_n] = flucs.sum() / flucs.size

    # Filter zero
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        dfa = np.nan
    else:
        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
    return dfa

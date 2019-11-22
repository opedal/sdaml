#import libraries
import importlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
import numpy as np
import pandas as pd
import sklearn
import copy
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, make_scorer
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import feature_selection
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, chi2, f_classif
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from sklearn import svm
from numpy import loadtxt
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from bayes_opt import BayesianOptimization
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools

# SMOTE stuff
from imblearn.utils import check_sampling_strategy, check_target_type

# Neural Net:
from sklearn.neural_network import MLPClassifier
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#define functions for loading data and producing final CSV

'''
eliminate highly correlated features
'''
def to_be_eliminated(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop

'''
loading training and test datasets
'''
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

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    return X_train, X_test, y_train


'''
produce submission file
'''
def produce_solution(y):
    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        writer.writerow(['id', 'y'])
        for i in range(y.shape[0]):
            writer.writerow([float(i), y[i]])

class SMOTEClassifier():
    def __init__(self, smote, classifier):
        self.smote = smote
        self.classifier = classifier
        self.classifier_ = sklearn.base.clone(self.classifier)

    def fit(self, X, y, *args):
        # self.smote_ = copy.deepcopy(self.smote)
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        # X_smote, y_smote = self.smote.fit_resample(X, y)
        self.classifier_.fit(X_smote, y_smote, *args)
        # self.classifier.fit(X_smote, y_smote)
        return self

    def predict(self, X):
        # return self.classifier.predict(X)
        return self.classifier_.predict(X)

    def score(self, ytest, ypred):
        return balanced_accuracy_score(ytest, ypred)

    def set_params(*params):
        self.classifier_.set_params(*params)
        return

    def set_params(*params):
        self.classifier_.set_params(*params)
        return


#### PLOTTING FUNCTIONS ####

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_validation_curve(estimator, title, X, y, param_name='gamma',
                    param_range=np.linspace(.1, 1.0, 5), cv=5,
                    scoring="accuracy", n_jobs=1):

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def trained_model1(X_train, X_test, y_train):
    '''

    '''
    X_train, X_test, y_train = load_data()
    from sklearn.ensemble import VotingClassifier

    class_weights0 = {
    0 : 2.67223382045929,
    1 : 0.44382801664355065,
    2 : 2.6834381551362685
    }

    class_weights1 = {
    0 : 2.6611226611226613,
    1 : 0.4435204435204435,
    2 : 2.7061310782241015
    }

    class_weights2 = {
    0 : 2.7176220806794054,
    1 : 0.44521739130434784,
    2 : 2.591093117408907
    }

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))


    select = SelectFromModel(RandomForestClassifier(n_estimators=300, random_state=42))
    select.fit(X_train, y_train)
    X_train = pd.DataFrame(select.transform(X_train))
    X_test = pd.DataFrame(select.transform(X_test))

    clf0 = svm.SVC(class_weight=class_weights0)
    clf1 = svm.SVC(class_weight=class_weights1)
    clf2 = svm.SVC(class_weight=class_weights2)
    eclf = VotingClassifier(estimators=[('clf0', clf0), ('clf1', clf1), ('clf2', clf2)], voting='hard')

    eclf.fit(X_train, y_train)
    pred = eclf.predict(pd.DataFrame(X_test))

    produce_solution(pred)

    return pred

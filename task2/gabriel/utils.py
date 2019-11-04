#import libraries
import importlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import RFECV
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# SMOTE stuff
from imblearn.utils import check_sampling_strategy, check_target_type

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

    to_drop = to_be_eliminated(X_train)

    for i in range(len(to_drop)):
        X_train = X_train.drop(to_drop[i], axis = 1)

    for i in range(len(to_drop)):
        X_test = X_test.drop(to_drop[i], axis = 1)

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


### THE ABOVE DOESN't Work, have to create a new META Estimator that contains SMOTE: #####
class SMOTEClassifier():
    def __init__(self, smote, classifier):
        self.smote = smote
        self.classifier = classifier

    def fit(self, X, y):
        self.smote_ = copy.deepcopy(self.smote)
        X_smote, y_smote = self.smote_.fit_sample(X, y)
        self.classifier_ = copy.deepcopy(self.classifier).fit(X_smote, y_smote)
        return self

    def predict(self, X):
        return self.classifier_.predict(X)

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

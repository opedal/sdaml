# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:16:05 2019

@author: Nicolò Grometto
"""


import pandas as pd
'''
loading training and test datasets
'''
X_train = pd.read_csv('X_train.csv')
X_train.head()
import os
os.chdir('c:\\Users\\Nicolò Grometto\\Desktop\\task_0_submit')

import pandas as pd
from sklearn import datasets, linear_model 
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

train_data = pd.read_csv('train.csv')
train_data = train_data.drop('Id', axis = 1)
test_data = pd.read_csv('test.csv')

index_test = list(range(10000, 12000))

test_data = test_data.drop('Id', axis = 1)

X_train = train_data.values[:,1:]
y_train = train_data.values[:,0]
X_test = test_data.values 
y_test = test_data.values[:,0]

lr = LinearRegression()
models = cross_validate(lr, X_train, y_train, cv = 3, return_estimator = True, scoring='neg_mean_squared_error')['estimator']

model = (models[0].coef_ + models[1].coef_ + models[2].coef_)/3
#print(model.shape, X_test.shape)
y_pred = np.matmul(model,X_test.T)
print(y_pred)

y_pred_frame = pd.DataFrame({'Id':index_test, 'y':y_pred}).to_csv('submission.csv', index = False)

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:07:11 2021

@author: yeehs
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

#load all data
rec_data_final_train = pd.read_csv('Data/Model Data/rec_data_final/rec_data_final_train.csv')
rec_data_final_test = pd.read_csv('Data/Model Data/rec_data_final/rec_data_final_test.csv')
rec_data_lag_2_train = pd.read_csv('Data/Model Data/rec_data_lag_2/rec_data_lag_2_train.csv')
rec_data_lag_2_test = pd.read_csv('Data/Model Data/rec_data_lag_2/rec_data_lag_2_test.csv')
rec_data_lag_4_train = pd.read_csv('Data/Model Data/rec_data_lag_4/rec_data_lag_4_train.csv')
rec_data_lag_4_test = pd.read_csv('Data/Model Data/rec_data_lag_4/rec_data_lag_4_test.csv')
rec_data_lag_10_train = pd.read_csv('Data/Model Data/rec_data_lag_10/rec_data_lag_10_train.csv')
rec_data_lag_10_test = pd.read_csv('Data/Model Data/rec_data_lag_10/rec_data_lag_10_test.csv')
rec_data_lag_12_train = pd.read_csv('Data/Model Data/rec_data_lag_12/rec_data_lag_12_train.csv')
rec_data_lag_12_test = pd.read_csv('Data/Model Data/rec_data_lag_12/rec_data_lag_12_test.csv')

selected_columns = ['Is_Recession']
def create_lagged_data(dataset, lag_value, columns = selected_columns):
    new_data = dataset.copy()
    # For each feature, create lagged features up to the given lagged value
    for feature in columns:
        feature_data = new_data[feature]
        new_data[str(feature) + '_T-' + str(lag_value)] = feature_data.shift(lag_value)
    new_data = new_data.dropna()
    return new_data

def mse(Y_pred,Y):
    return sum((Y_pred-Y)**2)

'''
Dynamic probit model from Kauppi's paper uses the first lag of the dependent variable 'Is_Recession'
Since the first lag their paper uses looks one quarter (3 months) back, we try lags of up to 5 periods (5 months).
'''

# Create 2D lists of train and test sets: First selecting the number of lags in the predictors, and Second selecting the number of lags in the dependent variable
# this step is a bit mechanical
xlags = [0, 2, 4, 10, 12]
rec_data_train_dict = {0:rec_data_final_train, 2:rec_data_lag_2_train, 4:rec_data_lag_4_train, 10:rec_data_lag_10_train, 12:rec_data_lag_12_train}
rec_data_test_dict = {0:rec_data_final_test, 2:rec_data_lag_2_test, 4:rec_data_lag_4_test, 10:rec_data_lag_10_test, 12:rec_data_lag_12_test}

# For each lag value in predictors, obtain train and test sets for different numbers of lags in the dependent variable y
for lag in xlags:
    
    rec_data_train = rec_data_train_dict[lag]
    rec_data_test = rec_data_test_dict[lag]
    
    # Create lists of train and test sets for lags of y from 1 to 5
    # N.B. when a model has the 4th lag of y as a predictor, the model does not have the first to third lags of y. Following Kauppi.
    
    rec_data_nlag_pred_train = []
    rec_data_nlag_pred_test = []
    probit_models = []
    n_ylags = 6
    for i in range(n_ylags):
        if i == 0:
            rec_data_nlag_pred_train.append(rec_data_train.dropna())
            rec_data_nlag_pred_test.append(rec_data_test.dropna())
        else:
            rec_data_nlag_pred_train.append(create_lagged_data(rec_data_train, i))
            rec_data_nlag_pred_test.append(create_lagged_data(rec_data_test, i))
        
        # train the model, obtain AIC, BIC and pseudo R-squared
        y_train = rec_data_nlag_pred_train[i]['Is_Recession']
        X_train = rec_data_nlag_pred_train[i].drop(['Is_Recession','DATE'],1)
        X_train = sm.add_constant(X_train)
        model = Probit(y_train, X_train.astype(float))
        # arbitrary number of max iterations. Less than the default since the model seems to overfit the training data very quickly
        probit_models.append(model.fit(maxiter=100)) 
        print("Model AIC: {}, Model BIC: {}".format(probit_models[i].aic, probit_models[i].bic))
        print("Pseudo R-squared value for model with {} y-lags and {} x-lags: {}".format(i,lag,probit_models[i].prsquared))
        
        # test the model. Obtain in-sample and OOS MSE.
        y_test = rec_data_nlag_pred_test[i]['Is_Recession']
        X_test = rec_data_nlag_pred_test[i].drop(['Is_Recession','DATE'],1)
        X_test = sm.add_constant(X_test)
        y_pred_train = probit_models[i].predict(X_train)
        y_pred_test = probit_models[i].predict(X_test)
        mse_train = mse(y_pred_train, y_train)
        mse_test = mse(y_pred_test, y_test)
        print("Train MSE for model with {} y-lags and {} x-lags: {}".format(i,lag,mse_train))
        print("Test MSE for model with {} y-lags and {} x-lags: {}".format(i,lag,mse_test))
        
        # plot model performance on the training and test sets
        plt.plot(rec_data_nlag_pred_train[i]['DATE'],y_train,rec_data_nlag_pred_train[i]['DATE'],y_pred_train)
        plt.title("Probability of Recession occuring - train data ({} y-lags, {} x-lags)".format(i,lag))
        plt.legend(["actual","predicted"])
        # x-axis ticks at every 24 months. First tick starts from 2001-01-01
        plt.xticks(np.arange(12-i, len(rec_data_train['DATE']), 24), rotation=45) 
        plt.show()
        plt.plot(rec_data_nlag_pred_test[i]['DATE'],y_test,rec_data_nlag_pred_test[i]['DATE'],y_pred_test)
        plt.title("Probability of Recession occuring - test data ({} y-lags, {} x-lags)".format(i,lag))
        plt.legend(["actual","predicted"])
        # x-axis ticks at every 24 months.
        plt.xticks(np.arange(12-i, len(rec_data_test['DATE']), 24), rotation=45) 
        plt.show()
    
    
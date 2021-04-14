# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:07:11 2021

@author: yeehs
"""
import os
os.chdir("C:/Users/yeehs/Desktop/Yee Hsien/NUS/Modules/AY2021 S2/EC4308/Assignments/Project")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score,confusion_matrix,log_loss, classification_report, precision_recall_curve, f1_score, auc
from statsmodels.discrete.discrete_model import Probit
from Forecasters.TSDatasetGenerator import TSDatasetGenerator

#load all data
rec_train = pd.read_csv('rec_train.csv')
rec_test = pd.read_csv('rec_test.csv')

target = 'Is_Recession'
generator = TSDatasetGenerator()

#store some model data
pr_squared = []
is_log_losses = []
oos_log_losses = []
accuracies = []
cf_reports_train = []
cf_reports_test = []
test = generator.fit_transform(rec_train, target, 1, 0, 2)
forecasts = [1, 3, 6, 12]
idx = 0
for i in forecasts:
    pr_squared.append([])
    is_log_losses.append([])
    oos_log_losses.append([])
    accuracies.append([])
    cf_reports_train.append([])
    cf_reports_test.append([])
    for l in range(0, 4):
        for k in range(0, 4):
            # generate data with relevant forecast horizon, dependent variable lag and predictor lags
            train_dates = rec_train['DATE']
            train_dates = train_dates.iloc[max(l,k):-i,]
            rec_data_train = rec_train.drop(columns=['DATE'])
            rec_data_train = generator.fit_transform(rec_data_train, target, i, l, k)
            
            #create train data and train the model
            X_train, y_train = rec_data_train.drop(columns=['Target Feature']), rec_data_train['Target Feature']
            X_train = sm.add_constant(X_train)
            model = Probit(y_train, X_train.astype(float))
            model = model.fit(maxiter=10)
            print("Model for {}-month ahead forecast with {} y-lags and {} x-lags\n\
                  pseudo-R^2: {}".format(i, l, k, model.prsquared))
            pr_squared[idx].append(model.prsquared)
            
            #create test data and predict in-sample and OOS y.
            test_dates = rec_test['DATE']
            test_dates = test_dates.iloc[max(l,k):-i,]
            rec_data_test = rec_test.drop(columns=['DATE'])
            rec_data_test = generator.fit_transform(rec_data_test, target, i, l, k)
            X_test, y_test = rec_data_test.drop(columns=['Target Feature']), rec_data_test['Target Feature']
            X_test = sm.add_constant(X_test)
            y_pred_train = model.predict(X_train)
            y_pred_train_classes = np.where(y_pred_train > 0.5,1,0)
            y_pred_test = model.predict(X_test)
            y_pred_classes = np.where(y_pred_test > 0.5,1,0)
            
            #Evaluate in-sample and OOS performance
            cf_reports_train[idx].append(pd.DataFrame(classification_report(y_train, y_pred_train_classes, output_dict=True)))
            cf_reports_test[idx].append(pd.DataFrame(classification_report(y_test, y_pred_classes, output_dict=True)))
            cm = confusion_matrix(y_test,y_pred_classes)
            print("Confusion matrix:")
            print(cm)
            accuracy = accuracy_score(y_test,y_pred_classes)
            print("Accuracy: " + str(accuracy))
            is_lloss = log_loss(np.array(y_train), np.array(y_pred_train))
            print("In-sample Log loss: " + str(is_lloss))
            oos_lloss = log_loss(np.array(y_test), np.array(y_pred_test))
            print("Out-of-sample Log loss: " + str(oos_lloss))
            accuracies[idx].append(accuracy)
            is_log_losses[idx].append(is_lloss)
            oos_log_losses[idx].append(oos_lloss)
            
            # plot model performance on the training and test sets
            plt.plot(train_dates,y_train,train_dates,y_pred_train)
            plt.title("Probability of Recession occuring: {}-month forecast - train data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.legend(["actual","predicted"])
            # x-axis ticks at every 24 months. First tick starts from 2001-01-01
            plt.xticks(np.arange(12-i, len(train_dates), 24), rotation=45) 
            plt.show()
            plt.plot(test_dates,y_test,test_dates,y_pred_test)
            plt.title("Probability of Recession occuring: {}-month forecast - test data ({} y-lags, {} x-lags)".format(i,l,k))
            #plt.plot(test_dates,y_test,test_dates,y_pred_classes)
            #plt.title("Recession predictions: {}-month forecast - test data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.legend(["actual","predicted"])
            # x-axis ticks at every 24 months.
            plt.xticks(np.arange(12-i, len(test_dates), 24), rotation=45) 
            plt.show()
            
            # display precision recall curves in-sample and out-of-sample
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train)
            plt.plot(recall_train, precision_train, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(['Probit'])
            plt.title("Precision-recall curve: {}-month forecast - train data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.show()
            
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_test)
            plt.plot(recall_test, precision_test, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(['Probit'])
            plt.title("Precision-recall curve: {}-month forecast - test data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.show()
            
    idx += 1
            


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
from sklearn.metrics import accuracy_score,confusion_matrix,log_loss, classification_report, precision_recall_curve,\
    f1_score, auc, average_precision_score
from statsmodels.discrete.discrete_model import Probit
from Forecasters.TSDatasetGenerator import TSDatasetGenerator

recession_dates = [('1969-12-01', '1970-11-01'),
                   ('1973-11-01', '1975-03-01'),
                   ('1980-01-01', '1980-07-01'),
                   ('1981-07-01', '1982-11-01'),
                   ('1990-07-01', '1991-03-01'),
                   ('2001-03-01', '2001-11-01'),
                   ('2007-12-01', '2009-06-01')]
train_recession_dates = recession_dates[0:5]
test_recession_dates = recession_dates[5:]

def plot_recession_curve(dates, y_pred, data_type, h, l, k):
    formatted_dates = dates.iloc[max(l, k)+h:, ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(formatted_dates, y_pred)
    rec_dates = train_recession_dates if data_type == 'Train' else test_recession_dates
    for start, end in rec_dates:
        if start >= formatted_dates[max(l, k)+h]:
            ax.axvspan(start,
                       end,
                       label="Recession", color="red", alpha=0.3)
        else:
            ax.axvspan(formatted_dates[max(l, k)+h],
                       end,
                       label="Recession", color="red", alpha=0.3)
    ax.set_title("Probability of Recession Occurring: {}-month forecast - {} data ({} y-lags, {} x-lags)".format(h, data_type, l, k))
    ax.legend(["Predicted", "Actual"])
    ax.set_xticks(np.arange(12 - h, len(formatted_dates), 24))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylim([0, 1.05])
    plt.show()
    
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
            #train_dates = train_dates.iloc[max(l,k):-i,]
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
            #test_dates = test_dates.iloc[max(l,k):-i,]
            rec_data_test = rec_test.drop(columns=['DATE'])
            rec_data_test = generator.fit_transform(rec_data_test, target, i, l, k)
            X_test, y_test = rec_data_test.drop(columns=['Target Feature']), rec_data_test['Target Feature']
            X_test = sm.add_constant(X_test)
            y_pred_train = model.predict(X_train)
            y_pred_train_classes = np.where(y_pred_train > 0.5,1,0)
            y_pred_test = model.predict(X_test)
            y_pred_classes = np.where(y_pred_test > 0.5,1,0)
            pd.concat([y_test.to_frame('y_test_%d'%i), y_pred_test.to_frame('y_pred_%d'%i)], axis=1).to_csv('Probit Data/DP_Test_Results %d mth %d y lags %d x lags.csv'%(i,l,k), index=False)
            
            
            #Evaluate in-sample and OOS performance
            cf_reports_train[idx].append(pd.DataFrame(classification_report(y_train, y_pred_train_classes, output_dict=True)))
            cf_reports_test[idx].append(pd.DataFrame(classification_report(y_test, y_pred_classes, output_dict=True)))
            cm = confusion_matrix(y_test,y_pred_classes)
            print("Confusion matrix:")
            print(cm)
            accuracy = accuracy_score(y_test,y_pred_classes)
            print("Accuracy: " + str(accuracy))
            is_lloss = log_loss(np.array(y_train), np.array(y_pred_train_classes))
            print("In-sample Log loss: " + str(is_lloss))
            oos_lloss = log_loss(np.array(y_test), np.array(y_pred_classes))
            print("Out-of-sample Log loss: " + str(oos_lloss))
            accuracies[idx].append(accuracy)
            is_log_losses[idx].append(is_lloss)
            oos_log_losses[idx].append(oos_lloss)
            
            # plot model performance on the training and test sets
            plot_recession_curve(train_dates, y_pred_train_classes, 'Train', i, l, k)
            #plot_recession_curve(train_dates, y_train, 'Train', i, l, k)
            plot_recession_curve(test_dates, y_pred_classes, 'Test', i, l, k)
            #plot_recession_curve(test_dates, y_test, 'Test', i, l, k)
            
            # display precision recall curves in-sample and out-of-sample
            precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_train)
            plt.plot(recall_train, precision_train)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(['Probit'])
            plt.title("Precision-recall curve: {}-month forecast - train data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.figure(figsize=(10, 5))
            plt.show()
            
            precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_test)
            plt.plot(recall_test, precision_test)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(['Probit'])
            plt.title("Precision-recall curve: {}-month forecast - test data ({} y-lags, {} x-lags)".format(i,l,k))
            plt.figure(figsize=(10, 5))
            plt.show()
            
    idx += 1
            


## Model Evaluation for XGBoost ##
## Import libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import seaborn as sns
from Forecasters import TSDatasetGenerator, XGBForecaster
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    plot_precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import log_loss
from datetime import datetime
from matplotlib.dates import date2num

## Load data
# With dates
train_data_date = pd.read_csv('Data/rec_train.csv').drop(columns=['Unnamed: 0'])
test_data_date = pd.read_csv('Data/rec_test.csv').drop(columns=['Unnamed: 0'])
# Without dates
train_data = pd.read_csv('Data/rec_train.csv').drop(columns=['Unnamed: 0', 'DATE'])
test_data = pd.read_csv('Data/rec_test.csv').drop(columns=['Unnamed: 0', 'DATE'])

## Get dates for plotting
train_dates = train_data_date['DATE']
test_dates = test_data_date['DATE']
recession_dates = [('1969-12-01', '1970-11-01'),
                   ('1973-11-01', '1975-03-01'),
                   ('1980-01-01', '1980-07-01'),
                   ('1981-07-01', '1982-11-01'),
                   ('1990-07-01', '1991-03-01'),
                   ('2001-03-01', '2001-11-01'),
                   ('2007-12-01', '2009-06-01')]
train_recession_dates = recession_dates[0:5]
test_recession_dates = recession_dates[5:]


## Auxiliary functions to generate metrics
def genClassificationRep(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


def plot_pr_curve(classifier, X_test, y_test):
    y_score = classifier.predict_proba(X_test)[:, 1]  # Get prediction probs for positive class
    average_precision = average_precision_score(y_test, y_score)
    plt.figure(figsize=(10, 5))
    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    disp.ax_.set_title('Recession prediction: PR Curve with AP = {0:0.2f}'.format(average_precision))
    return disp


def plot_recession_curve(dates, y_pred, data_type, h, l, k):
    formatted_dates = dates.iloc[max(l, k) + h: ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(formatted_dates, y_pred)
    rec_dates = train_recession_dates if data_type == 'Train' else test_recession_dates
    for start, end in rec_dates:
        ax.axvspan(start,
                   end,
                   label="Recession", color="red", alpha=0.3)
    ax.set_title("Probability of Recession Occurring: {}-month forecast - {} data ({} y-lags, {} x-lags)".format(h, data_type, l, k))
    ax.legend(["Predicted", "Actual"])
    ax.set_xticks(np.arange(12 - h, len(formatted_dates), 24))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylim([0, 1.05])
    plt.show()

def fit_model(model, dataset, val_dataset):
    X_train, y_train, X_val, y_val = get_train_test(dataset, val_dataset)
    model.fit(X_train,
              y_train,
              verbose=False,
              early_stopping_rounds=20,
              eval_metric='logloss',
              eval_set=[(X_val, y_val)])
    return model

def get_train_test(dataset, val_dataset):
    X_train = dataset.drop(columns=['Target Feature'])
    y_train = dataset['Target Feature']
    X_val = val_dataset.drop(columns=['Target Feature'])
    y_val = val_dataset['Target Feature']
    return (X_train, y_train, X_val, y_val)

def plt_featimpt(feat_impt, feat_names):
    data = pd.concat([pd.DataFrame(feat_names, columns=['Feature']), pd.DataFrame(feat_impt, columns=['Importance (Gain)'])], axis=1).sort_values(by=['Importance (Gain)'], ascending=False)
    sns.barplot(x='Importance (Gain)', y='Feature', data=data)
    plt.show()

## Instantiate dataset generator
dg = TSDatasetGenerator()

## Best 1 step model ##
# Best params
best_onestep_xg = {'objective': 'binary:logistic',
                   'n_estimators': 100,
                   'max_depth': 2,
                   'colsample_bytree': 0.8,
                   'seed': 42,
                   'missing': None,
                   'importance_type': 'gain',
                   'n_jobs': -1}

xgb_one_step = xgboost.XGBClassifier()
xgb_one_step.set_params(**best_onestep_xg)

# Best dataset (In and out of sample)
one_step_train = dg.fit_transform(train_data, 'Is_Recession', 1, 1, 0)
one_step_test = dg.fit_transform(test_data, 'Is_Recession', 1, 1, 0)
X_train_one, y_train_one, X_test_one, y_test_one = get_train_test(one_step_train, one_step_test)

## MODEL
# Fit model
xgb_one_step = fit_model(xgb_one_step, one_step_train, one_step_test)
# Predict (In sample)
y_pred_one_train = xgb_one_step.predict(X_train_one)

## IN SAMPLE
# Classification Report (In sample)
clf_rep_one_train = genClassificationRep(y_train_one, y_pred_one_train)
# Confusion matrix
confusion_matrix(y_train_one, y_pred_one_train)
# Precision recall graphs (In sample)
pr_curve_one_train = plot_pr_curve(xgb_one_step, X_train_one, y_train_one)
plt.show()
# Feature Importance
plt_featimpt(xgb_one_step.feature_importances_, X_train_one.columns)


## OOS
# Predict (Out of sample)
y_pred_one = xgb_one_step.predict(X_test_one)
# Classification Report (Out of sample)
clf_rep_one = genClassificationRep(y_test_one, y_pred_one)
# Log Loss
log_loss(y_test_one, y_pred_one)
# Confusion matrix
confusion_matrix(y_test_one, y_pred_one)
# Precision recall graphs (Out of sample)
pr_curve_one = plot_pr_curve(xgb_one_step, X_test_one, y_test_one)
plt.show()

## Recession prediction graph (In and Out of sample)
# In Sample
plot_recession_curve(train_dates, y_pred_one_train, 'Train', 1, 1, 0)
plt.show()
# Out of Sample
plot_recession_curve(test_dates, y_pred_one, 'Test', 1, 1, 0)
plt.show()

## Best 3 step model ##
# Best params
best_threestep_xg = {'objective': 'binary:logistic',
                     'n_estimators': 300,
                     'max_depth': 4,
                     'colsample_bytree': 0.8,
                     'seed': 42,
                     'missing': None,
                     'importance_type': 'gain',
                     'n_jobs': -1}

xgb_three_step = xgboost.XGBClassifier()
xgb_three_step.set_params(**best_threestep_xg)

# Best dataset (In and out of sample)
three_step_train = dg.fit_transform(train_data, 'Is_Recession', 3, 2, 0)
three_step_test = dg.fit_transform(test_data, 'Is_Recession', 3, 2, 0)
X_train_three, y_train_three, X_test_three, y_test_three = get_train_test(three_step_train, three_step_test)

## MODEL
# Fit model
xgb_three_step = fit_model(xgb_three_step, three_step_train, three_step_test)
# Predict (In sample)
y_pred_three_train = xgb_three_step.predict(X_train_three)

## IN SAMPLE
# Classification Report (In sample)
clf_rep_three_train = genClassificationRep(y_train_three, y_pred_three_train)
# Confusion matrix
confusion_matrix(y_train_three, y_pred_three_train)
# Precision recall graphs (In sample)
pr_curve_three_train = plot_pr_curve(xgb_three_step, X_train_three, y_train_three)
plt.show()
# Feature Importance
plt_featimpt(xgb_three_step.feature_importances_, X_train_three.columns)

## OOS
# Predict (Out of sample)
y_pred_three = xgb_three_step.predict(X_test_three)
# Classification Report (Out of sample)
clf_rep_three = genClassificationRep(y_test_three, y_pred_three)
# Log Loss
log_loss(y_test_three, y_pred_three)
# Confusion matrix
confusion_matrix(y_test_three, y_pred_three)
# Precision recall graphs (Out of sample)
pr_curve_three = plot_pr_curve(xgb_three_step, X_test_three, y_test_three)
plt.show()

## Recession prediction graph (In and Out of sample)
# In Sample
plot_recession_curve(train_dates, y_pred_three_train, 'Train', 3, 2, 0)
plt.show()
# Out of Sample
plot_recession_curve(test_dates, y_pred_three, 'Test', 3, 2, 0)
plt.show()

## Best 6 step model ##
# Best params
best_sixstep_xg = {'objective': 'binary:logistic',
                   'n_estimators': 100,
                   'max_depth': 4,
                   'colsample_bytree': 0.8,
                   'seed': 42,
                   'missing': None,
                   'importance_type': 'gain',
                   'n_jobs': -1}

xgb_six_step = xgboost.XGBClassifier()
xgb_six_step.set_params(**best_sixstep_xg)

# Best dataset (In and out of sample)
six_step_train = dg.fit_transform(train_data, 'Is_Recession', 6, 3, 0)
six_step_test = dg.fit_transform(test_data, 'Is_Recession', 6, 3, 0)
X_train_six, y_train_six, X_test_six, y_test_six = get_train_test(six_step_train, six_step_test)

## MODEL
# Fit model
xgb_six_step = fit_model(xgb_six_step, six_step_train, six_step_test)
# Predict (In sample)
y_pred_six_train = xgb_six_step.predict(X_train_six)

## IN SAMPLE
# Classification Report (In sample)
clf_rep_six_train = genClassificationRep(y_train_six, y_pred_six_train)
# Confusion matrix
confusion_matrix(y_train_six, y_pred_six_train)
# Precision recall graphs (In sample)
pr_curve_six_train = plot_pr_curve(xgb_six_step, X_train_six, y_train_six)
plt.show()
# Feature Importance
plt_featimpt(xgb_six_step.feature_importances_, X_train_six.columns)

## OOS
# Predict (Out of sample)
y_pred_six = xgb_six_step.predict(X_test_six)
# Classification Report (Out of sample)
clf_rep_six = genClassificationRep(y_test_six, y_pred_six)
# Log Loss
log_loss(y_test_six, y_pred_six)
# Confusion matrix
confusion_matrix(y_test_six, y_pred_six)
# Precision recall graphs (Out of sample)
pr_curve_six = plot_pr_curve(xgb_six_step, X_test_six, y_test_six)
plt.show()

## Recession prediction graph (In and Out of sample)
# In Sample
plot_recession_curve(train_dates, y_pred_six_train, 'Train', 6, 3, 0)
plt.show()
# Out of Sample
plot_recession_curve(test_dates, y_pred_six, 'Test', 6, 3, 0)
plt.show()

## Best 12 step model ##
# Best params
best_twstep_xg = {'objective': 'binary:logistic',
                  'n_estimators': 100,
                  'max_depth': 2,
                  'colsample_bytree': 0.8,
                  'seed': 42,
                  'missing': None,
                  'importance_type': 'gain',
                  'n_jobs': -1}

xgb_tw_step = xgboost.XGBClassifier()
xgb_tw_step.set_params(**best_twstep_xg)

# Best dataset (In and out of sample)
tw_step_train = dg.fit_transform(train_data, 'Is_Recession', 12, 3, 0)
tw_step_test = dg.fit_transform(test_data, 'Is_Recession', 12, 3, 0)
X_train_tw, y_train_tw, X_test_tw, y_test_tw = get_train_test(tw_step_train, tw_step_test)

## MODEL
# Fit model
xgb_tw_step = fit_model(xgb_tw_step, tw_step_train, tw_step_test)
# Predict (In sample)
y_pred_tw_train = xgb_tw_step.predict(X_train_tw)

## IN SAMPLE
# Classification Report (In sample)
clf_rep_tw_train = genClassificationRep(y_train_tw, y_pred_tw_train)
# Confusion matrix
confusion_matrix(y_train_tw, y_pred_tw_train)
# Precision recall graphs (In sample)
pr_curve_tw_train = plot_pr_curve(xgb_tw_step, X_train_tw, y_train_tw)
plt.show()
# Feature Importance
plt_featimpt(xgb_tw_step.feature_importances_, X_train_tw.columns)

## OOS
# Predict (Out of sample)
y_pred_tw = xgb_tw_step.predict(X_test_tw)
# Classification Report (Out of sample)
clf_rep_tw = genClassificationRep(y_test_tw, y_pred_tw)
# Log Loss
log_loss(y_test_tw, y_pred_tw)
# Confusion matrix
confusion_matrix(y_test_tw, y_pred_tw)
# Precision recall graphs (Out of sample)
pr_curve_tw = plot_pr_curve(xgb_tw_step, X_test_tw, y_test_tw)
plt.show()

## Recession prediction graph (In and Out of sample)
# In Sample
plot_recession_curve(train_dates, y_pred_tw_train, 'Train', 12, 3, 0)
plt.show()
# Out of Sample
plot_recession_curve(test_dates, y_pred_tw, 'Test', 12, 3, 0)
plt.show()

## Churn out predictions for GR model
## 1 Step ahead
pd.concat([y_test_one.to_frame('y_test_1'), pd.DataFrame(xgb_one_step.predict_proba(X_test_one)[:, 1], columns=['y_pred_1'])], axis=1).to_csv('XG_results_1.csv', index=False)
## 3 Step ahead
pd.concat([y_test_three.to_frame('y_test_3'), pd.DataFrame(xgb_three_step.predict_proba(X_test_three)[:, 1], columns=['y_pred_3'])], axis=1).to_csv('XG_results_3.csv', index=False)
## 6 Step ahead
pd.concat([y_test_six.to_frame('y_test_6'), pd.DataFrame(xgb_six_step.predict_proba(X_test_six)[:, 1], columns=['y_pred_6'])], axis=1).to_csv('XG_results_6.csv', index=False)
## 12 Step ahead
pd.concat([y_test_tw.to_frame('y_test_12'), pd.DataFrame(xgb_tw_step.predict_proba(X_test_tw)[:, 1], columns=['y_pred_12'])], axis=1).to_csv('XG_results_12.csv', index=False)

classification_report
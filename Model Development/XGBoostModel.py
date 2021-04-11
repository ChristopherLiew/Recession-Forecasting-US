### XGBoost Recession Forecasting Model ###
## Import relevant libraries
import pandas as pd
import numpy as np
from Forecasters import TSDatasetGenerator
from Forecasters.XGBForecaster import XGBForecaster
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report

## Load datasets
train_data = pd.read_csv('Data/rec_train_final.csv').drop(columns=['Unnamed: 0'])
test_data = pd.read_csv('Data/rec_test_final.csv').drop(columns=['Unnamed: 0'])

## Create datasets
dataset_generator = TSDatasetGenerator()

## Auxiliary function to generate Classification Report
def genClassificationRep(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))

## 1 STEP AHEAD FORECAST
# 1) h = 1, k = 2, l = 2 (Kauppi's best pseudo R^2 parameters)
train_1_2_1 = dataset_generator.fit_transform(train_data, 'Is_Recession', h=1, k=2, l=1)
train_1_2_1_X = train_1_2_1.drop(columns=['Target Feature'])
train_1_2_1_y = train_1_2_1['Target Feature']

# XGBoost Grid Search + recursive cross val on lagged data
xg_params_grid = {'objective': ['binary:logistic'],
                  'n_estimators': [100, 300, 500],
                  'max_depth': [2, 4, 6],
                  'seed': [42],
                  'missing': [None],
                  'importance_type': ['gain']}

xg_params = {'objective': 'binary:logistic',
             'seed': 42,
             'missing': None}

xgb = XGBForecaster()
xgb = xgb.fit(train_1_2_1_X, train_1_2_1_y)
model, model_pred, model_truth, cv_result, feat_impt = xgb.train_model_cv(xg_params, n_splits=300)
genClassificationRep(model_truth, model_pred)

# Test set results
test_1_2_1 = dataset_generator.fit_transform(test_data, 'Is_Recession', h=1, k=2, l=1)
test_1_2_1_X = test_1_2_1.drop(columns=['Target Feature'])
test_1_2_1_y = test_1_2_1['Target Feature']

y_pred = model.predict(np.asmatrix(test_1_2_1_X))
genClassificationRep(test_1_2_1_y, y_pred)

## 6 STEP AHEAD FORECAST
# 1) h = 6, k = 2, l = 2 (Kauppi's best pseudo R^2 parameters)
train_6_2_1 = dataset_generator.fit_transform(train_data, 'Is_Recession', h=6, k=2, l=1)
train_6_2_1_X = train_6_2_1.drop(columns=['Target Feature'])
train_6_2_1_y = train_6_2_1['Target Feature']

xgb6 = XGBForecaster()
xgb6 = xgb6.fit(train_6_2_1_X, train_6_2_1_y)
model6, model_pred6, model_truth6, cv_result6, feat_impt6 = xgb6.train_model_cv(xg_params, n_splits=300)
genClassificationRep(model_truth6, model_pred6)

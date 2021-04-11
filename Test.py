import os
import pandas as pd
from Forecasters import TSDatasetGenerator
from Forecasters.RFForecaster import RFForecaster
from Forecasters.XGBForecaster import XGBForecaster

def gather_all_data(data_path):
    """
    Assumes that data is stored like in our GDrive: Data > Folders containing data of different lags > Train & Test.
    NOTE: Folders should only contain csv data.
    :param data_path: Path to data
    :return: Nested dictionary containing all relevant files
    """
    repo = {}
    idx = 0
    for root, dirs, files in os.walk(data_path):
        if idx == 0:
            for directory in dirs:
                repo[directory] = {}
            print('Created dictionaries with names: %s' % (repo.keys()))
            idx += 1
        else:
            dict_key = os.path.basename(root)
            for file in files:
                repo[dict_key][file] = pd.read_csv(os.path.join(root, file))
    return repo

all_data = gather_all_data('./Model Data')

### Training Procedure ###
# 1) Define candidate parameters for GridSearch + Get relevant data
# 2) Fit model on relevant data (Only need to do this once, but class checks if its already generated so no worries)
# 3) Pass in candidate parameters for GridSearch
# 4) Get best model, params, score
# 4*) Get feature importances and perform feature selection
# 5) Use best cross validated and grid searched model to run predictions on the test set
# 6) Get test MSE etc (Model eval proper)

## Generate Lagged Dataset with Forecast Horizon ##
dataset_transformer = TSDatasetGenerator()
test_dataset = pd.read_csv('/Users/MacBookPro15/Desktop/Y3SEM2/EC4308/Project/Model Development/Model Data/rec_data_final/rec_data_final_train.csv')

# Create dataset with 3 lags in X (k = 3), 3 lags in y (l = 3) and a 12 step ahead forecast horizon
dataset_12_3_3 = dataset_transformer.fit_transform(test_dataset, 'Is_Recession', 12, 3, 3)
dataset_12_3_3

## Random Forest Tests ##
# Test run on no lags data
rec_data_train = all_data['rec_data_final']['rec_data_final_train.csv'].dropna()
X_train, y_train = rec_data_train.drop(columns=['DATE', 'Is_Recession']), rec_data_train['Is_Recession']
rec_data_test = all_data['rec_data_final']['rec_data_final_test.csv']
X_test, y_test = rec_data_test.drop(columns=['DATE', 'Is_Recession']), rec_data_test['Is_Recession']

# Set params for RF model
rf_params = {
    'n_estimators': 100,
    'criterion': 'gini',
    'bootstrap': True,
    'max_depth': 10,
    'class_weight': 'balanced_subsample' #Balancing is done on each bootstrapped subsample (n_samples / (n_classes * np.bincount(y)))
}

# Test recursive cross val
forecaster = RFForecaster()
forecaster = forecaster.fit(X_train, y_train)
forecaster.train_model_cv(rf_params, n_splits=50)

# Set params grid for RFBoost model (Remember values in the dictionary must be iterables)
rf_params_grid = {
    'n_estimators': [100, 300, 500, 800, 1000, 1200, 1400],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'max_depth': [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'class_weight': ['balanced_subsample', 'balanced', None]
}

# Test Grid Search with Recursive cross val
forecaster.grid_search_CV(rf_params_grid)
best_rf_model = forecaster.getBestModel()
best_rf_params = forecaster.getBestParams()
best_rf_score = forecaster.getBestScore()
best_rf_feat_impt =forecaster.getFeatureImportance()
rf_gridsearch_log = forecaster.getGridSearchLog()

## XGBoost Tests
# Test run on no lags data
rec_data_train = all_data['rec_data_final']['rec_data_final_train.csv'].dropna()
X_train, y_train = rec_data_train.drop(columns=['DATE', 'Is_Recession']), rec_data_train['Is_Recession']
rec_data_test = all_data['rec_data_final']['rec_data_final_test.csv']
X_test, y_test = rec_data_test.drop(columns=['DATE', 'Is_Recession']), rec_data_test['Is_Recession']

# Set params for XGBoost model
xg_params = {'objective': 'binary:logistic',
             'seed': 42,
             'missing': None}

# Test recursive cross val
forecaster = XGBForecaster()
forecaster = forecaster.fit(X_train, y_train)
forecaster.train_model_cv(xg_params, n_splits=50)

# Set params grid for XGBoost model (Remember values in the dictionary must be iterables)
xg_params_grid = {'objective': ['binary:logistic'],
                  'n_estimators': [100, 300, 500],
                  'max_depth': [2, 4, 6],
                  'seed': [42],
                  'missing': [None],
                  'importance_type': ['gain']}

# Test Grid Search with Recursive cross val
forecaster.grid_search_CV(xg_params_grid)
best_xg_model = forecaster.getBestModel()
best_xg_params = forecaster.getBestParams()
best_xg_score = forecaster.getBestScore()
best_xg_feat_impt =forecaster.getFeatureImportance()
xg_gridsearch_log = forecaster.getGridSearchLog()

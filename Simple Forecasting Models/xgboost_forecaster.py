## Import libraries ##
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_X_y
assert sklearn.__version__ >= '0.24.1'

## Load our Training & Testing Data ##
# Auxiliary function to pull all data
def gather_all_data(data_path):
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

# XGBoost has a default behaviour for missing data (Simply label the missing data as 0)
# XGBoost handles this by clustering these samples with the observations that eventually give us the greatest gain

## Define custom training loop with Recursive Cross Val + GridSearch (TBD) ##
# Handling imbalanced classification with scale_pos_weight
# > XGBoost recommends using the ratio of the number of instances belonging to the negative class to that of the positive class
# Define callbacks: callbacks = [xgb.callback.EarlyStopping(rounds=20, metric_name='logloss', save_best=True)]
# For each combination of the model -> Perform Standardisation -> Perform recursive cross validation (SUPER EXPENSIVE)

class XGBForecaster(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with Grid Search and Recursive Cross Validation capabilities.
    """

    def __init__(self):
        # Class Params
        self.X_train_ = pd.DataFrame()
        self.y_train_ = pd.DataFrame()
        self.XBGModelParams = {}
        self.FeatureNames = []
        # Cross Val Stuff
        self.CVPredictions = []
        self.CVTrue = []
        self.CrossValData = []
        self.FeatureImportanceRepo = []
        self.BaseModel = xgb.XGBClassifier()
        # Grid Search Stuff (TBD)
        self.GridSearchParams = {}
        self.BestModel = xgb.XGBClassifier()
        self.BestScore = np.inf
        self.BestParams = {}

    def get_CrossValData(self):
        return self.CrossValData

    def get_cv_score(self):
        try:
            return log_loss(np.array(self.CVTrue), np.array(self.CVPredictions))
        except ValueError:
            print('Predictions are all 0 or 1 :(')

    def get_cv_preds(self):
        return self.CVPredictions

    def get_cv_truelabels(self):
        return self.CVTrue

    def get_all_feature_importances(self):
        return self.FeatureImportanceRepo

    def get_cv_feature_importance(self):
        return pd.DataFrame(np.mean(self.FeatureImportanceRepo, axis=0).reshape(1, len(self.FeatureNames)),
                            columns=self.FeatureNames).melt().sort_values(by='value')

    def get_base_model(self):
        return self.BaseModel

    def compute_scale_pos_weight(self, y_train_labels):
        """
        Computes the ratio of the number of instances belonging to the negative class to that of the positive class.
        Assumes that the POSITIVE CLASS = 1 and the NEGATIVE CLASS = 0.
        :param y_train_labels: Series or matrix of binary training labels.
        :return: Recommended scale pos weight value.
        """

        num_pos_instances = sum(y_train_labels)
        num_neg_instances = len(y_train_labels) - num_pos_instances
        return (num_neg_instances / num_pos_instances)

    def generate_recursive_folds(self, n_splits):
        """
        Splits our training data recursively into batches for training and 1 step ahead forecasting.
        :return: None. Updates the XGBForecaster's CrossValData attribute.
        """
        time_series_split = TimeSeriesSplit(n_splits, max_train_size=None, test_size=1)
        for train_idx, val_idx in time_series_split.split(self.X_train_):
            X_train_folds, X_val_folds = self.X_train_[train_idx, :], self.X_train_[val_idx, :]
            y_train_folds, y_val_folds = self.y_train_[train_idx], self.y_train_[val_idx]
            self.CrossValData.append({'X_train': X_train_folds,
                                      'y_train': y_train_folds,
                                      'X_val': X_val_folds,
                                      'y_val': y_val_folds})

    def fit(self, X, y):
        self.FeatureNames.extend(X.columns)
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def train_model_cv(self, xg_params_grid, n_splits=10):
        """
        Trains our model using recursive cross validation.
        :param xg_params_grid: If None, default XGB params for classification will be used.
        :param n_splits: Default = 10.
        :return: None. Use getter to obtain best model.
        """
        # Check if fitted
        check_is_fitted(self)

        # Create recursive cross val data
        self.generate_recursive_folds(n_splits)
        print("Recursive cross val data generated, number of split: %d" % n_splits)

        # Loop over each training fold
        for dataset in tqdm(self.CrossValData):
            # Get data
            X_train, y_train = dataset['X_train'], dataset['y_train']
            X_val, y_val = dataset['X_val'], dataset['y_val']
            # Calculate scale_pos_weights
            scale_pos_weights = self.compute_scale_pos_weight(y_train)
            # Create model
            xg_params_grid_iter = xg_params_grid
            xg_params_grid_iter['scale_pos_weight'] = scale_pos_weights
            xgb_clf = xgb.XGBClassifier()
            xgb_clf.set_params(**xg_params_grid_iter)
            # Fit model
            xgb_clf.fit(X_train,
                        y_train,
                        verbose=False,
                        early_stopping_rounds=20,
                        eval_metric='logloss',
                        eval_set=[(X_val, y_val)])
            # Get prediction
            y_pred = xgb_clf.predict(X_val)
            # Store log loss and feature importance
            self.CVPredictions.append(y_pred.item())
            self.CVTrue.append(y_val.item())
            self.FeatureImportanceRepo.append(xgb_clf.feature_importances_)
            # Store latest model
            self.BaseModel = xgb_clf

    def predict(self, X):
        check_is_fitted(self)
        return self.BaseModel.predict(X)

# Test run on no lags data
rec_data_train = all_data['rec_data_final']['rec_data_final_train.csv'].dropna()
X_train, y_train = rec_data_train.drop(columns=['DATE', 'Is_Recession']), rec_data_train['Is_Recession']
rec_data_test = all_data['rec_data_final']['rec_data_final_test.csv']
X_test, y_test = rec_data_test.drop(columns=['DATE', 'Is_Recession']), rec_data_test['Is_Recession']

# Set params for XGBoost model
xg_params_grid = {'objective': 'binary:logistic',
                  'seed': 42,
                  'missing': None}

## TUNING FEAT IMPORTANCE ITER 1 ##
# Instantiate and train model
forecaster = XGBForecaster()
forecaster = forecaster.fit(X_train, y_train)
forecaster.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster.get_cv_score()

# Get log loss score
forecaster_cv_pred = forecaster.get_cv_preds()
forecaster_cv_true = forecaster.get_cv_truelabels()
forecaster_score = forecaster.get_cv_score()

# Get feature importances
feat_impt_log = forecaster.get_all_feature_importances()
feat_impt = forecaster.get_cv_feature_importance()
sum(feat_impt.value)

# Select features with contribution >= 2%
select_features_1 = feat_impt[feat_impt.value >= 0.02]['variable'].to_list()
# Subset features (11 features)
X_train_subset, X_test_subset = X_train.loc[:, select_features_1], X_test.loc[:, select_features_1]

## TUNING FEAT IMPORTANCE ITER 2 ##
# Train new model
forecaster2 = XGBForecaster()
forecaster2 = forecaster2.fit(X_train_subset, y_train)
forecaster2.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster2.get_cv_score()

# Get feature importance
feat_impt = forecaster2.get_cv_feature_importance()
# Select features with contribution >= 5%
select_features_2 = feat_impt[feat_impt.value >= 0.05]['variable'].to_list()
# Subset features (11 features)
X_train_subset2, X_test_subset2 = X_train.loc[:, select_features_2], X_test.loc[:, select_features_2]

## TUNING FEAT IMPORTANCE ITER 3 ##
# Train new model
forecaster3 = XGBForecaster()
forecaster3 = forecaster3.fit(X_train_subset2, y_train)
forecaster3.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster3.get_cv_score()

# Get feature importance
feat_impt = forecaster2.get_cv_feature_importance()
# Select features with contribution >= 9%
select_features_3 = feat_impt[feat_impt.value >= 0.09]['variable'].to_list()
# Subset features (11 features)
X_train_subset3, X_test_subset3 = X_train.loc[:, select_features_3], X_test.loc[:, select_features_3]

## TUNING FEAT IMPORTANCE ITER 4 ##
# Train new model
forecaster4 = XGBForecaster()
forecaster4 = forecaster4.fit(X_train_subset3, y_train)
forecaster4.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster4.get_cv_score()

# Get feature importance
feat_impt = forecaster4.get_cv_feature_importance()

## TUNING FEAT IMPORTANCE ITER 5 - SINGLE FEATURES ##
# SPREAD
forecaster_spread = XGBForecaster()
forecaster_spread = forecaster_spread.fit(X_train.loc[:, ['10Y3MTH_SPREAD']], y_train)
forecaster_spread.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster_spread.get_cv_score()

# FEDFUNDS_ROLMEAN
forecaster_FFROLMEAN = XGBForecaster()
forecaster_FFROLMEAN = forecaster_FFROLMEAN.fit(X_train.loc[:, ['FEDFUNDS_ROLMEAN3']], y_train)
forecaster_FFROLMEAN.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster_FFROLMEAN.get_cv_score()

# PAYEMS_3MTHCHANGE
forecaster_PAYEMSCHANGE = XGBForecaster()
forecaster_PAYEMSCHANGE = forecaster_PAYEMSCHANGE.fit(X_train.loc[:, ['PAYEMS_3MTHCHANGE']], y_train)
forecaster_PAYEMSCHANGE.train_model_cv(xg_params_grid=xg_params_grid, n_splits=50)
forecaster_PAYEMSCHANGE.get_cv_score()

## TBD ##
# 1) Grid search
# 2) Aggregation and prettifying of Feature Importances
# 3) Feature selection and building optimal model

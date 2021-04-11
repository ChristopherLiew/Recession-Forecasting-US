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
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_X_y
assert sklearn.__version__ >= '0.24.1'

## Load our Training & Testing Data ##
# Auxiliary function to pull all data
# def gather_all_data(data_path):
#     repo = {}
#     idx = 0
#     for root, dirs, files in os.walk(data_path):
#         if idx == 0:
#             for directory in dirs:
#                 repo[directory] = {}
#             print('Created dictionaries with names: %s' % (repo.keys()))
#             idx += 1
#         else:
#             dict_key = os.path.basename(root)
#             for file in files:
#                 repo[dict_key][file] = pd.read_csv(os.path.join(root, file))
#     return repo
#
# all_data = gather_all_data('./Model Data')

## Handling Missing Data in XGBoost
# XGBoost has a default behaviour for missing data (Simply label the missing data as 0)
# XGBoost handles this by clustering these samples with the observations that eventually give us the greatest gain

## Define custom training loop with Recursive Cross Val + GridSearch (TBD) ##
# > Handling imbalanced classification with scale_pos_weight:
#   XGBoost recommends using the ratio of the number of instances belonging to the negative class to that of the positive class
# > Define callbacks:
#   callbacks = [xgb.callback.EarlyStopping(rounds=20, metric_name='logloss', save_best=True)]

class XGBForecaster(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier with Grid Search and Recursive Cross Validation capabilities.
    """

    def __init__(self):
        # Class Params
        self.X_train_ = pd.DataFrame()
        self.y_train_ = pd.DataFrame()
        self.FeatureNames = []
        # Cross Val Stuff
        self.CrossValData = []
        self.BaseModel = xgb.XGBClassifier()
        # Grid Search Stuff
        self.GridSearchParams = {}
        self.BestModel = None
        self.BestScore = np.inf
        self.BestParams = {}
        self.GridSearchLogger = []
        self.BestModelFeatImportance = None

    def get_CrossValData(self):
        return self.CrossValData

    def get_cv_score(self, y_true, y_pred):
        try:
            return log_loss(np.array(y_true), np.array(y_pred))
        except ValueError:
            print('Predictions are all 0 or 1 :(')
            return np.nan

    def get_cv_feature_importance(self, feature_importances):
        return pd.DataFrame(np.mean(feature_importances, axis=0).reshape(1, len(self.FeatureNames)),
                            columns=self.FeatureNames).melt().sort_values(by='value')

    def get_base_model(self):
        return self.BaseModel

    def getBestModel(self):
        if None:
            print("Please run Grid Search CV to obtain a best model")
            return np.nan
        else:
            return self.BestModel

    def getBestParams(self):
        if not self.BestParams:
            print("Please run Grid Search CV to obtain best params given candidate values")
            return np.nan
        else:
            return self.BestParams

    def getBestScore(self):
        """
        Returns the best score. In our case this is set to be the log loss or negative log likelihood.
        :return: Lowest negative log likelihood achieved after Grid Search.
        """
        if self.BestScore != np.inf:
            return self.BestScore
        else:
            print("Please run Grid Search CV to obtain best score")
            return self.BestScore

    def getFeatureImportance(self):
        """
        Get best model's aggregate feature importance after recursive cross validation.
        :return: Best model's feature importance (Gain)
        """
        if self.BestModelFeatImportance is None:
            print("Please run Grid Search CV to obtain the best model's feature importances")
            return np.nan
        else:
            return self.BestModelFeatImportance

    def getGridSearchLog(self):
        return self.GridSearchLogger

    def compute_scale_pos_weight(self, y_train_labels):
        """
        Computes the ratio of the number of instances belonging to the negative class to that of the positive class.
        Assumes that the POSITIVE CLASS = 1 and the NEGATIVE CLASS = 0.
        :param y_train_labels: Series or matrix of binary training labels.
        :return: Recommended scale pos weight value.
        """
        num_pos_instances = sum(y_train_labels)
        num_neg_instances = len(y_train_labels) - num_pos_instances
        return num_neg_instances / num_pos_instances

    def fit(self, X, y):
        self.FeatureNames.extend(X.columns)
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        return self

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

    def train_model_cv(self, xg_params, n_splits=10):
        """
        Trains our model using recursive cross validation.
        :param xg_params: XGBoost classifier parameters.
        :param n_splits: Default = 10.
        :return: Model, Model Predictions, Model Log Loss and Feature importances
        """
        # Placeholders for predictions
        model_predictions = []
        model_truth = []
        model_feat_impt = []

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
            xg_params_grid_iter = xg_params
            xg_params_grid_iter['scale_pos_weight'] = scale_pos_weights
            # Create model
            self.BaseModel.set_params(**xg_params_grid_iter)
            # Fit model
            self.BaseModel.fit(X_train,
                               y_train,
                               verbose=False,
                               early_stopping_rounds=20,
                               eval_metric='logloss',
                               eval_set=[(X_val, y_val)])

            # Get and store prediction
            model_predictions.append(self.BaseModel.predict(X_val).item())
            # Store true label
            model_truth.append(y_val.item())
            # Store feature importance
            model_feat_impt.append(self.BaseModel.feature_importances_)

        # Compute model log loss
        model_log_loss = self.get_cv_score(model_truth, model_predictions)
        # Compute aggregate feature importance
        model_agg_feat_impt = self.get_cv_feature_importance(model_feat_impt)

        return self.BaseModel, model_predictions, model_truth, model_log_loss, model_agg_feat_impt

    def generate_grid(self, params_grid):
        """
        Generates all combinations of parameters for Grid Search.
        :param params_grid: Parameters and their respective candidate values.
        :return: List of all possible model hyper-parameter combinations given the grid.
        """
        return ParameterGrid(params_grid)

    def grid_search_CV(self, params_grid, n_splits=50):
        # Generate all combinations
        self.GridSearchParams = params_grid
        generated_grid = self.generate_grid(self.GridSearchParams)

        for g in generated_grid:
            model, model_predictions, model_truth, model_log_loss, model_agg_feat_impt = self.train_model_cv(g, n_splits)
            # Save if best performer
            if model_log_loss < self.BestScore:
                self.BestScore = model_log_loss
                self.BestParams = g
                self.BestModel = model
                self.BestModelFeatImportance = model_agg_feat_impt

            # Store in log
            log = {'Score': model_log_loss, 'Params': g, 'Feature importance (Gain)': model_agg_feat_impt}
            self.GridSearchLogger.append(log)

    def predict(self, X):
        check_is_fitted(self)
        if self.BestModel is None:
            print('Using base model with default parameters')
        else:
            print('Using grid search optimised model')
            return self.BestModel.predict(X)

## Tests
# Test run on no lags data
# rec_data_train = all_data['rec_data_final']['rec_data_final_train.csv'].dropna()
# X_train, y_train = rec_data_train.drop(columns=['DATE', 'Is_Recession']), rec_data_train['Is_Recession']
# rec_data_test = all_data['rec_data_final']['rec_data_final_test.csv']
# X_test, y_test = rec_data_test.drop(columns=['DATE', 'Is_Recession']), rec_data_test['Is_Recession']
#
# # Set params for XGBoost model
# xg_params = {'objective': 'binary:logistic',
#              'seed': 42,
#              'missing': None}
#
# # Test recursive cross val
# forecaster = XGBForecaster()
# forecaster = forecaster.fit(X_train, y_train)
# forecaster.train_model_cv(xg_params, n_splits=50)
#
# # Set params grid for XGBoost model (Remember values in the dictionary must be iterables)
# xg_params_grid = {'objective': ['binary:logistic'],
#                   'n_estimators': [100, 300, 500],
#                   'max_depth': [2, 4, 6],
#                   'seed': [42],
#                   'missing': [None],
#                   'importance_type': ['gain']}
#
# # Test Grid Search with Recursive cross val
# forecaster.grid_search_CV(xg_params_grid)
# best_xg_model = forecaster.getBestModel()
# best_xg_params = forecaster.getBestParams()
# best_xg_score = forecaster.getBestScore()
# best_xg_feat_impt =forecaster.getFeatureImportance()
# xg_gridsearch_log = forecaster.getGridSearchLog()

### Training Procedure ###
# 1) Define candidate parameters for GridSearch + Get relevant data
# 2) Fit model on relevant data
# 3) Pass in candidate parameters for GridSearch
# 4) Get best model, params, score
# 4*) Get feature importances and perform feature selection
# 5) Use best cross validated and grid searched model to run predictions on the test set
# 6) Get test MSE etc (Model eval proper)

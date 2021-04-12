### XGBoost Recession Forecasting Model ###
## Import relevant libraries
import pandas as pd
import numpy as np
from Forecasters import TSDatasetGenerator, RFForecaster
from Forecasters.XGBForecaster import XGBForecaster
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report

## Load datasets
# Without DATE col
train_data = pd.read_csv('Data/rec_train.csv').drop(columns=['Unnamed: 0', 'DATE'])
test_data = pd.read_csv('Data/rec_test.csv').drop(columns=['Unnamed: 0', 'DATE'])

# With DATE col (For sanity check)
train_data_date = pd.read_csv('Data/rec_train.csv').drop(columns=['Unnamed: 0'])
test_data_date = pd.read_csv('Data/rec_test.csv').drop(columns=['Unnamed: 0'])


## Auxiliary function to generate Classification Report
def genClassificationRep(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


## Grid Search Candidate Parameters
xg_params_grid = {'objective': ['binary:logistic'],
                  'n_estimators': [100, 300],
                  'max_depth': [2, 4],
                  # Controls for overfitting (build shallow trees, Gamma can be considered as well)
                  'colsample_bytree': [0.8, 1.0],  # Adds randomness for noise robustness
                  'seed': [42],
                  'missing': [None],  # None as we do not have any missing data
                  'importance_type': ['gain']}


## Auxiliary function to train models and log results for various combinations of h, k and l
def runBehemoth(grid_params, data, target_feature, n_splits, model='XGB', h=1, k_range=[], l_range=[]):
    results = []
    label_index = []

    # Instantiate dataset generator
    dataset_generator = TSDatasetGenerator()

    # Loop over all combinations of k and l
    for k in k_range:
        for l in l_range:
            # Instantiate model
            if model == 'XGB':
                selected_model = XGBForecaster()
            elif model == 'RF':
                selected_model = RFForecaster()
            else:
                print('Please select a valid model: RF or XGB')
                return None

            # Label for output (h, k, l)
            res_label = "(h=%d, k=%d, l=%d)" % (h, k, l)
            print("Running grid search for the dataset with params: " + res_label)
            # Create datasets
            train = dataset_generator.fit_transform(data, target_feature, h, k, l)
            train_X = train.drop(columns=['Target Feature'])
            train_y = train['Target Feature']

            # Fit model
            fitted_model = selected_model.fit(train_X, train_y)
            fitted_model.grid_search_CV(grid_params, n_splits=n_splits)

            # Log results
            result = {}
            result['Best score (log-loss)'] = fitted_model.getBestScore()
            result['Best params'] = fitted_model.getBestParams()
            result['Best model'] = fitted_model.getBestModel()
            result['Feature Importance'] = fitted_model.getFeatureImportance()
            results.append(result)
            label_index.append(res_label)

    labels_index = pd.DataFrame(label_index, columns=['Dataset'])
    results_final = pd.DataFrame(results)
    collected_results = pd.concat([labels_index, results_final], axis=1)
    return collected_results

## 1 Step Ahead Forecast
one_step_results = runBehemoth(xg_params_grid,
                               train_data,
                               'Is_Recession',
                               300,
                               k_range=[1, 2, 3, 4, 5, 6],
                               l_range=[1, 2, 3, 4, 5, 6])
## Write out results
one_step_results.to_csv('one_step_xg_results.csv', index=False)

## 3 Step Ahead Forecast
three_step_results = runBehemoth(xg_params_grid,
                                 train_data,
                                 'Is_Recession',
                                 300,
                                 k_range=[1, 2, 3, 4, 5, 6],
                                 l_range=[1, 2, 3, 4, 5, 6])

## Write out results
three_step_results.to_csv('three_step_xg_results.csv', index=False)

## 6 Step Ahead Forecast
six_step_results = runBehemoth(xg_params_grid,
                               train_data,
                               'Is_Recession',
                               300,
                               k_range=[1, 2, 3, 4, 5, 6],
                               l_range=[1, 2, 3, 4, 5, 6])

## Write out results
six_step_results.to_csv('six_step_xg_results.csv', index=False)

## 12 Step Ahead Forecast
twelve_step_results = runBehemoth(xg_params_grid,
                                  train_data,
                                  'Is_Recession',
                                  300,
                                  k_range=[1, 2, 3, 4, 5, 6],
                                  l_range=[1, 2, 3, 4, 5, 6])

## Write out results
twelve_step_results.to_csv('twelve_step_xg_results.csv', index=False)
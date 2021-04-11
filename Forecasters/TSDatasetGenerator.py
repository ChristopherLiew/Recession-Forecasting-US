import pandas as pd
from sklearn.base import TransformerMixin

class TSDatasetGenerator(TransformerMixin):
    """
    Takes in a time series dataset and generates the relevant lags of X and y.
    """

    def __init__(self):
        self.BaseDataset = pd.DataFrame()
        self.TransformedDataset = pd.DataFrame()
        self.h = 0
        self.l = 0
        self.k = 0
        self.TargetFeature = ""

    def getInfo(self):
        return "Dataset generator will create a Panel dataset with\n" \
               "Forecast horizon: %d\n" \
               "Target variable lags: %d\n" \
               "Predictor variable lags: %d\n" % (self.h, self.l, self.k)

    def fit_transform(self, Dataset, TargetFeature, h, l, k):
        """
        Fits and transform the given dataset. Note: All null observations will be dropped.
        :param X: Dataset including both predictors and the target feature.
        :param TargetFeature: Name of the target feature.
        :param h: Forecast horizon
        :param l: Lag order of the target feature.
        :param k: Lag order of the predictors.
        :return: Dataset with lags incorporated.
        """
        # Handle any nulls
        print("Dropping any existing invalid observations")

        self.BaseDataset = Dataset.dropna(axis=0)
        self.h = h
        self.l = l
        self.k = k

        y = self.BaseDataset.loc[:, TargetFeature]
        X = self.BaseDataset.drop(columns=[TargetFeature])
        Store = self.BaseDataset.copy()

        # Create lags for Target feature
        for i in range(1, self.l + 1):
            Store[TargetFeature + '_T_%d' % i] = y.shift(i)

        # Create lags for Predictors
        for j in range(1, self.k + 1):
            new_col_names = [old_name + '_T_%d' % j for old_name in X.columns]
            lagged_X = X.shift(j)
            lagged_X.columns = new_col_names
            Store = pd.concat([Store, lagged_X], axis=1)

        # Generate Horizon
        DatasetWithHorizon = pd.concat([y.shift(-h), Store], axis=1)
        self.TransformedDataset = DatasetWithHorizon.dropna(axis=0)

        # Return dataset
        return self.TransformedDataset


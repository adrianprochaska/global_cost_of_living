import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer only uses categorical columns of the handed data.
    The continent column is dropped if drop_continents is set to True.
    """
    def __init__(self, b_drop_continent = False):
        self.b_drop_continent = b_drop_continent
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # drop numericals
        X = self.drop_numericals(X)

        # drop continents
        X = self.drop_continent(X)

        self.columns = X.columns

        return X

    def get_feature_names_out(self, X):
        return self.columns.tolist()    

    def drop_numericals(self, X):
        """
        Drop all columns except categoricals.
        """
        X = X.select_dtypes(include=['object'])

        return X

    def drop_continent(self, X):
        """
        Drops the continent column, if the flag b_drop_continent is True.
        """
        drop_col = 'continent'
        cols = X.columns

        if self.b_drop_continent and drop_col in cols:
            X = X.drop(labels=drop_col, axis=1)
        # end if

        return X


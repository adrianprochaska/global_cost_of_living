import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer only uses numerical columns of the handed data.
    It drops columns based on a threshold for na values.
    The remaining na values are imputed using the selected impute method.
    """
    def __init__(self, impute_method='mean', max_na_share=0):
        self.impute_method = impute_method
        self.max_na_share = max_na_share
        return 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # drop categorical columns 
        X = self.drop_categoricals(X)

        # drop na columns
        X = self.drop_na_cols(X)

        # impute data
        X = self.impute_data(X)
       
        self.columns = X.columns

        return X

    def get_feature_names_out(self, X):
        return self.columns.tolist()    

    def drop_categoricals(self, X):
        """
        Drop categorical columns.
        """
        X = X.select_dtypes(exclude=['object'])

        return X

    def drop_na_cols(self, X):
        """ 
        Function drops columns with a high share of na values.
        """
        high_na_cols = X.columns[(1-X.count()/X.shape[0])>self.max_na_share]

        if len(high_na_cols)>0:
            X = X.drop(labels=high_na_cols, axis=1)
        # end if

        return X

    def impute_data(self, X):
        """
        Impute all the na data with mean or median.
        """
        col_with_na = X.columns[X.count() < X.shape[0]]

        for col in col_with_na:
            impute_fun = eval('X.loc[:,col].' + self.impute_method)
            X.loc[:,col] = X.loc[:,col].fillna(impute_fun())

        return X


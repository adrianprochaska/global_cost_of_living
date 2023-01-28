import numpy as np
import pandas as pd

class DataHandling: 
    """
    Class to do some data handling:
    - dropping columns with a low share of non-nan values

"""
    def __init__(self, df, target_variable:str, max_na_share, drop_cols_l:list = [], one_hot_cols_l:list = [], impute_sel:str='mean') -> None:
        self._orig_data = df
        self.data = pd.DataFrame(df)
        self.target_variable = target_variable
        self.max_na_share = max_na_share
        self.drop_cols_l = drop_cols_l
        self.one_hot_cols_l = one_hot_cols_l
        self.impute_sel = impute_sel
    
        self.drop_cols()
        self.drop_target_rows()
        self.one_hot_encoding()
        self.drop_cat_cols()
        self.drop_na_cols()
        self.impute_data()

        pass
        
    def drop_cols(self):
        """ 
        Function drops columns mentioned in self.
        drop_cols_l.Uses self.data and overwrites it with the new dataframe
        """
        new_df = self.data
        for col in self.drop_cols_l:
            try:
                new_df = new_df.drop(labels=col, axis=1)
            except:
                print(
                    'Column {} not found and therefore not dropped.'.format(col)
                )

        self.data = new_df
        return new_df

    def one_hot_encoding(self):
        """
        Function executes one hot encoding on columns in self.one_hot_cols_l.
        Uses self.data and overwrites it with the new dataframe
        """
        new_df = self.data
        for col in self.one_hot_cols_l:
            try:
                df_dummies = pd.get_dummies(new_df[col])
                new_df = new_df.drop(labels=col, axis=1)
                new_df = pd.concat([new_df, df_dummies], axis=1)
                print_str = 'Column {} one-hot-encoded.'.format(col)
            except KeyError:
                print_str = 'Column {} not found and therefore not one-hot-encoded.'.format(col)
            # end try
            
            print(print_str)

        self.data = new_df

        return new_df

    def drop_na_cols(self):
        """ 
        Function drops columns with a high share of na values.
        Uses self.data and overwrites it with the new dataframe
        """
        new_df = self.data
        high_na_cols = new_df.columns[(1-new_df.count()/new_df.shape[0])>self.max_na_share]

        if len(high_na_cols)>0:
            new_df = new_df.drop(labels=high_na_cols, axis=1)
        # end if

        print('{} columns dropped due to high na share.'.format(len(high_na_cols)))

        self.data = new_df
        return new_df

    def impute_data(self):
        """
        Impute all the na data with mean or median.
        """
        new_df = self.data
        col_with_na = new_df.columns[new_df.count() < new_df.shape[0]]


        for col in col_with_na:
            impute_fun = eval('new_df.loc[:,col].' + self.impute_sel)
            new_df.loc[:,col] = new_df.loc[:,col].fillna(impute_fun())

        self.data = new_df
        return new_df

    def drop_cat_cols(self):
        """
        Drop categorical columns and output information.
        """
        new_df = self.data

        cols_categorical = list(new_df.columns[new_df.dtypes == 'object'])
        print_str_fun = 'Column {} contains categoricals and is therefore dropped.'.format
        print_str = [print_str_fun(col) for col in cols_categorical]

        if len(print_str)>0:
            print('\n'.join(print_str))
        # end if

        new_df = new_df.select_dtypes(exclude=['object'])

        self.data = new_df
        return new_df

    def drop_target_rows(self):
        new_df=self.data

        new_df = new_df.dropna(subset=self.target_variable, axis=0)

        self.data = new_df
        return new_df

    def get_X_and_y(self):
        X = self.data.drop(labels=self.target_variable, axis=1)
        y = self.data[self.target_variable]

        return X, y

    def __repr__(self) -> str:
        # TODO
        print_str = 'Dataset has {} datapoints and {} columns.'.format(
            self.data.shape[0],
            self.data.shape[1]
        )
        return print_str



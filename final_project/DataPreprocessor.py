
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor(object):

    def __init__(self, dataframe):
        self.df = dataframe

        
    def runTestSetPreprocessor(self):
        self.removeWhitespaceFromData()
        self.replaceMissingValuesWithNaN()
        self.replaceSomeFields()
        self.replaceOrRemoveNaN()

        #balanced_data = self.balanceDataframe()
        numeric_df = self.categoricalValuesToNumbers(self.df)
        normalized_df = self.normalizeTheData()

        return normalized_df        
        
    def runProjectPreprocessor(self):

        self.removeWhitespaceFromData()
        self.replaceMissingValuesWithNaN()
        self.replaceSomeFields()
       
        self.replaceOrRemoveNaN()

        balanced_data = self.balanceDataframe()
        numeric_df = self.categoricalValuesToNumbers()
        normalized_df = self.normalizeTheData()

        return normalized_df

    def removeWhitespaceFromData(self):

        self.df.replace(' <=50K', '<=50K', inplace=True)
        self.df.replace(' >50K', '>50K', inplace=True)
        
        self.df['income'] = self.df['income'].str.strip()


        return self

    def replaceMissingValuesWithNaN(self):
        self.df.replace(' ?', np.nan, inplace=True)
   
        return self

    def replaceOrRemoveNaN(self):
        self.df = self.df[self.df['workclass'].notnull()]
        self.df = self.df[self.df['occupation'].notnull()]
        self.df = self.df[self.df['native-country'].notnull()]

           
        return self
    
    def replaceSomeFields(self):
        self.df.replace([' Divorced', ' Married-AF-spouse', 
              ' Married-civ-spouse', ' Married-spouse-absent', 
              ' Never-married',' Separated',' Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)
        
        self.df.replace([' Self-emp-not-inc',' Self-emp-inc',' Federal-gov',' State-gov',' Local-gov'],
               [' Self-emp', ' Self-emp', ' Gov', ' Gov', ' Gov',],inplace=True)
        
        del self.df['capital-gain']
        del self.df['capital-loss']

    def balanceDataframe(self):
        df = self.df
        below_50 = df[df['income'] == '<=50K']
        above_50 = df[df['income'] == '>50K']
        frames = [below_50.head(10000), above_50]
        df_balanced = pd.concat(frames)

        self.df_balanced = df_balanced
              
    
        return df_balanced

    def categoricalValuesToNumbers(self, data=None):
        
        if data is None:
            df_balanced = self.df_balanced
        else:
            df_balanced = self.df
            
        gender_split = pd.get_dummies(df_balanced['gender'])
        occupation_split = pd.get_dummies(df_balanced['occupation'])
        rel_split = pd.get_dummies(df_balanced['relationship'])
        race_split = pd.get_dummies(df_balanced['race'])
        work_split = pd.get_dummies(df_balanced['workclass'])
        country_split = pd.get_dummies(df_balanced['native-country'])
        marital_status = pd.get_dummies(df_balanced['marital-status'])
        
        
        df_balanced_dummies = pd.concat([df_balanced,gender_split, occupation_split, rel_split,race_split, work_split, country_split,marital_status], axis=1)
        
        col_names = list(df_balanced_dummies.columns)
        col_names_strip = [a.strip() for a in col_names]
        
        df_balanced_dummies.columns = col_names_strip
        
        
        if data is None:
            df_balanced_dummies['income'] = df_balanced_dummies['income'].map({'<=50K':0, '>50K':1})
        else:
            df_balanced_dummies['income'] = df_balanced_dummies['income'].map({'<=50K.':0, '>50K.':1})
        df_balanced_dummies = df_balanced_dummies.select_dtypes(['number'])

        self.df_balanced_dummies = df_balanced_dummies

        return df_balanced_dummies

    def normalizeTheData(self):
        
        mms = MinMaxScaler()

        df_balanced_dummies = self.df_balanced_dummies

        cols_to_norm = ['age','fnlwgt', 'hours-per-week', 'educational-num']
        df_balanced_dummies[cols_to_norm] = df_balanced_dummies[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.df_norm = df_balanced_dummies
  
        return df_balanced_dummies


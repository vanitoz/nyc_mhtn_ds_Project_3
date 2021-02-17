import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
import re


def categor_var_analysis(df, features):
    """
    Features - list of categorical features
    df - data frame where features located
    """
    print('Discrete Variables')
    
    for feature in features:
        print(df.groupby('{}'.format(feature))['{}'.format(feature)].count())
        print('--'*40)
        

def numeric_var_analysis(df, features):
    """
    Features - list of numerical features
    df - data frame where features located
    """
    print('Continuous Variables')
    print(df[features].describe().transpose())
    


    
    


def to_dummies(df, features):
    """
    Generating dummy variables for features
    parama: df
            features - list of features
    """
    d_df = []
    for f in features:
        d_df.append(pd.get_dummies(df[f], prefix='{}'.format(str(f)[:]), drop_first=True))
    #import pdb;pdb.set_trace()
    df = df.drop(features, axis = 1)
    df = pd.concat([df] + d_df ,axis=1)
    
    return df

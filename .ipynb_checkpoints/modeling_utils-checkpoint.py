import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
import re




def to_dummies(df, features):
    """
    Generating dummy variables for features
    parama: df
            features - list of features
    """
    d_df = []
    for f in features:
        d_df.append(pd.get_dummies(df[f], prefix='{}'.format(str(f)[:5]), drop_first=True))
    #import pdb;pdb.set_trace()
    df = df.drop(features, axis = 1)
    df = pd.concat([df] + d_df ,axis=1)
    
    return df
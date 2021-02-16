import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
import re




def compliance(row):
    """
    Checking if payment was made 
    before judgment_date or after 
    params : 
            row - each row in df
    return 0 if payment made late
    return 1 if payment made on time
    
    """
    if row['payment_date'] > row['judgment_date']:
        
        return 0 
    else:
        return 1
    

    
def extract_lat_crime(x):
    try:
        if '\n' in x:
            x_ = x.split('\n')[-1]
        else:
            x_ = x
            
        ret = float(x_.split()[0].split('(')[1].split(',')[0])
        return ret
    
    except:
        print('Error:',x)
        breakpoint()

def extract_long_crime(x):
    try:
        if '\n' in x:
            x_ = x.split('\n')[-1]
        else:
            x_ = x
            
        ret = float(x_.split(',')[1][:-1])
        return ret
    
    except:
        print('Error:',x)
        breakpoint()
        
        
def coord_bligth(row):
    if np.isnan(row['lat']) or np.isnan(row['lon']):
        return np.nan
    else:        
        return row['lat'],row['lon']
    
        
def crime_count(row, crimes_coord):
    """
    Calculate a number of crimes within 
    500 meters from place with blight ticket violetion
    
    params: 
            row : row in dataframe (2-tuple)
            crimes_coord : Series/array with crime coordinates (2-tuples)
    
    returns number of crimes
    """
    
    count = 0
    for crime in crimes_coord:
        dist = distance(row['coordinates'], crime).m
        if dist < 600:
            count+=1                 
    return count

def parse_parcel(row):
   # try:
        if '-' in row['parcel_id']:
            return int(re.split('-',row['parcel_id'])[0])
        if '.' in row['parcel_id']:
            return int(re.split('\.',row['parcel_id'])[0])
        else:
            return 0
#     except:
#         breakpoint()